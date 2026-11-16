"""在线打分服务：把三层检测中"能在线跑"的部分包成 HTTP 接口。

定位
----
数据层(kNN / 字符 n-gram)和蒸馏学生模型都能在 CPU 上毫秒级完成，
所以可以放在数据入库的同步路径上；LLM 判官不进在线路径，
它只在离线阶段产出软标签和抽查报告。

接口
----
* `GET  /health`  存活 + 当前模型版本、阈值、可用组件
* `POST /score`   `{"texts": ["..."], "explain": true}` → 逐条分数、是否拦截、各视图明细
* `GET  /stats`   调用量、延迟分位数（接监控用）
* `POST /batch`   `{"texts": [...]}` → 提交异步批量任务，立即返回 job_id（不阻塞）
* `GET  /batch/{job_id}`  轮询批量任务状态/结果（queued/running/done/failed）
* `GET  /admin/models` `POST /admin/reload` `POST /admin/switch`
  `POST /admin/rollback`  模型版本热管理（方向③缺口2）

起服务
------
    python -m poisonguard.service --config configs/default.yaml --port 8000

设计取舍
--------
* 阈值来自**干净 dev 集**的 FPR 分位（和数据层同一套口径），不在服务里拍 0.5；
* `score` 用连续分，`flag` 只是它的阈值化视图，前端不必自己比大小；
* 每个视图的原始分都返回（`explain`），否则线上误报了无法归因。
"""

from __future__ import annotations

import logging
import threading

import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np

from .log import event, get_logger
from .config import Config, load_config
from .detectors import KNNDetector, Calibration, CharNgramDetector
from .encoders import get_encoder

MAX_BATCH = 512

try:  # 没装 fastapi 时也要能 import 本模块（只影响服务，不影响检测流水线）
    from pydantic import BaseModel, Field

    class ScoreRequest(BaseModel):
        # 批量上限在端点里判（返回 413 而不是 pydantic 的 422），前端好区分"太快"和"太多"
        texts: list[str] = Field(default_factory=list)
        explain: bool = False

    class SwitchIn(BaseModel):
        """热切换请求体：version 为空 = 重建一个新版本并切换过去。"""
        version: str = ""

    class BatchIn(BaseModel):
        """异步批量打分：texts 可超 MAX_BATCH，服务端切块排队执行。"""
        texts: list[str] = Field(default_factory=list)
        explain: bool = False
except Exception:  # pragma: no cover
    ScoreRequest = None  # type: ignore[assignment]
    SwitchIn = None  # type: ignore[assignment]
    BatchIn = None  # type: ignore[assignment]


class PoisonScorer:
    """在线两段式打分器：语义近似(kNN) + 字面自然度(n-gram) + 蒸馏学生(可选)。"""

    version = "v3-online"

    def __init__(self, cfg: Config, student_path: str | Path | None = None):
        self.cfg = cfg
        self.enc = get_encoder(cfg)
        self.knn: KNNDetector | None = None
        self.ngram: CharNgramDetector | None = None
        self.cal: Calibration | None = None
        self.student = None
        self.student_path = student_path
        self.threshold = 0.5
        self.created = time.time()
        self.n_calls = 0
        self.n_texts = 0
        self.latencies: deque[float] = deque(maxlen=500)

    # ---------------- 装配 ----------------
    def fit(self, train_texts: list[str], clean_texts: list[str],
            labeled: tuple[list[str], np.ndarray] | None = None) -> "PoisonScorer":
        """train_texts 只有干净样本；带标注的攻击样本另给(labeled)。

        词面分类器必须有正类才能训练，所以不能拿 train 单独喂它——
        这一点在 v3 尤其明显：真实语料口径下训练集是**纯干净**的。
        """
        Xtr = self.enc.encode(train_texts)
        self.knn = KNNDetector(self.cfg.detector.knn_k).fit(Xtr, train_texts)
        if labeled is not None and len(labeled[0]) and len(set(labeled[1].tolist())) > 1:
            ltexts, ly = labeled
            self.ngram = CharNgramDetector().fit_labeled(list(ltexts), np.asarray(ly))
        if clean_texts:
            clean_scores = self.knn.score(self.enc.encode(clean_texts), clean_texts)
            self.cal = Calibration.fit(clean_scores, self.cfg.detector.target_fpr)
        if self.student_path and Path(self.student_path).exists():
            from .distill import JudgeStudent
            self.student = JudgeStudent.load(self.student_path)
        return self

    def score(self, texts: list[str], explain: bool = False) -> dict:
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return {"results": [], "n": 0, "latency_ms": 0.0,
                    "threshold": self.threshold, "model": self.version}
        t0 = time.perf_counter()
        X = self.enc.encode(texts)
        knn = self.knn.score(X, texts) if self.knn else np.zeros(len(texts))
        # kNN 原始距离分对运维不可读，转成"经验 FPR"更直观：越小越可疑
        fpr = self.cal.to_fpr(knn) if self.cal else np.full(len(texts), np.nan)
        ngram = self.ngram.score(X, texts) if self.ngram else np.full(len(texts), np.nan)
        stu = self.student.score(texts, X) if self.student else np.full(len(texts), np.nan)

        out = []
        for i, t in enumerate(texts):
            # 最终分优先用学生模型的概率(它就是判官的替身)，
            # 没有学生模型时退化为 1-FPR（越大越可疑，语义和其他视图对齐）
            if np.isfinite(stu[i]):
                s = float(stu[i])
            elif np.isfinite(fpr[i]):
                s = float(1.0 - fpr[i])
            else:
                s = float(knn[i])
            rec = {"score": round(s, 6), "flag": bool(s > self.threshold)}
            if explain:
                rec["views"] = {
                    "knn": round(float(knn[i]), 6),
                    "fpr_est": None if not np.isfinite(fpr[i]) else round(float(fpr[i]), 6),
                    "char_ngram": None if not np.isfinite(ngram[i]) else round(float(ngram[i]), 6),
                    "student": None if not np.isfinite(stu[i]) else round(float(stu[i]), 6),
                    "chars": len(t),
                }
            out.append(rec)

        dt = (time.perf_counter() - t0) * 1000.0
        self.n_calls += 1
        self.n_texts += len(texts)
        self.latencies.append(dt / max(len(texts), 1))
        return {
            "results": out,
            "n": len(texts),
            "latency_ms": round(dt, 3),
            "ms_per_item": round(dt / max(len(texts), 1), 4),
            "threshold": self.threshold,
            "model": self.version,
        }

    def stats(self) -> dict:
        lat = np.array(self.latencies, dtype=float) if self.latencies else np.array([0.0])
        return {
            "uptime_s": round(time.time() - self.created, 1),
            "n_calls": self.n_calls,
            "n_texts": self.n_texts,
            "ms_per_item_p50": round(float(np.percentile(lat, 50)), 4),
            "ms_per_item_p99": round(float(np.percentile(lat, 99)), 4),
            "threshold": self.threshold,
            "model": self.version,
        }


# ----------------------------------------------------------------------
def build_scorer(cfg: Config, student_path: str | Path | None = None,
                 splits=None) -> PoisonScorer:
    """从已构建的数据集装配打分器；数据不存在时给出可执行的下一步。"""
    from . import data as D

    splits = splits or D.load_splits(cfg.data.processed_dir)
    train = splits["train"]
    dev = splits["dev"]
    scorer = PoisonScorer(cfg, student_path=student_path)
    clean = [s.text for s in dev if s.label == 0 and s.family in ("normal", "hardneg")]
    scorer.fit([s.text for s in train], clean,
               labeled=([s.text for s in dev], np.array([s.label for s in dev])))
    return scorer


def _default_student(cfg: Config) -> Path | None:
    """找蒸馏学生模型：优先当前实验标签，其次任意一份。"""
    mdir = Path(cfg.metrics_dir)
    tag = Path(cfg.data.processed_dir).name[len("processed"):]
    for cand in (mdir / f"student_judge{tag}.pkl", *sorted(mdir.glob("student_judge*.pkl"))):
        if Path(cand).exists():
            return Path(cand)
    return None


LOG = get_logger("service")


def create_app(cfg: Config | None = None, student_path: str | Path | None = None,
               scorer: PoisonScorer | None = None, security=None,
               scorer_factory=None):
    """构造 FastAPI 应用（拆出来是为了测试可以注入现成的 scorer）。

    `security`：`poisonguard.security.Security` 实例；默认 None = 无鉴权开发模式
    （保持既有行为，也不落文件审计），上公网/跨团队时由 CLI 显式构造传入。

    `scorer_factory`：`callable() -> PoisonScorer`，用于装配新打分器实例
    （方向③缺口2 热更新/版本切换的测试钩子；默认走 build_scorer + 阈值标定）。
    """
    from fastapi import Depends, FastAPI, HTTPException
    from .security import Security, SecurityConfig

    cfg = cfg or load_config()
    if security is None:
        # 测试/默认路径：无鉴权（保持既有行为），也不落文件审计（避免测试副作用）
        security = Security(SecurityConfig(audit_enabled=False))
    state: dict = {}

    def _student() -> Path | None:
        return Path(student_path) if student_path else _default_student(cfg)

    app = FastAPI(title="poisonguard 在线打分服务", version="2.0")

    _build_lock = threading.Lock()
    _versions: dict[str, PoisonScorer] = {}
    _active: str | None = None
    _history: list[str] = []  # 近 N 个 active 版本名（滚回上上版用）

    def _factory() -> PoisonScorer:
        if scorer_factory is not None:
            return scorer_factory()
        return build_scorer(cfg, student_path=_student())

    def _recalibrate(sc: PoisonScorer) -> PoisonScorer:
        # P0-1：阈值口径统一（dev 干净集 1% 工作点），否则线上=0.5
        try:
            from .threshold import describe, resolve_threshold

            res = resolve_threshold(sc, cfg, calibrate=True)
            sc.threshold = float(res["threshold"])
            print(f"[serve] 阈值口径：{describe(res)}")
        except Exception as exc:  # 没有 processed 目录时保持默认 / 注入的测试 scorer
            print(f"[serve] 警告: 阈值标定跳过({exc})")
        return sc

    def _scorer() -> PoisonScorer:
        """懒装配 + 加锁：并发首请求会同时触发 build_scorer（装配要几秒），
        不加锁时两个线程会各自 import transformers，先到的那个可能撞上
        半初始化的模块（实测过一次 ImportError: cannot import name 'AutoModel'）。

        registry 里的版本号从 scorer.version 取；重名时拼 `#n` 递增后缀，
        保证每个快照在 registry 里都有唯一 key 可被 switch/回滚。
        """
        if scorer is not None:
            # 注入的测试 scorer：不参与版本管理（就一个，直接用）
            return scorer
        nonlocal _active
        if _active is None:
            with _build_lock:
                if _active is None:
                    sc = _recalibrate(_factory())
                    _register(sc.version, sc)
                    _active = list(_versions)[-1]
        return _versions[_active]

    def _register(version: str, sc: PoisonScorer) -> str:
        key = version
        n = 2
        while key in _versions:
            key = f"{version}#{n}"
            n += 1
        _versions[key] = sc
        return key

    def _switch_to(key: str, *, build: bool = False, push_history: bool = True,
                   via: str = "/admin/switch") -> dict:
        nonlocal _active
        with _build_lock:
            old = _active
            if build or key not in _versions:
                sc = _recalibrate(_factory())
                key = _register(getattr(sc, "version", "rebuilt"), sc)
            if push_history and old:
                _history.append(old)
            _active = key
            event(LOG, "model.switch", to=key, from_=old,
                  n_versions=len(_versions))
            security.audit.record(
                action="model.switch", method="POST", path=via,
                status=200, detail=f"from={old} to={key}")
            return {
                "ok": True,
                "active": key,
                "model": _versions[key].version,
                "from": old,
                "n_versions": len(_versions),
            }

    # ---- 方向③缺口2：模型版本热管理 --------------------------------
    @app.get("/admin/models")
    def admin_models(_=Depends(security.auth(admin=True))) -> dict:
        s = _scorer()
        return {
            "active": _active,
            "model": s.version,
            "versions": [{"key": k, "model": v.version,
                          "threshold": getattr(v, "threshold", None),
                          "active": (k == _active)}
                         for k, v in _versions.items()],
            "student_path": str(_student()),
        }

    @app.post("/admin/reload")
    def admin_reload(_=Depends(security.auth(admin=True)),
                     __=Depends(security.admin_limited())) -> dict:
        """从磁盘重新装配当前模型（换数据/换学生模型后热生效），失败回滚旧版。"""
        nonlocal _active
        old = _active
        try:
            sc = _recalibrate(_factory())
            with _build_lock:
                _history.append(old)
                key = _register(sc.version, sc)
                _active = key
                event(LOG, "model.reload", to=key, from_=old)
                security.audit.record(
                    action="model.reload", method="POST", path="/admin/reload",
                    status=200, detail=f"from={old} to={key}")
                return {"ok": True, "active": key, "model": sc.version,
                        "from": old}
        except Exception as exc:
            event(LOG, "model.reload_error", level=logging.ERROR,
                  err=type(exc).__name__, msg=str(exc))
            raise HTTPException(status_code=500,
                                detail=f"重载失败，保持旧模型: {exc}")

    @app.post("/admin/switch")
    def admin_switch(req: SwitchIn,
                     _=Depends(security.auth(admin=True)),
                     __=Depends(security.admin_limited())) -> dict:
        """切换到 registry 里的另一个已装配版本（version 为空=重建新的）。"""
        if req.version:
            if req.version not in _versions:
                raise HTTPException(status_code=404,
                                    detail=f"未知版本 {req.version!r}；"
                                           f"可用: {sorted(_versions)}")
            return _switch_to(req.version)
        return _switch_to("", build=True)

    @app.post("/admin/rollback")
    def admin_rollback(_=Depends(security.auth(admin=True)),
                       __=Depends(security.admin_limited())) -> dict:
        """回滚到上一个 active 版本（热更新翻车时一键还原）。"""
        if not _history:
            raise HTTPException(status_code=409, detail="没有可回滚的历史")
        target = _history.pop()
        if target not in _versions:
            raise HTTPException(status_code=404, detail=f"历史版本 {target!r} 已不存在")
        return _switch_to(target, push_history=False, via="/admin/rollback")

    # ---- 方向③缺口3：异步批量打分 ---------------------------------
    # 一次性大批量（>512）走"提交即回"的队列：不占请求线程，判定产出在后台执行。
    # 无第三方队列依赖：进程内限长队列 + 固定 worker 线程池（多实例下游再挂 MQ）。
    _batch: dict[str, dict] = {}          # job_id -> 任务记录
    _batch_order: deque[str] = deque()    # FIFO 顺序（用于淘汰最老任务）
    _batch_max_jobs = 128                 # 同时驻留任务数上限
    _batch_running = 0                    # 正在执行的 worker 数（节流用）
    _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="batch")

    def _batch_submit(texts: list[str], explain: bool) -> str:
        """登记任务并调度 worker；返回 job_id。"""
        nonlocal _batch_running
        # 容量保护：超出时淘汰最老且未运行的任务
        while len(_batch) >= _batch_max_jobs:
            old = _batch_order.popleft()
            if _batch[old]["state"] not in ("queued", "running"):
                del _batch[old]
                continue
            _batch[old]["state"] = "cancelled"
            _batch[old]["error"] = "任务过饱和，已被淘汰"
            del _batch[old]  # cancelled 也不占用（poll 409 说明被淘汰）
            if not _batch_order:
                break
        job = {"job_id": "", "state": "queued", "explain": explain,
               "total": len(texts), "done": 0, "results": [], "error": "",
               "created": time.time()}
        job_id = uuid.uuid4().hex[:12]
        job["job_id"] = job_id
        _batch[job_id] = job
        _batch_order.append(job_id)

        def _run():
            nonlocal _batch_running
            _batch_running += 1
            try:
                sc = _scorer()   # 惰性装配（同步路径建好也可能被 admin 切走）
                out = []
                # 切块：落下层 MAX_BATCH 上限，也避免单个大 list 占内存
                for i in range(0, len(texts), MAX_BATCH):
                    chunk = texts[i:i + MAX_BATCH]
                    res = sc.score(chunk, explain=explain)
                    out.extend(res.get("results", []))
                    job["done"] = len(out)
                    job["state"] = "running"
                job["results"] = out
                job["state"] = "done"
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["state"] = "failed"
            finally:
                _batch_running -= 1

        _executor.submit(_run)
        return job_id

    @app.post("/batch")
    def batch_submit(req: "BatchIn",
                     identity=Depends(security.auth()),
                     _=Depends(security.limited())) -> dict:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空")
        if _batch_running >= 2 * _batch_max_jobs:  # 防御：worker 全忙也限总驻留
            raise HTTPException(status_code=503, detail="批处理队列已满，请稍后再试")
        job_id = _batch_submit(req.texts, req.explain)
        event(LOG, "batch.submit", job=job_id, n=len(req.texts))
        security.audit.record(
            action="batch.submit", method="POST", path="/batch", status=200,
            api_key_id=identity[1], client_ip=identity[0],
            n=len(req.texts), ms=0.0)
        return {"job_id": job_id, "status": "queued", "total": len(req.texts)}

    @app.get("/batch/{job_id}")
    def batch_status(job_id: str,
                     identity=Depends(security.auth()),
                     _=Depends(security.limited())) -> dict:
        job = _batch.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="未知 job_id（可能已过期淘汰）")
        resp = {"job_id": job["job_id"], "state": job["state"],
                "total": job["total"], "done": job["done"]}
        if job["state"] == "done":
            resp["results"] = job["results"]
            resp["threshold"] = getattr(_scorer(), "threshold", None)
        if job["error"]:
            resp["error"] = job["error"]
        return resp

    @app.get("/batch/")
    def batch_list(identity=Depends(security.auth()),
                   _=Depends(security.limited())) -> dict:
        """列出驻留任务（状态/进度），管理端排障用。"""
        return {"running": _batch_running,
                "jobs": [{"job_id": j["job_id"], "state": j["state"],
                          "total": j["total"], "done": j["done"]}
                         for j in _batch.values()]}

    # 请求审计：无论成败都落一条（鉴权/限流中间的异常也尽量记）
    @app.middleware("http")
    async def _audit_mw(request, call_next):
        t0 = time.perf_counter()
        status = 500
        try:
            resp = await call_next(request)
            status = resp.status_code
            return resp
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            security.audit.record(
                action="http.request",
                method=request.method,
                path=request.url.path,
                status=status,
                api_key_id=request.headers.get("X-API-Key", ""),
                client_ip=request.client.host if request.client else "",
                n=0, ms=ms,
                detail="")

    @app.get("/health")
    def health(identity=Depends(security.auth())) -> dict:
        s = _scorer()
        return {
            "status": "ok",
            "model": s.version,
            "encoder": cfg.encoder.name,
            "has_knn": s.knn is not None,
            "has_char_ngram": s.ngram is not None,
            "has_student": s.student is not None,
            "threshold": s.threshold,
            "target_fpr": cfg.detector.target_fpr,
        }

    @app.post("/score")
    def score(req: "ScoreRequest",
              identity=Depends(security.auth()),
              _=Depends(security.limited())) -> dict:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空")
        if len(req.texts) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"单次最多 {MAX_BATCH} 条")
        t0 = time.perf_counter()
        try:
            res = _scorer().score(req.texts, explain=req.explain)
        except Exception as exc:  # 记下来再抛，否则线上排障只能靠猜
            event(LOG, "score.error", level=logging.ERROR,
                  n=len(req.texts), err=type(exc).__name__)
            raise
        n_flag = sum(1 for r in res.get("results", []) if r.get("flag"))
        event(LOG, "score.request", n=len(req.texts), n_flag=n_flag,
              ms=round((time.perf_counter() - t0) * 1000, 1),
              threshold=res.get("threshold"))
        security.audit.record(
            action="score.request", method="POST", path="/score", status=200,
            api_key_id=identity[1], client_ip=identity[0],
            n=len(req.texts), ms=round((time.perf_counter() - t0) * 1000, 1))
        return res

    @app.get("/stats")
    def stats(identity=Depends(security.auth())) -> dict:
        return _scorer().stats()

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="poisonguard 在线打分服务")
    ap.add_argument("--config", default=None)
    ap.add_argument("--student", default=None, help="蒸馏学生模型路径(pkl)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    # 方向③ 缺口1：鉴权/限流/审计
    ap.add_argument("--api-key", action="append", default=None,
                    help="API Key（可多次传，多把 key 同时有效）")
    ap.add_argument("--admin-key", action="append", default=None,
                    help="管理 Key（热更新/管理端点专用，可多次传）")
    ap.add_argument("--auth-disabled", action="store_true",
                    help="无任何 key 时默认关闭鉴权（内网直连）；给了 key 自动开启")
    ap.add_argument("--rate-rps", type=float, default=None, help="每秒速率(默认20)")
    ap.add_argument("--rate-burst", type=float, default=None, help="突发容量(默认60)")
    ap.add_argument("--audit", default=None,
                    help="审计 sqlite 路径(默认 outputs/audit/audit.sqlite3)")
    a = ap.parse_args()
    import uvicorn

    from .security import Security, SecurityConfig

    cfg = load_config(a.config)
    # 学生模型优先命令行指定，否则自动找：当前实验标签的 pkl → 任意 student_judge*.pkl
    student = a.student or _default_student(cfg)
    sec_cfg = SecurityConfig.from_cli(
        api_keys=a.api_key, admin_keys=a.admin_key,
        rate_rps=a.rate_rps, rate_burst=a.rate_burst, audit_path=a.audit)
    if a.auth_disabled:
        # 显式 --auth-disabled：即使给了 key 也强制关（本地/压测场景）
        sec_cfg.disabled = True
        sec_cfg.api_keys = None
        sec_cfg.admin_keys = None
    security = Security(sec_cfg)
    app = create_app(cfg, student_path=student, security=security)
    mode = "鉴权关闭(内网/开发)" if sec_cfg.disabled else \
        f"鉴权开启(API Keys={len(security.keys.count() or 0)})"
    print(f"[serve] 阈值装配完成，监听 http://{a.host}:{a.port}"
          f"  (student={student}, {mode})")
    if not sec_cfg.disabled:
        loc = sec_cfg.audit_path or "outputs/audit/audit.sqlite3"
        print(f"[serve] 审计日志: {loc}  (rate={sec_cfg.rate_rps}rps"
              f" burst={sec_cfg.rate_burst})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
