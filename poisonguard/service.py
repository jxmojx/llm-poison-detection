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

import threading

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np

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
except Exception:  # pragma: no cover
    ScoreRequest = None  # type: ignore[assignment]


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


def create_app(cfg: Config | None = None, student_path: str | Path | None = None,
               scorer: PoisonScorer | None = None):
    """构造 FastAPI 应用（拆出来是为了测试可以注入现成的 scorer）。"""
    from fastapi import FastAPI, HTTPException

    cfg = cfg or load_config()
    state: dict = {}

    def _student() -> Path | None:
        return Path(student_path) if student_path else _default_student(cfg)

    app = FastAPI(title="poisonguard 在线打分服务", version="1.0")

    _build_lock = threading.Lock()

    def _scorer() -> PoisonScorer:
        """懒装配 + 加锁：并发首请求会同时触发 build_scorer（装配要几秒），
        不加锁时两个线程会各自 import transformers，先到的那个可能撞上
        半初始化的模块（实测过一次 ImportError: cannot import name 'AutoModel'）。"""
        if scorer is not None:
            return scorer
        if "s" not in state:
            with _build_lock:
                if "s" not in state:
                    sc = build_scorer(cfg, student_path=_student())
                    # P0-1：阈值口径统一（dev 干净集 1% 工作点），否则线上=0.5
                    try:
                        from .threshold import describe, resolve_threshold

                        res = resolve_threshold(sc, cfg, calibrate=True)
                        sc.threshold = float(res["threshold"])
                        print(f"[serve] 阈值口径：{describe(res)}")
                    except Exception as exc:  # 没有 processed 目录时保持默认
                        print(f"[serve] 警告: 阈值标定跳过({exc})")
                    state["s"] = sc
        return state["s"]

    @app.get("/health")
    def health() -> dict:
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
    def score(req: "ScoreRequest") -> dict:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts 不能为空")
        if len(req.texts) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"单次最多 {MAX_BATCH} 条")
        return _scorer().score(req.texts, explain=req.explain)

    @app.get("/stats")
    def stats() -> dict:
        return _scorer().stats()

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="poisonguard 在线打分服务")
    ap.add_argument("--config", default=None)
    ap.add_argument("--student", default=None, help="蒸馏学生模型路径(pkl)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    import uvicorn

    cfg = load_config(a.config)
    # 学生模型优先命令行指定，否则自动找：当前实验标签的 pkl → 任意 student_judge*.pkl
    student = a.student or _default_student(cfg)
    app = create_app(cfg, student_path=student)
    print(f"[serve] 阈值装配完成，监听 http://{a.host}:{a.port}"
          f"  (student={student})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
