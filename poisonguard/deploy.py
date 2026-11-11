"""poisonguard deploy：把训练产物打包成单文件发布物，离线评分不入域。

背景
----
在线(service)与批扫(scan)都需要"装配"：
    build_scorer(cfg) -> enc + knn + ngram + cal + student
而装配依赖 train/dev 原文本与编码器 —— 部署侧往往没有这些，也不应该把
用户原始语料发给下游。于是"发布物"应当包含的是**拟合态**而不是数据：
    - kNN 近邻索引(参考句向量，非原文)
    - 字符 n-gram 分类器 + 向量化器
    - 校准分布(参考分数)
    - 蒸馏学生模型
    - 该域运营阈值(学生阈值 + 判官 logprob 阈值)
打包成**单个 pickle(.pkg)**。运行时只需公开的编码器权重(公共模型)，
不需要原始文本、不需要 dev/test、不需要判官大模型 —— "数据不出域"成立。

两个命令(挂进 cli)：
    poisonguard deploy --domain afqmc            # 作者侧：打 .pkg
    poisonguard run --bundle x.pkg --input f.txt # 部署侧：离线批扫(复用 scan 报告)

一致性保证：
    - .pkg 记录 encoder.name 与 max_length，运行时必须用同一编码器重编码
      (换编码器 = 换特征空间，kNN / 学生就失效，故意在加载时校验并报错)。
    - JudgeStudent / CharNgramDetector / Calibration 均为纯 sklearn/标准库对象，
      可 pickle；`score()` 语义与 service.PoisonScorer.score 完全一致。
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .detectors import Calibration, CharNgramDetector, KNNDetector
from .service import build_scorer

PKG_VERSION = 1


# ----------------------------------------------------------------------
@dataclass
class DeployState:
    """在 .pkg 里实际落盘的内容(全部为拟合态, 不含原始文本)。"""

    version: int = PKG_VERSION
    domain: str = ""
    encoder: dict[str, Any] = field(default_factory=dict)   # {name, max_length, ...}
    detector: dict[str, Any] = field(default_factory=dict)  # {target_fpr, knn_k}
    threshold: float = 0.5
    judge_threshold: float = 0.5
    knn: KNNDetector | None = None
    ngram: CharNgramDetector | None = None
    cal: Calibration | None = None
    student: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeployedScorer:
    """轻量运行时评分器：与 service.PoisonScorer.score 同输出形状。"""

    state: DeployState
    enc: Any

    # 与 PoisonScorer 对齐：学生来自包内(state.student)，没有外部 pkl 路径
    student_path = "bundled"

    @property
    def version(self) -> str:
        return f"pkg-v{PKG_VERSION}/{self.state.domain}"

    @property
    def threshold(self) -> float:
        return self.state.threshold

    @property
    def has_student(self) -> bool:
        return self.state.student is not None

    @property
    def judge_threshold(self) -> float:
        return self.state.judge_threshold

    @property
    def domain(self) -> str:
        """包内领域名（报告口径用；部署机没有 processed_<域> 目录）。"""
        return self.state.domain or "bundled"

    @property
    def encoder_name(self) -> str:
        return (self.state.encoder or {}).get("name", "?")

    def score(self, texts: list[str], explain: bool = False) -> dict:
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return {"results": [], "n": 0, "latency_ms": 0.0,
                    "threshold": self.threshold, "model": self.version}
        t0 = time.perf_counter()
        X = self.enc.encode(texts)
        s = self.state
        knn = s.knn.score(X, texts) if s.knn else np.zeros(len(texts))
        fpr = s.cal.to_fpr(knn) if s.cal else np.full(len(texts), np.nan)
        ngram = s.ngram.score(X, texts) if s.ngram else np.full(len(texts), np.nan)
        stu = s.student.score(texts, X) if s.student else np.full(len(texts), np.nan)

        out = []
        for i, t in enumerate(texts):
            if np.isfinite(stu[i]):
                v = float(stu[i])
            elif np.isfinite(fpr[i]):
                v = float(1.0 - fpr[i])
            else:
                v = float(knn[i])
            rec = {"score": round(v, 6), "flag": bool(v > self.threshold)}
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
        return {
            "results": out, "n": len(out),
            "latency_ms": round(dt, 3),
            "ms_per_item": round(dt / max(len(out), 1), 4),
            "threshold": self.threshold,
            "model": self.version,
        }


# ----------------------------------------------------------------------
def _domain_tag(cfg: Config) -> str:
    """processed 目录后缀中的领域名：processed_afqmc -> afqmc / processed_real_v3 -> real_v3。

    注意与 student 文件名区分：学生模型是 `student_judge{tag}.pkl`（tag 含前导下划线），
    而 judge 指标与 .pkg 文件名用**去掉前导下划线**的领域名（judge_afqmc.json）。
    """
    tag = Path(cfg.data.processed_dir).name[len("processed"):]
    return tag.lstrip("_")


def _student_path_for(cfg: Config, fixed: str | Path | None = None) -> Path | None:
    mdir = Path(cfg.metrics_dir)
    tag = Path(cfg.data.processed_dir).name[len("processed"):]
    cands = []
    if fixed is not None:
        cands.append(Path(fixed))
    cands += [mdir / f"student_judge{tag}.pkl",
              *sorted(mdir.glob("student_judge*.pkl"))]
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None


def _load_judge_threshold(cfg: Config) -> float:
    mdir = Path(cfg.metrics_dir)
    domain = _domain_tag(cfg)
    for cand in (mdir / f"judge_{domain}.json", *sorted(mdir.glob("judge_*.json"))):
        if not cand.exists():
            continue
        try:
            j = json.loads(cand.read_text(encoding="utf-8"))
            for _name, det in (j.get("detectors") or {}).items():
                thr = det.get("threshold")
                if thr is not None:
                    return float(thr)
        except Exception:
            continue
    return 0.5


class PackedKNN:
    """量化后的 kNN 参考集：只存码字+缩放，运行时分块重建做余弦近邻。

    全精度 float32 参考向量是 .pkg 体积的主要来源（13807×768×4B ≈ 42MB）。
    - ``f16``：按行 half 存储，体积约减半，余弦距离误差 ~1e-4；
    - ``i8`` ：按行（L2 归一化后）对称量化，体积约 1/4，误差 ~1e-3。
    语义与 ``detectors.KNNDetector.score`` 相同：到最近 k 个干净样本的平均余弦距离。
    """

    name = "knn"

    def __init__(self, codes: np.ndarray, scale: np.ndarray, k: int,
                 dtype: str = "f16", chunk: int = 2048):
        self.codes = codes
        self.scale = scale
        self.k = int(k)
        self.dtype = dtype
        self.chunk = int(chunk)

    @classmethod
    def from_knn(cls, knn: KNNDetector, mode: str = "i8") -> "PackedKNN":
        X = np.asarray(knn.nn._fit_X, dtype=np.float32)
        k = int(getattr(knn, "k_eff", knn.k))
        if mode == "f16":
            return cls(X.astype(np.float16), np.float32(1.0), k, dtype="f16")
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        Xn = X / norm
        scale = (np.abs(Xn).max(axis=1, keepdims=True) / 127.0).astype(np.float32)
        scale[scale == 0] = np.float32(1e-8)
        codes = np.clip(np.rint(Xn / scale), -127, 127).astype(np.int8)
        return cls(codes, scale, k, dtype="i8")

    def _chunk(self, i: int) -> np.ndarray:
        blk = self.codes[i:i + self.chunk]
        if self.dtype == "f16":
            return np.asarray(blk, dtype=np.float32)
        sc = np.asarray(self.scale[i:i + self.chunk], dtype=np.float32)
        return np.asarray(blk, dtype=np.float32) * sc

    def score(self, X: np.ndarray, texts: list[str]) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Xn = X / nrm
        n = Xn.shape[0]
        k = max(1, min(self.k, len(self.codes)))
        best = np.full((n, 0), -np.inf, dtype=np.float32)
        for j in range(0, len(self.codes), self.chunk):
            sim = Xn @ self._chunk(j).T              # 余弦相似度
            merged = np.concatenate([best, sim], axis=1)
            kk = min(k, merged.shape[1])
            best = np.partition(merged, merged.shape[1] - kk, axis=1)[:, -kk:]
        return (1.0 - best).mean(axis=1).astype(np.float32)


def _quantize_knn(knn: KNNDetector | None, mode: str) -> Any:
    """按 mode 把 kNN 参考集压成 f16/int8 码字（none 原样返回）。"""
    if knn is None or mode in ("", "none", None):
        return knn
    if mode not in ("f16", "i8"):
        raise ValueError(f"未知量化模式: {mode}（可选 none/f16/i8）")
    return PackedKNN.from_knn(knn, mode=mode)


def bundle_scorer(cfg: Config, out_path: str | Path | None = None,
                  student_path: str | Path | None = None,
                  threshold: float | None = None,
                  quantize: str = "none") -> Path:
    """作者侧：装配该域打分器 → 提取拟合态 → 落单文件 .pkg。"""
    scorer = build_scorer(cfg, student_path=_student_path_for(cfg, student_path))
    if threshold is not None:
        scorer.threshold = float(threshold)

    domain = _domain_tag(cfg)
    state = DeployState(
        domain=domain,
        encoder={
            "name": cfg.encoder.name,
            "max_length": cfg.encoder.max_length,
            "pooling": cfg.encoder.pooling if hasattr(cfg.encoder, "pooling") else "mean",
        },
        detector={"target_fpr": cfg.detector.target_fpr, "knn_k": cfg.detector.knn_k},
        threshold=float(scorer.threshold),
        judge_threshold=_load_judge_threshold(cfg),
        knn=_quantize_knn(scorer.knn, quantize),
        ngram=scorer.ngram,
        cal=scorer.cal,
        student=scorer.student,
        provenance={
            "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "encoder": cfg.encoder.name,
            "has_student": scorer.student is not None,
            "has_ngram": scorer.ngram is not None,
            "has_knn": scorer.knn is not None,
            "quantize": quantize if quantize not in ("", None) else "none",
            "knn_refs": int(len(getattr(scorer.knn, "nn", None)._fit_X))
            if getattr(scorer.knn, "nn", None) is not None else 0,
        },
    )
    if out_path is None:
        out_path = Path("outputs") / "deploy" / f"poisonguard-{domain or 'default'}.pkg"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 容器格式：payload(拟合态 pickle) + sha256 + manifest → 部署侧可校验完整性
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(payload).hexdigest()
    container = {
        "format": "poisonguard.pkg",
        "version": PKG_VERSION,
        "sha256": digest,
        "bytes": len(payload),
        "manifest": {
            "domain": domain,
            "encoder": cfg.encoder.name,
            "max_length": cfg.encoder.max_length,
            "quantize": state.provenance.get("quantize", "none"),
            "knn_refs": state.provenance.get("knn_refs", 0),
            "threshold": state.threshold,
            "judge_threshold": state.judge_threshold,
            "created": state.provenance.get("created"),
        },
        "payload": payload,
    }
    with open(out_path, "wb") as f:
        pickle.dump(container, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[deploy] {domain} 拟合态已打包 -> {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB, 含原始文本: no, "
          f"量化: {container['manifest']['quantize']})")
    print(f"[deploy] sha256={digest[:16]}…  payload={len(payload) / 1024:.0f} KB  "
          f"knn_refs={container['manifest']['knn_refs']}")
    return out_path


def load_bundle(path: str | Path, cfg: Config | None = None) -> DeployedScorer:
    """部署侧：加载 .pkg 并重建编码器(必须是打包容器的同一编码器)。"""
    from .encoders import get_encoder

    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    manifest: dict[str, Any] = {}
    digest = ""
    if isinstance(obj, dict) and obj.get("format") == "poisonguard.pkg":
        payload = obj.get("payload") or b""
        want = str(obj.get("sha256") or "")
        got = hashlib.sha256(payload).hexdigest()
        if want and got != want:
            raise ValueError(
                f".pkg 完整性校验失败: sha256 {got[:16]}… != {want[:16]}…"
                "（文件被截断或篡改，拒绝加载）")
        manifest = obj.get("manifest") or {}
        digest = got
        state: DeployState = pickle.loads(payload)
    else:  # 兼容旧格式：裸 DeployState（无 sha256/manifest）
        state = obj
        print("[run] 旧格式 .pkg（无 sha256/manifest），建议重新 deploy 以启用校验")
    if getattr(state, "version", 0) != PKG_VERSION:
        raise ValueError(f".pkg 版本不兼容: 包内 {getattr(state,'version',0)} vs 运行时 {PKG_VERSION}")

    base = cfg or Config()
    enc_spec = state.encoder or {}
    name = enc_spec.get("name", base.encoder.name)
    if name != base.encoder.name:
        # 换编码器 = 换特征空间，kNN/学生全部失效 → 必须同编码器
        base.encoder.name = name
    if enc_spec.get("max_length"):
        base.encoder.max_length = int(enc_spec["max_length"])
    try:
        enc = get_encoder(base)
    except FileNotFoundError:
        raise FileNotFoundError(
            "部署机缺少编码器权重(公共模型)。请用 `python -m poisonguard.cli "
            "download-encoder --encoder crom` 下载后重试。"
        )
    sc = DeployedScorer(state, enc)
    quant = manifest.get("quantize") or state.provenance.get("quantize", "none")
    print(f"[run] 发布物校验通过: domain={manifest.get('domain') or state.domain} "
          f"quantize={quant} "
          f"sha256={(digest[:16] + '…') if digest else '(legacy, 无校验)'}")
    return sc


# ----------------------------------------------------------------------
def cmd_deploy(cfg: Config, args: argparse.Namespace) -> Path:
    return bundle_scorer(cfg, out_path=getattr(args, "out", None),
                         student_path=getattr(args, "student", None),
                         threshold=getattr(args, "threshold", None),
                         quantize=getattr(args, "quantize", "none") or "none")


def cmd_run(cfg: Config, args: argparse.Namespace) -> dict:
    """部署侧批扫：加载 .pkg → 复用 scan 报告管线（注入 scorer，不重新装配）。"""
    if not getattr(args, "bundle", None):
        print("[run] 需要 --bundle path.pkg")
        import sys
        sys.exit(2)
    from .scan import scan_file

    scorer = load_bundle(args.bundle, cfg)
    result = scan_file(cfg, args.input, threshold=None,
                       judge_topk=getattr(args, "judge_topk", 0),
                       fmt=getattr(args, "format", "auto"),
                       column=getattr(args, "column", "text"),
                       limit=getattr(args, "limit", 0),
                       render_html=not getattr(args, "no_html", False),
                       out_prefix=getattr(args, "out_prefix", None),
                       scorer=scorer)
    return result


if __name__ == "__main__":
    import sys
    from .config import load_config

    ap = argparse.ArgumentParser(description="poisonguard 部署打包/离线评分")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("deploy", help="把训练产物打成 .pkg")
    p1.add_argument("--config", default=None)
    p1.add_argument("--domain", default="", help="processed 目录后缀(如 afqmc)；空=默认 v3")
    p1.add_argument("--student", default=None)
    p1.add_argument("--threshold", type=float, default=None)
    p1.add_argument("--out", default=None)
    p1.add_argument("--quantize", default="none", choices=["none", "f16", "i8"],
                    help="kNN 参考向量量化(none/f16/i8)，用于压缩发布物体积")
    p2 = sub.add_parser("run", help="加载 .pkg 离线批扫")
    p2.add_argument("--config", default=None)
    p2.add_argument("--bundle", required=True)
    p2.add_argument("--input", required=True)
    p2.add_argument("--format", default="auto")
    p2.add_argument("--column", default="text")
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--judge-topk", type=int, default=0)
    p2.add_argument("--out-prefix", default=None)
    p2.add_argument("--no-html", action="store_true")
    a = ap.parse_args(sys.argv[1:])
    cfg = load_config(a.config)
    if getattr(a, "domain", ""):
        cfg.data.processed_dir = str(
            Path(cfg.data.processed_dir).with_name(f"processed_{a.domain}"))
    (cmd_deploy if a.cmd == "deploy" else cmd_run)(cfg, a)
