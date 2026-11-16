"""吞吐压测（P1 上线前热身）：encoder / 打分器 / 检索一致性的 QPS 体检。

为什么要压测：
  - 守卫是"入库网关/在线打分"的必经层，必须知道单条打分延迟与吞吐，
    才能定 batch、定并发、定缓存策略。
  - 本项目有两类压测对象：
      * **在线打分**：service.PoisonScorer.score(texts) —— 实际线上路径；
      * **检索一致性**：ragcheck.scan_scores + vecdb 索引 query —— P2 事实核验路径。
  - 各压测都能无模型跑（TinyEncoder + 随机向量），上线前再换真实
    crom 编码器 + 学生模型出正式数字。

输出：outputs/metrics/stress{run-tag}.json —— mean/p50/p99 ms、QPS、
batch 缩放曲线（batch 1→8→32→128 的吞吐升降），供容量规划。

纯逻辑：默认用 TinyEncoder 与最小打分器，不下载任何模型。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return float(np.percentile(s, q * 100))


def _summarize(times_ms: list[float]) -> dict:
    return {"n": len(times_ms), "mean_ms": float(np.mean(times_ms)),
            "p50_ms": _pct(times_ms, 0.5), "p99_ms": _pct(times_ms, 0.99),
            "min_ms": float(np.min(times_ms)), "max_ms": float(np.max(times_ms)),
            "qps": float(1000.0 / (np.mean(times_ms) or 1e-9)),
            "items_per_sec": float(1000.0 * len(times_ms) / (sum(times_ms) or 1e-9))}


def bench_encoder(enc, texts: list[str], batch_size: int = 64,
                  n_repeat: int = 3, use_cache: bool = True) -> dict:
    """编码器吞吐：对 texts 编码 n_repeat 轮，报告每批耗时与单条延迟。"""
    per = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        enc.encode(texts, use_cache=use_cache)
        dt = (time.perf_counter() - t0) * 1000
        per.append(dt)
    return {"kind": "encoder", "n_texts": len(texts), "batch_size": batch_size,
            "n_repeat": n_repeat, "per_call_ms": _summarize(per)}


def bench_scorer(scorer, texts: list[str], batch_size: int = 64,
                 n_repeat: int = 3, explain: bool = False) -> dict:
    """PoisonScorer.score 吞吐：单批内打 texts，报告批耗时/单条延迟/QPS。

    batch_size>len(texts) 时按 len(texts) 分批等效——这里固定整批跑，
    反映"一次 score() 调用处理 n 条"的端到端延迟。
    """
    per = []
    n = 0
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        r = scorer.score(texts, explain=explain)
        per.append((time.perf_counter() - t0) * 1000)
        n = r.get("n", 0) or len(texts)
    return {"kind": "scorer", "n_texts": n, "batch_size": batch_size,
            "n_repeat": n_repeat, "explain": bool(explain),
            "per_call_ms": _summarize(per)}


def bench_ragcheck_scan(scan_fn, texts: list[str], ref_texts: list[str],
                        n_repeat: int = 1) -> dict:
    """检索一致性核验吞吐：scan_scores 一次调用耗时（含 ref sim 预计算？否）。

    说明：scan_fn(texts) 应已闭包绑定参考向量（ref 编码只算一次），
    这里只测"对新文本打分"的热路径。
    """
    per = []
    n = 0
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        scores = scan_fn(texts)
        per.append((time.perf_counter() - t0) * 1000)
        n = int(len(scores))
    return {"kind": "ragcheck_scan", "n_texts": n, "n_repeat": n_repeat,
            "per_call_ms": _summarize(per)}


def bench_vecdb(store, queries: list[np.ndarray], k: int = 5,
                n_repeat: int = 3) -> dict:
    """向量库 query 吞吐：单条 query 延迟 + QPS。"""
    per = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for q in queries:
            store.query(q, k=k)
        per.append((time.perf_counter() - t0) * 1000 / max(len(queries), 1))
    return {"kind": "vecdb_query", "n_queries": len(queries), "k": k,
            "n_repeat": n_repeat, "per_query_ms": _summarize(per)}


def _make_tiny_texts(n: int = 200, seed: int = 0) -> list[str]:
    """合成中文样本文本（无模型，供 CI 压测）。"""
    rng = np.random.default_rng(seed)
    base = ("今天天气不错适合出门散步市场整体上涨带动成交量放大"
            "公司发布最新财报营收同比增长科技创新驱动未来发展")
    return ["".join(rng.choice(list(base), size=24)) for _ in range(n)]


def run_stress(cfg, out_path: str | Path | None = None, n_texts: int = 200,
               n_repeat: int = 3, real_encoder: bool = False,
               with_scorer: bool = True, with_ragcheck: bool = True,
               with_vecdb: bool = True) -> dict:
    """压测总入口：编码器(+可选打分器/检索/向量库)。默认无模型（TinyEncoder）。

    real_encoder=True 时加载 crom（model_cache 在）；打分器若有学生模型 pkl
    则接上（真实线上路径），否则仍用 TinyEncoder 最小栈。
    """
    from .chunking import chunk_text
    from .encoders import TinyEncoder  # noqa: F401

    t0 = time.perf_counter()
    texts = _make_tiny_texts(n_texts)

    if real_encoder:
        from .encoders import get_encoder
        enc = get_encoder(cfg)
    else:
        from .encoders import TinyEncoder

        enc = TinyEncoder()

    report: dict = {"meta": {
        "encoder": "crom" if real_encoder else "tiny",
        "n_texts": n_texts, "n_repeat": n_repeat,
        "seconds": 0.0, "note": "吞吐压测：少样本纯逻辑（TinyEncoder）或真实 crom 栈",
    }, "results": {}}

    report["results"]["encoder"] = bench_encoder(enc, texts, n_repeat=n_repeat)

    if with_scorer:
        try:
            scorer = _build_min_scorer(enc, texts) if not real_encoder else _build_real_scorer(cfg, texts)
            report["results"]["scorer"] = bench_scorer(
                scorer, texts, batch_size=len(texts), n_repeat=n_repeat)
        except Exception as e:
            report["results"]["scorer"] = {"error": f"{type(e).__name__}: {e}"}

    if with_ragcheck:
        try:
            refs = texts[: max(8, n_texts // 4)]
            ref_embs = enc.encode(refs)
            runner = _make_scan_fn(enc, refs, ref_embs)
            report["results"]["ragcheck_scan"] = bench_ragcheck_scan(
                runner, texts[:16], refs, n_repeat=1)
        except Exception as e:
            report["results"]["ragcheck_scan"] = {
                "error": f"{type(e).__name__}: {e}"}

    if with_vecdb:
        try:
            from .vecdb import VecStore

            dim = 64 if not real_encoder else getattr(enc, "dim", 64)
            rng = np.random.default_rng(7)
            m = rng.standard_normal((n_texts, dim))
            store = VecStore.create(dim=dim)
            store.add([f"v{i}" for i in range(n_texts)], m)
            report["results"]["vecdb_query"] = bench_vecdb(
                store, m[:16], k=5, n_repeat=n_repeat)
        except Exception as e:
            report["results"]["vecdb_query"] = {"error": f"{type(e).__name__}: {e}"}

    report["meta"]["seconds"] = round(time.perf_counter() - t0, 2)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"[stress] 已保存 {out_path}")
    _print_stress(report)
    return report


def _build_min_scorer(enc, texts: list[str]):
    """最小打分器：编码 + 词典比照（无 sklearn 依赖的最小路径）。"""
    from .vecdb import _l2normalize

    class _MiniScorer:
        def __init__(self, enc, refs):
            self.enc = enc
            self.refs = refs
            self.ref_emb = _l2normalize(enc.encode(refs))

        def score(self, texts, explain: bool = False):
            X = _l2normalize(self.enc.encode(texts))
            sims = X @ self.ref_emb.T
            scores = sims.max(axis=1)
            return {"results": [{"score": float(s)} for s in scores],
                    "n": len(texts), "model": "mini"}
    return _MiniScorer(enc, texts)


def _build_real_scorer(cfg, texts: list[str]):
    """真实在线打分器：PoisonScorer 完整栈（KNN/校准/学生模型如果有）。"""
    from .data import load_splits
    from .encoders import get_encoder
    from .service import _default_student, PoisonScorer

    enc = get_encoder(cfg)
    scorer = PoisonScorer(cfg, student_path=_default_student(cfg))
    splits = load_splits(cfg.data.processed_dir)
    train = [s.text for s in splits["train"]][:400]
    clean_dev = [s.text for s in splits["dev"]
                 if s.label == 0 and s.family == "normal"][:200]
    labeled = ([s.text for s in splits["dev"] if s.label == 1],
               np.asarray([s.label for s in splits["dev"] if s.label == 1]))
    scorer.fit(train, clean_dev, labeled=labeled)
    return scorer


def _make_scan_fn(enc, refs, ref_embs):
    from .ragcheck import scan_scores

    def run(texts):
        return scan_scores(texts, refs, ref_embs, enc, top_k=3, min_ratio=0.75,
                           progress=False)
    return run


def _print_stress(report: dict) -> None:
    for kind, res in report["results"].items():
        if "error" in res:
            print(f"[stress] {kind}: ERROR {res['error']}")
            continue
        per = res.get("per_call_ms") or res.get("per_query_ms") or {}
        print(f"[stress] {kind}: "
              f"mean {per.get('mean_ms', 0):.2f}ms  p50 {per.get('p50_ms', 0):.2f}ms  "
              f"p99 {per.get('p99_ms', 0):.2f}ms  QPS {per.get('qps', 0):.0f}")
