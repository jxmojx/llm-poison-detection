"""吞吐压测（P1）纯逻辑测试：TinyEncoder + 手搓最小打分器，不下载模型。"""

from __future__ import annotations

import numpy as np

from poisonguard.stress import (_pct, _summarize, _make_tiny_texts, bench_encoder,
                                bench_ragcheck_scan, bench_scorer, bench_vecdb)
from poisonguard.encoders import TinyEncoder


def test_make_tiny_texts_shapes():
    t = _make_tiny_texts(50)
    assert len(t) == 50
    assert all(isinstance(s, str) and len(s) >= 10 for s in t)


def test_summarize_basic():
    s = _summarize([1.0, 2.0, 3.0])
    assert s["mean_ms"] == 2.0
    assert s["p50_ms"] == 2.0
    assert s["max_ms"] == 3.0
    assert s["qps"] > 0


def test_pct():
    assert _pct([1, 2, 3, 4], 0.5) == pytest_approx(2.5)
    assert _pct([], 0.5) != _pct([], 0.5)


def pytest_approx(x):
    return x


def test_bench_encoder_tiny():
    enc = TinyEncoder()
    texts = _make_tiny_texts(32)
    b = bench_encoder(enc, texts, batch_size=32, n_repeat=2, use_cache=False)
    assert b["kind"] == "encoder"
    assert b["n_texts"] == 32
    assert b["per_call_ms"]["qps"] > 0


def test_bench_scorer_mini():
    from poisonguard.stress import _build_min_scorer

    enc = TinyEncoder()
    texts = _make_tiny_texts(16)
    scorer = _build_min_scorer(enc, texts)
    b = bench_scorer(scorer, texts, batch_size=16, n_repeat=2)
    assert b["kind"] == "scorer"
    assert b["per_call_ms"]["qps"] > 0


def test_bench_ragcheck_scan_closure():
    from poisonguard.stress import _make_scan_fn

    enc = TinyEncoder()
    texts = _make_tiny_texts(10)
    refs = texts[:5]
    ref_embs = enc.encode(refs)
    fn = _make_scan_fn(enc, refs, ref_embs)
    b = bench_ragcheck_scan(fn, texts[:4], refs, n_repeat=1)
    assert b["kind"] == "ragcheck_scan"
    assert b["n_texts"] == 4


def test_bench_vecdb_query():
    from poisonguard.vecdb import VecStore

    rng = np.random.default_rng(3)
    m = rng.standard_normal((40, 8))
    st = VecStore.create(dim=8)
    st.add([f"v{i}" for i in range(40)], m)
    b = bench_vecdb(st, m[:5], k=5, n_repeat=2)
    assert b["kind"] == "vecdb_query"
    assert b["per_query_ms"]["qps"] > 0
