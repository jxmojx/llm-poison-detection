"""向量库集成（P1）纯逻辑测试：随机向量走暴力和 sklearn 两个后端，零模型。"""

from __future__ import annotations

import numpy as np
import pytest

from poisonguard.vecdb import VecStore, _l2normalize, bench_query


def _rng(seed=7):
    return np.random.default_rng(seed)


def _random_set(n=50, dim=8, seed=7):
    rng = _rng(seed)
    ids = [f"id{i}" for i in range(n)]
    m = rng.standard_normal((n, dim))
    return ids, m


def test_create_and_len():
    st = VecStore.create(dim=8)
    assert len(st) == 0
    ids, m = _random_set()
    st.add(ids, m)
    assert len(st) == 50


def test_add_normalizes():
    st = VecStore.create()
    ids, m = _random_set()
    st.add(ids, m)
    norms = np.linalg.norm(st.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_add_dim_mismatch_raises():
    st = VecStore.create(dim=8)
    with pytest.raises(ValueError):
        st.add(["a"], np.zeros((1, 5)))


def test_add_ids_len_mismatch_raises():
    st = VecStore.create(dim=8)
    with pytest.raises(ValueError):
        st.add(["a", "b"], np.zeros((1, 8)))


def test_query_returns_topk_sorted():
    ids, m = _random_set()
    st = VecStore.create(dim=m.shape[1])
    st.add(ids, m)
    res = st.query(m[0], k=5)
    assert len(res) == 5
    # 查询自身应排第一（cosine≈1）
    assert res[0]["id"] == ids[0]
    assert abs(res[0]["score"] - 1.0) < 1e-6
    # 分数降序
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_query_k_greater_than_n():
    ids, m = _random_set(n=3, dim=4)
    st = VecStore.create(dim=4)
    st.add(ids, m)
    res = st.query(m[0], k=100)
    assert len(res) == 3


def test_add_upsert_replaces_id():
    ids, m = _random_set(n=2, dim=4)
    st = VecStore.create(dim=4)
    st.add(ids, m)
    st.add([ids[0]], np.array([[9.0, 0, 0, 0]]))
    assert len(st) == 2  # 不重复
    res = st.query(np.array([[9.0, 0, 0, 0]]), k=1)
    assert res[0]["id"] == ids[0]
    assert res[0]["score"] > 0.999


def test_delete():
    ids, m = _random_set(n=10, dim=4)
    st = VecStore.create(dim=4)
    st.add(ids, m)
    assert st.delete([ids[0], ids[1]]) == 2
    assert len(st) == 8
    assert ids[0] not in st.ids and ids[1] not in st.ids


def test_persist_load_roundtrip(tmp_path):
    ids, m = _random_set(n=20, dim=6)
    st = VecStore.create(dim=6, meta={"encoder": "crom"})
    st.add(ids, m)
    p = tmp_path / "store.npz"
    st.persist(str(p))
    st2 = VecStore.load(str(p))
    assert len(st2) == 20
    assert st2.meta["encoder"] == "crom"
    assert st2.dim == 6
    # 查询等价
    r1 = st.query(m[0], k=3)
    r2 = st2.query(m[0], k=3)
    assert [x["id"] for x in r1] == [x["id"] for x in r2]


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VecStore.load(str(tmp_path / "nope.npz"))


def test_sklearn_backend_matches_brute():
    ids, m = _random_set(n=30, dim=6, seed=11)
    stb = VecStore.create(dim=6, backend="brute")
    stb.add(ids, m)
    sts = VecStore.create(dim=6, backend="sklearn")
    sts.add(ids, m)
    for qi in range(5):
        rb = stb.query(m[qi], k=4)
        rs = sts.query(m[qi], k=4)
        assert [x["id"] for x in rb] == [x["id"] for x in rs]
        # 分数近似（两种后端 float 误差）
        assert max(abs(a["score"] - b["score"]) for a, b in zip(rb, rs)) < 1e-9


def test_bench_query_runs():
    ids, m = _random_set(n=100, dim=16)
    st = VecStore.create(dim=16)
    st.add(ids, m)
    b = bench_query(st, m[:5], k=3, n_repeat=2)
    assert b["n_queries"] == 5 and b["qps"] > 0
    assert b["p50_ms"] >= 0


def test_l2normalize_zero_rows_safe():
    m = np.zeros((3, 4))
    out = _l2normalize(m)
    # 零向量没有方向：应保持零行（而非 NaN/Inf），且不崩溃
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)
