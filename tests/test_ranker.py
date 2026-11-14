"""A5 学生排序器的纯逻辑测试（不下载模型、不碰数据）。

这些断言守的是**排序器的数学契约**：换特征/换监督源之后，变体本身的
语义不能漂移——否则"edit 排不上来"的结论就可能只是实现 bug。
"""

from __future__ import annotations

import numpy as np
import pytest

from poisonguard.ranker import (VARIANTS, StudentRanker, family_lift_at_rates,
                               topk_family_lift)


def _toy(n: int = 60, dim: int = 8, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, dim))
    y = (np.arange(n) % 2 == 0).astype(int)
    X[y == 1] += 2.5  # 可分离
    texts = [f"样本编号{i}：这是一条测试文本" for i in range(n)]
    return texts, X, y.astype(np.float64)


def test_all_variants_train_and_rank_monotonically():
    texts, X, y = _toy()
    for v in VARIANTS:
        st = StudentRanker(variant=v, C=1.0, seed=0).fit(texts, X, y)
        s = st.score(texts, X)
        assert len(s) == len(y)
        assert s.min() >= 0.0 and s.max() <= 1.0
        assert s[y == 1].mean() > s[y == 0].mean(), f"{v} 正类均分应更高"


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        StudentRanker(variant="nope")


def test_stack_is_mean_of_two_heads():
    texts, X, y = _toy()
    st = StudentRanker(variant="stack", C=1.0, seed=0).fit(texts, X, y)
    s = st.score(texts, X)
    a = st.clf.predict_proba(st._F("emb", texts, X))[:, 1]
    b = st._clf2.predict_proba(st._F("ngram", texts, X))[:, 1]
    assert np.allclose(s, (a + b) / 2.0)


def test_pca_variant_reduces_dimension():
    texts, X, y = _toy(n=80, dim=16)
    st = StudentRanker(variant="emb_pca", pca_dim=4, seed=0).fit(texts, X, y)
    F = st._build(texts, X, fit=False)
    assert F.shape == (80, 4)


def test_ngram_only_ignores_embedding_dimension():
    texts, X, y = _toy()
    st = StudentRanker(variant="ngram", seed=0).fit(texts, X, y)
    s1 = st.score(texts, X)
    s2 = st.score(texts, np.zeros_like(X))  # ngram 变体不看句向量
    assert np.allclose(s1, s2)


def test_topk_family_lift_math():
    labels = np.array([1, 1, 1, 1])
    fam = np.array(["edit", "edit", "template", "template"])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    r = topk_family_lift(labels, fam, scores, "edit", 2)
    assert r["n_family_total"] == 2
    assert r["n_family_topk"] == 2
    assert r["lift"] == pytest.approx(2.0)  # Top2 里 edit 占 2/2，总体占 2/4
    r2 = topk_family_lift(labels, fam, scores, "edit", 1)
    assert r2["n_family_topk"] == 1
    assert r2["lift"] == pytest.approx(2.0)


def test_family_lift_at_rates_keys_and_skips_clean():
    n = 1000
    rng = np.random.RandomState(1)
    labels = np.array([1] * 500 + [0] * 500)
    fam = np.array(["edit"] * 250 + ["template"] * 250 + ["normal"] * 500)
    scores = rng.rand(n)
    out = family_lift_at_rates(labels, fam, scores, rates=(0.05,))
    assert "top50" in out
    assert "edit" in out["top50"]
    assert "normal" not in out["top50"]  # 干净家族不参与富集统计
    assert set(out["top50"]["edit"]) >= {"n_top", "recall_in_topk", "lift"}


def test_save_load_roundtrip(tmp_path):
    texts, X, y = _toy()
    st = StudentRanker(variant="emb", seed=0).fit(texts, X, y)
    p = tmp_path / "r.pkl"
    st.save(str(p))
    back = StudentRanker.load(str(p))
    assert np.allclose(st.score(texts, X), back.score(texts, X))
