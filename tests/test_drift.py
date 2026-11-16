"""域漂移体检（P1 换域）的纯逻辑测试：不下载任何模型。

测试只走字符统计 + 数值轴路径；向量轴用随机矩阵注入（不跑编码器）。
"""

from __future__ import annotations

import numpy as np

from poisonguard.drift import (char_profile, char_unigram, drift_report,
                               format_report, kl_divergence, length_stats,
                               run_drift_check, vocab_coverage)


def test_char_unigram_counts():
    c = char_unigram("abcabc中")
    assert c["a"] == 2 and c["b"] == 2 and c["中"] == 1


def test_kl_divergence_same_distribution_zero():
    p = {"a": 0.5, "b": 0.5}
    assert kl_divergence(p, dict(p)) == 0.0


def test_kl_divergence_disjoint_positive():
    p = {"a": 1.0}
    q = {"中": 1.0}
    assert kl_divergence(p, q) > 1.0


def test_vocab_coverage():
    p = {"a": 0.5, "中": 0.5}
    assert vocab_coverage(p, {"a", "b"}) == 0.5
    assert vocab_coverage(p, {"中", "a"}) == 1.0


def test_length_stats():
    s = length_stats(["abc", "abcd", "abcde"])
    assert s["n"] == 3 and s["mean_len"] == 4.0
    assert s["std_len"] > 0


def test_char_profile_vocab():
    c = char_profile(["abcab", "中中"])
    assert "b" in c["vocab"] and "中" in c["vocab"]
    assert abs(1.0 - sum(c["freq"].values())) < 1e-6


def test_drift_report_green_same_domain():
    ref = ["今天天气不错适合出门散步", "市场整体上涨带动成交量放大"] * 5
    inp = ["今天天气不错适合出门散步", "市场整体上涨带动成交量放大"] * 5
    r = drift_report(inp, ref)
    assert r["summary"]["status"] == "Green"
    assert r["summary"]["len_drift"] < 5


def test_drift_report_red_far_domain():
    ref = ["今天天气不错适合出门散步"] * 5
    inp = ["这是一段与训练领域毫无关系的极长内容" * 20] * 3
    r = drift_report(inp, ref)
    # 长文本理论差距应触发
    assert r["summary"]["worst_norm"] > 0.5


def test_drift_report_vector_axis_detects_gap():
    ref = ["今天天气不错适合出门散步", "市场整体上涨带动成交量放大"] * 3
    inp = ["今天天气不错适合出门散步"]
    ref_embs = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1))
    bad_embs = np.array([[0.0, 1.0]], dtype=np.float32)
    r = drift_report(inp, ref, embs_in=bad_embs, embs_ref=ref_embs,
                     vec_thr=0.1)
    assert r["axes"]["vector"]["ok"] is False
    assert "vector" in r["summary"]["dead_axes"]


def test_drift_report_empty_input_raises():
    import pytest

    with pytest.raises(ValueError):
        drift_report([], ["参考"])


def test_format_report_runs():
    ref = ["今天天气不错适合出门散步"] * 3
    inp = ["今天天气不错适合出门散步"]
    s = format_report(drift_report(inp, ref))
    assert "域漂移体检" in s


def test_drift_vocab_axis_catches_oov():
    ref = ["中文领域样本句子有词汇偏好"]
    inp = ["AcmeX9Ry23 SKU 404 not_found 100% pure ascii tokens"]
    r = drift_report(inp, ref, len_thr=1000.0)  # 拉高长度轴，专测词表轴
    assert r["axes"]["vocab"]["ok"] is False
    assert r["axes"]["vocab"]["coverage"] < 0.5

def test_run_drift_check_importable():
    assert callable(run_drift_check)
