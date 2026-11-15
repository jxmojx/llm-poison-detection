"""数字级事实校验（P0-2）的纯逻辑测试：不下载模型。

钉住的是三件事：数字怎么找、分数怎么翻、融合怎么配——这三处错了，
跑出来的 AUROC 就没有意义。
"""

from __future__ import annotations

import numpy as np

from poisonguard.factcheck import (_number_token_mask, evaluation_rows, fill_missing,
                                   fuse, has_number, number_spans, run_factcheck,
                                   suspiciousness)


# ---------------------------------------------------------------- 抽取
def test_number_spans_find_value_and_unit():
    text = "今年高考报名人数首次突破1300万人，同比增长30%"
    spans = number_spans(text)
    assert len(spans) == 2
    got = [text[a:b] for a, b in spans]
    assert got == ["1300万", "30%"], got


def test_number_spans_handles_decimal_and_year():
    text = "2024年GDP增长5.2%，总额约134.9万亿元"
    got = [text[a:b] for a, b in number_spans(text)]
    assert "2024年" in got and "5.2%" in got


def test_has_number_and_empty_text():
    assert has_number("增长30%") is True
    assert has_number("没有任何数字") is False
    assert has_number("") is False
    assert number_spans("") == []


def test_number_token_mask_picks_only_digit_tokens():
    toks = ["今年", "增长", "30", "%", "，", "达", "到", "1300", "万"]
    assert _number_token_mask(toks) == [2, 3, 7, 8]


# ---------------------------------------------------------------- 分数变换
def test_suspiciousness_flips_sign_and_keeps_nan():
    s = np.array([-1.0, -5.0, np.nan])
    out = suspiciousness(s)
    assert out[0] == 1.0 and out[1] == 5.0, "对数概率越负 → 可疑度越高"
    assert np.isnan(out[2])


def test_fill_missing_replaces_nan_only():
    out = fill_missing(np.array([1.0, np.nan, 3.0]), -2.0)
    assert out.tolist() == [1.0, -2.0, 3.0]


def test_fuse_is_zscore_weighted():
    base = np.array([0.0, 1.0, 2.0, 3.0])
    fact = np.array([3.0, 2.0, 1.0, 0.0])
    w0 = fuse(base, fact, w=0.0)
    w1 = fuse(base, fact, w=1.0)
    # w=0 时排序应与 base 一致（严格增）
    assert w0[0] < w0[1] < w0[2] < w0[3]
    # w=1 时排序应与 fact 一致（严格减）
    assert w1[0] > w1[1] > w1[2] > w1[3]


def test_fuse_ignores_scale_of_inputs():
    base = np.array([0.0, 1.0, 2.0])
    a = fuse(base, base * 1.0, w=0.5)
    b = fuse(base, base * 1000.0, w=0.5)
    assert np.allclose(a, b), "z-score 后应与原尺度无关"


# ---------------------------------------------------------------- 分家族
def test_evaluation_rows_coverage():
    scores = np.array([1.0, np.nan, 3.0, np.nan])
    labels = np.array([1, 1, 0, 0])
    fams = ["edit", "edit", "normal", "normal"]
    rows = evaluation_rows(scores, labels, fams)
    assert rows["edit"]["n"] == 2 and rows["edit"]["n_with_number"] == 1
    assert rows["edit"]["coverage"] == 0.5
    assert rows["normal"]["mean_susp"] == 3.0


def test_run_factcheck_without_model_is_importable():
    """确保模块可导入、函数签名稳定（真正跑模型由 CLI 端到端覆盖）。"""
    assert callable(run_factcheck)
