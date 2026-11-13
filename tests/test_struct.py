"""结构隔离判官（A1）的回归测试：二层头训练 / 打分单调 / 校准边界。

共同点：**不下载模型、不跑 Qwen 编码**。用随机合成特征验证“逻辑对不对”，
真正的 0.5B 编码实验属于 `struct-train` 子命令的跑实验流程。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard.detectors import Calibration  # noqa: E402
from poisonguard.struct import (  # noqa: E402
    StructuredJudge,
    _logit_scores,
    calibrate,
    suffix_of,
    train_head,
)


# ----------------------------------------------------------------------
def _cfg(processed_dir: str = "data/processed_real_v3"):
    from types import SimpleNamespace

    return SimpleNamespace(
        data=SimpleNamespace(processed_dir=processed_dir, seed=42),
        detector=SimpleNamespace(target_fpr=0.01),
    )


# ----------------------------------------------------------------------
def test_train_head_learns_separable_clusters():
    """结构头的核心不变量：给定可分离的冻结特征，LR 头能学会二分类。"""
    rng = np.random.RandomState(0)
    Xpos = rng.randn(60, 16) + 2.0
    Xneg = rng.randn(60, 16) - 2.0
    X = np.vstack([Xpos, Xneg])
    y = np.array([1] * 60 + [0] * 60)
    W, b = train_head(_cfg(), {"dev": X}, y, C=1.0, seed=7)
    assert W.shape == (16,)
    s = _logit_scores(W, b, X)
    pred = (s > 0.5).astype(int)
    assert pred[y == 1].mean() > 0.95
    assert pred[y == 0].mean() < 0.05
    # 分数必须是概率（严格在开区间）
    assert s.min() > 0.0 and s.max() < 1.0


def test_train_head_requires_both_classes():
    X = np.zeros((10, 8))
    with pytest.raises(ValueError, match="正负两类标签"):
        train_head(_cfg(), {"dev": X}, np.zeros(10, dtype=int))


def test_train_head_requires_matching_length():
    X = np.zeros((10, 8))
    with pytest.raises(ValueError, match="长度不一致"):
        train_head(_cfg(), {"dev": X}, np.zeros(9, dtype=int))


# ----------------------------------------------------------------------
def test_logit_scores_matches_structured_judge_scores():
    rng = np.random.RandomState(1)
    W = rng.randn(896)
    b = 0.3
    X = rng.randn(5, 896)
    j = StructuredJudge(W, b=b)
    np.testing.assert_allclose(_logit_scores(W, b, X), j._scores(X), rtol=1e-9)


def test_structured_judge_supports_two_output_head():
    """兼容双输出头（[hidden,2]）：P(污染)=softmax 第二类的 logit 差值。"""
    rng = np.random.RandomState(2)
    X = rng.randn(4, 896)
    w = rng.randn(896)              # 两列共享同一权重 → 差值为 0 → sigmoid(0)=0.5
    W2 = np.column_stack([w, w])
    j = StructuredJudge(W2, b=0.0)
    s = j._scores(X)
    np.testing.assert_allclose(s, np.full(4, 0.5), atol=1e-9)


def test_structured_judge_score_monotonic_in_feature_projection():
    """分数在固定投影方向上单调：投影越大 → P(污染) 越大（sigmoid 单调性）。"""
    W = np.array([1.0, -1.0, 0.5])
    j = StructuredJudge(W, b=0.0)
    X = np.array([
        [0.0, 0.0, 0.0],   # z=0
        [1.0, 0.0, 0.0],   # z=1
        [2.0, 0.0, 0.0],   # z=2
    ])
    s = j._scores(X)
    assert s[0] < s[1] < s[2]


def test_structured_judge_predict_uses_threshold():
    j = StructuredJudge(np.array([1.0]), b=0.0, threshold=0.5)
    s = np.array([0.3, 0.6, 0.5])
    np.testing.assert_array_equal(j.predict(s), np.array([0, 1, 0]))  # 严格 >


# ----------------------------------------------------------------------
def test_calibrate_sets_threshold_from_clean_scores():
    rng = np.random.RandomState(3)
    clean = rng.rand(2000)          # 全为负样本侧的分数
    cal = calibrate(_cfg(), {"dev": np.zeros((2000, 8))}, clean, target_fpr=0.01)
    assert isinstance(cal, Calibration)
    assert cal.raw_threshold > 0.0
    assert abs(cal.achieved_fpr - 0.01) < 0.02
    # 95 分位以上的样本应被阈值为阳
    hi = cal.predict(clean)
    assert hi.mean() <= 0.02


def test_suffix_of_derives_same_as_redteam():
    assert suffix_of(_cfg("data/processed_real_v3")) == "_real_v3"
    assert suffix_of(_cfg("data/processed_afqmc")) == "_afqmc"
    assert suffix_of(_cfg("data/processed")) == ""
