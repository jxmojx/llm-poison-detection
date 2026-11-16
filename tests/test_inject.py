# -*- coding: utf-8 -*-
"""P3 注入头（inject.py）的回归测试。

共同点：**不下载模型、不跑 Qwen 编码**。用随机合成特征验证“逻辑对不对”，
真正的 0.5B 编码实验属于 `inject-train` 子命令的跑实验流程。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard.inject import (  # noqa: E402
    build_dataset,
    report_metrics,
    score_injection,
    train_injection_head,
)


# ----------------------------------------------------------------------
def test_build_dataset_splits_unseen_families():
    """unseen 家族必须从训练模板里剔除，只出现在 eval 侧；8 类全覆盖。"""
    from poisonguard.redteam import INJECT_PAYLOADS

    all_fams = {nm for nm, _ in INJECT_PAYLOADS}
    d = build_dataset(train_per_family=20, n_eval=30, seed=0)
    assert set(d["seen_families"]) | set(d["unseen_families"]) == all_fams
    # 两列互斥
    assert not (set(d["seen_families"]) & set(d["unseen_families"]))
    # 训练/评测正负均衡
    ys = [y for _, y in d["train"]]
    assert sum(ys) == len(ys) - sum(ys)
    assert len(d["train"]) > 0


def test_build_dataset_can_watch_all_families():
    """unseen_families=() 时所有家族都进 seen（训练可见）。"""
    d = build_dataset(train_per_family=10, n_eval=10, seed=2,
                      unseen_families=())
    from poisonguard.redteam import INJECT_PAYLOADS

    assert set(d["seen_families"]) == {nm for nm, _ in INJECT_PAYLOADS}
    assert d["unseen_families"] == []
    assert all(y == 1 for _, y, f in d["seen_eval"] if f != "normal")


def test_build_dataset_eval_carries_family_labels():
    d = build_dataset(train_per_family=10, n_eval=20, seed=1)
    seen_fams = {f for _, _, f in d["seen_eval"]} - {"normal"}
    assert seen_fams == set(d["seen_families"])
    unseen_fams = {f for _, _, f in d["unseen_eval"]} - {"normal"}
    assert unseen_fams == set(d["unseen_families"])


# ----------------------------------------------------------------------
def test_train_head_separable_clusters():
    """给定可分离的冻结特征，LR 注入头能学会二分类。"""
    rng = np.random.RandomState(0)
    Xpos = rng.randn(60, 16) + 2.0
    Xneg = rng.randn(60, 16) - 2.0
    W, b = train_injection_head(_cfg(), np.vstack([Xpos, Xneg]),
                                np.array([1] * 60 + [0] * 60), C=1.0, seed=0)
    assert W.shape == (16,)
    s = score_injection(W, b, np.vstack([Xpos, Xneg]))
    pred = (s > 0.5).astype(int)
    assert pred[:60].mean() > 0.95
    assert pred[60:].mean() < 0.05


def test_train_head_requires_both_classes():
    with pytest.raises(ValueError, match="两类标签"):
        train_injection_head(_cfg(), np.zeros((10, 8)), np.zeros(10, dtype=int))


def test_train_head_mismatch_length():
    with pytest.raises(ValueError, match="长度不一致"):
        train_injection_head(_cfg(), np.zeros((10, 8)), np.zeros(9, dtype=int))


def test_score_injection_is_probability():
    """分数必须是严格开区间的概率（小权重下不饱和）。"""
    rng = np.random.RandomState(1)
    W = rng.randn(896) * 0.01        # 小权重避免 sigmoid 饱和到端点
    s = score_injection(W, 0.0, rng.randn(5, 896) * 0.1)
    assert s.min() > 0.0 and s.max() < 1.0
    # 投影越大 → P(注入) 越大（sigmoid 单调）
    X = np.array([[1.0, 0.0], [2.0, 0.0]])
    W2 = np.array([1.0, 0.0])
    s2 = score_injection(W2, 0.0, X)
    assert s2[0] < s2[1]


# ----------------------------------------------------------------------
def _cfg() -> object:
    from types import SimpleNamespace

    return SimpleNamespace(metrics_dir="outputs/metrics")


def test_report_metrics_seen_and_unseen_columns():
    """两列口径（seen/unseen）必须是同一套函数 + 两套阈值（devcal/fixed）。"""
    rng = np.random.RandomState(2)
    y = np.array([1] * 20 + [0] * 20)
    s_seen = np.r_[rng.rand(20) * 0.1 + 0.9, rng.rand(20) * 0.1]
    s_unseen = np.r_[rng.rand(20) * 0.1 + 0.85, rng.rand(20) * 0.1]
    cal_y = np.zeros(40, dtype=int)
    cal_s = rng.rand(40) * 0.2

    seen = report_metrics("seen", y, s_seen, cal_y, cal_s)
    unseen = report_metrics("unseen", y, s_unseen, cal_y, cal_s)

    assert set(seen) == set(unseen)
    for k in ("auroc", "auprc", "tpr_at_fpr1_devcal",
              "precision_fixed", "recall_fixed", "f1_fixed",
              "precision_devcal", "recall_devcal", "f1_devcal",
              "threshold_fixed", "threshold_devcal"):
        assert k in seen
    assert seen["auroc"] > 0.9 and unseen["auroc"] > 0.9
    assert seen["recall_fixed"] >= 1.0 - 1e-9
    assert 0 < seen["threshold_devcal"] < 1


def test_report_metrics_threshold_respects_cal():
    """校准阈值应使干净侧误报接近 1%（上限）。"""
    rng = np.random.RandomState(3)
    y = np.array([1] * 10 + [0] * 400)
    s = np.r_[rng.rand(10) * 0.1 + 0.9, rng.rand(400)]
    cal_y = np.zeros(400, dtype=int)
    cal_s = rng.rand(400)
    m = report_metrics("x", y, s, cal_y, cal_s)
    thr = m["threshold_devcal"]
    fp = float((cal_s > thr).mean())
    assert fp <= 0.02
