"""阈值口径（P0-1）的测试。

这组测试的意义：阈值口径不一致不会让任何单测变红，却会让线上误报率从 1% 变成 60%。
所以把「分位怎么取、来源怎么标、空集怎么拒」都钉死。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from poisonguard.threshold import (CLEAN_FAMILIES, calibrate_threshold, clean_dev_texts,
                                   describe, resolve_threshold, scorer_scores)


class StubScorer:
    """按文本长度给分：足够确定，又能造出想要的分布。"""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(self, texts: list[str]) -> dict:
        return {"results": [{"score": (len(t) % 100) / 100.0} for t in texts]}


def _clean(n: int = 500, hi: float = 0.9) -> list[str]:
    """造 n 条分数落在 [0, hi] 的"干净文本"（长度 mod 100）。"""
    return ["x" * (1 + i % int(hi * 100)) for i in range(n)]


# ---------------------------------------------------------------- 标定
def test_calibrate_returns_high_quantile_not_mean():
    sc = StubScorer()
    res = calibrate_threshold(sc, _clean(1000), target_fpr=0.01)
    assert res["method"] == "empirical_quantile"
    assert res["n_clean"] == 1000
    assert 0.0 <= res["achieved_fpr"] <= 0.02, "标定后实测 FPR 必须贴着目标"
    assert res["threshold"] > 0.5, "1% 分位应落在干净分布的尾部，而不是中位数"


def test_target_fpr_monotonic():
    sc = StubScorer()
    lo = calibrate_threshold(sc, _clean(), target_fpr=0.001)["threshold"]
    mid = calibrate_threshold(sc, _clean(), target_fpr=0.01)["threshold"]
    hi = calibrate_threshold(sc, _clean(), target_fpr=0.10)["threshold"]
    assert lo >= mid >= hi, "目标 FPR 越大，阈值应越低（更宽松）"


def test_empty_clean_rejected():
    with pytest.raises(ValueError, match="非空"):
        calibrate_threshold(StubScorer(), [], target_fpr=0.01)
    with pytest.raises(ValueError):
        calibrate_threshold(StubScorer(), ["", "   "], target_fpr=0.01)


def test_scorer_scores_alignment():
    s = scorer_scores(StubScorer(), ["a", "bb", "ccc"])
    assert s.tolist() == [0.01, 0.02, 0.03]


# ---------------------------------------------------------------- 来源
def test_override_wins_over_calibration():
    res = resolve_threshold(StubScorer(0.5), None, override=0.7, calibrate=False)
    assert res["source"] == "cli_override" and res["threshold"] == 0.7
    assert "覆盖" in res["detail"]


def test_no_calibration_falls_back_to_scorer_default():
    res = resolve_threshold(StubScorer(0.37), None, calibrate=True)
    assert res["source"] == "scorer_default"
    assert res["threshold"] == pytest.approx(0.37)


def test_describe_mentions_source_and_threshold():
    txt = describe({"threshold": 0.9981, "source": "calibrated", "detail": "在 2969 条…"})
    assert "0.9981" in txt and "calibrated" in txt


# ---------------------------------------------------------------- 干净集
def test_clean_dev_texts_filters_label_and_family(tmp_path: Path):
    rows = [
        {"text": "正常新闻一", "label": 0, "family": "normal", "split": "dev"},
        {"text": "硬负样一", "label": 0, "family": "hardneg", "split": "dev"},
        {"text": "模板攻击", "label": 1, "family": "template", "split": "dev"},
        {"text": "OOD 文本", "label": 0, "family": "ood", "split": "dev"},
    ]
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8", newline="\n")

    class Cfg:
        class data:
            processed_dir = str(tmp_path)

    got = clean_dev_texts(Cfg())
    assert got == ["正常新闻一", "硬负样一"], "只取 label=0 且在分布内(normal/hardneg)的样本"
    assert set(CLEAN_FAMILIES) == {"normal", "hardneg"}


def test_calibration_on_noisy_clean_is_conservative():
    """干净集里有少量脏样本时，1% 分位不应被单个极端值带飞。"""
    sc = StubScorer()
    texts = _clean(999) + ["y" * 99]  # 一条 0.99 的极端值
    res = calibrate_threshold(sc, texts, target_fpr=0.01)
    assert res["threshold"] < 1.0
    assert res["achieved_fpr"] <= 0.02
