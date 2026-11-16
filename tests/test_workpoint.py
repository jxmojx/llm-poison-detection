"""环节 30-5 客户工作点调优测试：target_fpr 参数化 + 报告产物。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poisonguard.threshold import calibrate_threshold
from poisonguard.workpoint import (
    DEFAULT_FPRS, _estimate_recall, sweep_workpoints, workpoint_report,
)


class _FakeScorer:
    """假打分器：分数 = 0.1 + 0.8 * (len/500)，落在 [0.1, 0.9] 且随长度单调。

    len 1..500 映射到 0.1..0.9：500 条长度各不相同的样本会把分数铺满区间，
    标定的阈值/实测 FPR 有区分度（不会挤在端点）。
    """

    threshold = 0.5

    def score(self, texts):
        scores = [min(0.1 + 0.8 * (len(t) / 500.0), 0.9) for t in texts]
        return {"results": [{"score": s} for s in scores], "n": len(scores)}


def test_calibrate_threshold_monotonic():
    """target_fpr 越小阈值越高；实测 FPR ≈ target。"""
    clean = ["短" + "好" * i for i in range(1, 401)]  # len 1..400 → 分数铺满 [0.1,0.74]
    sc = _FakeScorer()
    lo = calibrate_threshold(sc, clean, target_fpr=0.001)
    mid = calibrate_threshold(sc, clean, target_fpr=0.01)
    hi = calibrate_threshold(sc, clean, target_fpr=0.05)
    assert lo["threshold"] >= mid["threshold"] >= hi["threshold"]
    assert lo["achieved_fpr"] <= 0.005
    assert 0.008 <= mid["achieved_fpr"] <= 0.02
    assert 0.03 <= hi["achieved_fpr"] <= 0.08


def test_sweep_workpoints_shape_and_order():
    clean = ["好" * (12 + i % 5) for i in range(400)]
    pts = sweep_workpoints(_FakeScorer(), clean,
                           fprs=[0.05, 0.01, 0.001])
    assert [p["target_fpr"] for p in pts] == [0.001, 0.01, 0.05]
    for p in pts:
        assert p["threshold"] >= 0 and p["n_clean"] == 400


def test_workpoint_report_artifacts(tmp_path: Path):
    """报告产出 md+csv+json；毒样本估计召回随阈值单调不减（越严召回越高）。"""
    clean = ["干净" + "好" * (12 + i % 4) for i in range(300)]
    poison = ["投毒" + "恶" * (30 + i % 3) for i in range(100)]
    out = workpoint_report(
        _FakeScorer(), clean_texts=clean, poison_texts=poison,
        fprs=[0.001, 0.01, 0.05], out_dir=tmp_path / "rep",
        label="客户A", cfg=None)
    assert out["label"] == "客户A"
    assert len(out["points"]) == 3
    # 报告文件都存在
    for k in ("report_json", "report_md", "report_csv"):
        assert Path(out[k]).exists(), k
    rj = json.loads(open(out["report_json"], encoding="utf-8").read())
    assert rj["n_clean"] == 300 and rj["n_poison"] == 100
    md = open(out["report_md"], encoding="utf-8").read()
    assert "客户工作点报告" in md
    assert rj["points"][0]["target_fpr"] == 0.001
    assert "<<" not in md  # 无残留模板占位


def test_estimate_recall_direction():
    """更严格的阈值（更高）在毒样本上的召回应该 >= 宽松阈值。"""
    sc = _FakeScorer()
    poison = ["恶" * 40 for _ in range(50)]
    r_strict = _estimate_recall(sc, poison, 0.95)
    r_loose = _estimate_recall(sc, poison, 0.3)
    assert r_strict <= r_loose + 1e-9


def test_workpoint_no_poison_gives_none_recall(tmp_path: Path):
    """不给毒样本：recall 列为 None，报告仍可生成。"""
    clean = ["净" * 15 for _ in range(100)]
    out = workpoint_report(
        _FakeScorer(), clean_texts=clean, poison_texts=None,
        out_dir=tmp_path / "rep2", label="nopoison")
    assert all(p.get("est_recall_on_poison") is None for p in out["points"])
    md = open(out["report_md"], encoding="utf-8").read()
    assert "--" in md


def test_workpoint_default_fprs():
    clean = ["好" * 20 for _ in range(120)]
    pts = sweep_workpoints(_FakeScorer(), clean)
    assert [p["target_fpr"] for p in pts] == DEFAULT_FPRS
