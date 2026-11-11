"""scan 批扫产品的回归测试：输入解析 / 判官域阈值加载 / 联合结论 / 报告渲染。

不下载模型、不跑 infer，只验证"产品逻辑对不对"（与 test_v3.py 同哲学）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poisonguard.config import load_config  # noqa: E402
from poisonguard.scan import (_joint_verdict, _load_judge_threshold,  # noqa: E402
                              _render_markdown, load_rows)


# ----------------------------------------------------------------------
def test_load_rows_txt_respects_rows_and_row_id():
    p = ROOT / "outputs" / "scan" / "_unit_tmp_rows.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("第一句\n\n第二句\n第三句\n", encoding="utf-8")
    try:
        rows = load_rows(p)
        # row_id = 源文件物理行号（0 基，含空行，空行占据行号）→ [0, 2, 3]
        assert [r["row_id"] for r in rows] == [0, 2, 3]
        assert [r["text"] for r in rows] == ["第一句", "第二句", "第三句"]
        assert load_rows(p, limit=2) == rows[:2]
    finally:
        p.unlink(missing_ok=True)


def test_load_rows_csv_or_jsonl():
    p = ROOT / "outputs" / "scan" / "_unit_tmp.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"text": "甲", "id": 1}, ensure_ascii=False) + "\n" +
        json.dumps({"sentence": "乙"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        rows = load_rows(p)   # .jsonl -> jsonl 分支（字典或句子字段都取）
        assert [r["text"] for r in rows] == ["甲", "乙"]
    finally:
        p.unlink(missing_ok=True)


# ----------------------------------------------------------------------
def test_load_judge_threshold_reads_domain_json(tmp_path):
    """判官阈值来自当前域 judge_{tag}.json 的 detectors 阈值，兜底 0.5。"""
    cfg = load_config()
    # 造一个假的 metrics_dir，写 judge_afqmc.json 结构
    cfg.metrics_dir = str(tmp_path)
    tag = Path(cfg.data.processed_dir).name
    assert tag.startswith("processed"), tag
    domain = tag[len("processed"):]
    (tmp_path / f"judge_{domain}.json").write_text(
        json.dumps({"detectors": {"qwen_logprob": {"threshold": 0.8845, "auroc": 0.78}}}),
        encoding="utf-8",
    )
    assert _load_judge_threshold(cfg) == pytest.approx(0.8845)

    # 空目录 -> 兜底 0.5
    cfg.metrics_dir = str(tmp_path / "empty")
    assert _load_judge_threshold(cfg) == 0.5


def test_joint_verdict_four_lanes():
    """联合结论：判官阈值 j_thr 独立于粗筛阈值 thr。"""
    # 双越界 -> 确认风险；判官命中->捕获；粗筛报警判官未确认->待人工；都过->低风险
    assert _joint_verdict(0.99, 0.90, 0.5, 0.88) == "确认风险"
    assert _joint_verdict(0.40, 0.91, 0.5, 0.88) == "判官捕获（粗筛放过）"
    assert _joint_verdict(0.99, 0.50, 0.5, 0.88) == "待人工（粗筛报警，判官未确认）"
    assert "低风险" in _joint_verdict(0.40, 0.60, 0.5, 0.88)


def test_render_markdown_uses_domain_judge_threshold():
    """报告头插值判官域阈值，且联合结论按它算。"""
    import numpy as np

    summary = {
        "input": "demo.txt", "domain": "processed_afqmc", "has_student": True,
        "threshold": 0.5, "target_fpr": 0.01, "n_rows": 2,
        "mean_len_input": 20.0, "mean_len_dev": 19.0, "len_drift": 1.0,
        "ms_per_item": 5.0, "judge_reviewed": 2, "judge_threshold": 0.8845,
        "tiers": {"BLOCK": 1, "WATCH": 0, "PASS": 1},
    }
    rows = [{"row_id": 1}, {"row_id": 2}]
    texts = ["句一", "句二"]
    s = np.array([0.99, 0.40])
    flags = ["BLOCK", "PASS"]
    views = [{"student": 0.99, "char_ngram": 0.1, "fpr_est": 0.9}] * 2
    judge_rows = [
        {"row_id": 1, "score": 0.99, "judge": 0.93, "text": "句一"},
        {"row_id": 2, "score": 0.40, "judge": 0.91, "text": "句二"},
    ]
    md = _render_markdown(summary, rows, texts, s, flags, views, [0, 1], judge_rows)
    assert "判官阈值 0.8845" in md
    assert "确认风险" in md and "判官捕获（粗筛放过）" in md
