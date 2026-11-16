# -*- coding: utf-8 -*-
"""Gap 2 可解释性：explain.py 纯逻辑测试（不加载模型、不读写真实数据）。

用合成 guard 结果 dict 验证解释构建；用 tmp_path 里手工落盘的 jsonl
验证 read_alert/list_alerts 的读取与溯源（不依赖 ingest 业务）。
"""
from __future__ import annotations

import json

import pytest

from poisonguard.explain import (
    attach_explanation,
    build_explanation,
    explain_alert,
    list_alerts,
    read_alert,
    reason_from_result,
    signal_explain,
)


def _result(verdict: str = "block", signals=None, explain=None) -> dict:
    return {
        "verdict": verdict,
        "score": 0.9,
        "threshold": 0.3,
        "scene": "prompt",
        "signals": signals if signals is not None else [
            {"name": "rule_injection", "score": 1.0, "detail": "ignore"},
            {"name": "corpus_poison", "score": 0.8, "detail": "threshold=0.5"},
        ],
        "rules_hit": ["ignore"],
        "budget_ms": 0.5,
        "judge_called": False,
        "explain": explain if explain is not None else {
            "effective_watch": 0.3, "effective_block": 0.6,
            "block_headroom": 0.1, "threshold_source": "calibrated",
            "target_fpr": 0.01, "watch_mult": 1.0, "block_mult": 1.2,
            "views": {}, "sanitized": None,
        },
    }


# ----------------------------------------------------------------------
# 解释构建
# ----------------------------------------------------------------------
def test_signal_explain_names_known_source():
    s = signal_explain({"name": "corpus_poison", "score": 0.87, "detail": "threshold=0.5"})
    assert "语料投毒检测器" in s and "0.87" in s and "threshold=0.5" in s


def test_signal_explain_unknown_name_passthrough():
    s = signal_explain({"name": "weird_signal", "score": 0.2})
    assert "weird_signal" in s


def test_reason_from_result_joins_signals_and_threshold():
    r = _result()
    reason = reason_from_result(r)
    assert "显式注入指令词" in reason
    assert "语料投毒检测器" in reason
    assert "block 阈值 0.6" in reason


def test_reason_empty_signals_says_normal():
    r = _result(signals=[])
    assert "未命中任何信号" in reason_from_result(r)


def test_build_explanation_shape_and_verdict_cn():
    exp = build_explanation(_result())
    assert exp["verdict"] == "block"
    assert exp["verdict_cn"] == "拦截"
    assert isinstance(exp["signals"], list) and len(exp["signals"]) == 2
    assert exp["action"] and "拦截" in exp["action"]
    assert exp["threshold"]["watch"] == 0.3
    assert len(exp["trace"]) == 2


def test_attach_explanation_is_non_mutating():
    r = _result()
    out = attach_explanation(r)
    assert out is not r
    assert "explanation" in out
    assert "explanation" not in r


# ----------------------------------------------------------------------
# 告警读取（tmp_path 手工落盘）
# ----------------------------------------------------------------------
def _make_alert(out, flag: str, source: str, row: int, text: str,
                score: float) -> None:
    sub = {"BLOCK": "quarantine", "WATCH": "review", "PASS": "released"}[flag]
    d = out / sub
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "source": source, "row_id": row, "text": text,
        "text_sha": text.encode("utf-8").hex()[:16],
        "score": score, "flag": flag,
        "views": {"knn": 0.42, "student": 0.71},
    }
    with (d / "20261117.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_list_alerts_sorts_by_score_desc(tmp_path):
    _make_alert(tmp_path, "BLOCK", "a.txt", 0, "bad1", 0.9)
    _make_alert(tmp_path, "WATCH", "b.txt", 0, "skip1", 0.4)
    _make_alert(tmp_path, "BLOCK", "c.txt", 0, "med1", 0.7)
    recs = list_alerts(str(tmp_path), limit=10)
    assert len(recs) == 3
    scores = [float(r["score"]) for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_list_alerts_filters_flags(tmp_path):
    _make_alert(tmp_path, "BLOCK", "a.txt", 0, "bad1", 0.9)
    _make_alert(tmp_path, "PASS", "b.txt", 0, "fine1", 0.1)
    recs = list_alerts(str(tmp_path), flags=("BLOCK",), limit=10)
    assert len(recs) == 1 and recs[0]["flag"] == "BLOCK"


def test_read_alert_returns_explained_detail(tmp_path):
    _make_alert(tmp_path, "BLOCK", "a.txt", 3, "非常可疑的内容", 0.87)
    recs = list_alerts(str(tmp_path), limit=10)
    key = recs[0]["key"]
    a = read_alert(key, str(tmp_path))
    assert a is not None
    assert a["flag"] == "BLOCK"
    assert a["flag_cn"] == "拦截"
    assert a["row_id"] == 3
    assert "来源" in a["reason"] and "0.87" in a["reason"]
    assert a["views"]["knn"] == 0.42
    # key 由 source|row_id|text_sha 拼接
    assert key.startswith("a.txt|3|")


def test_read_alert_missing_key_returns_none(tmp_path):
    assert read_alert("no|such|key", str(tmp_path)) is None


def test_explain_alert_without_views_is_robust(tmp_path):
    rec = {"flag": "WATCH", "score": None, "source": "", "row_id": None,
           "text": "x", "views": None}
    e = explain_alert(rec)
    assert e["flag_cn"] == "观察"
    assert isinstance(e["views"], (dict, type(None)))
