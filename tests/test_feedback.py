"""误报闭环（C4）的测试：统计口径与"只建议不生效"的边界。

这里最怕两类错：
1. 统计口径错（把小样本误报率当结论、把选择偏差当漏报率）→ 用合成数据钉死；
2. 影子阈值偷偷生效（改线上判定）→ 断言只读、只算、不写回。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from poisonguard.feedback import (apply_shadow, build_report, confirmed_positives,
                                  daily_stats, drift_alarm, export_corpus,
                                  false_positives, fp_rate, load_decisions,
                                  run_feedback, shadow_threshold, wilson_interval)


def _rec(score: float, flag: str, decision: str, sha: str, day: float = 0.0,
         text: str = "x") -> dict:
    return {"score": score, "flag": flag, "decision": decision, "text_sha": sha,
            "source": "/d/f.txt", "row_id": 0, "text": text,
            "model": "m", "threshold": 0.5, "decided_at": day or 1_700_000_000.0}


def _db(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "r.sqlite3"
    with sqlite3.connect(p) as c:
        c.executescript("""CREATE TABLE decisions(key TEXT PRIMARY KEY, source TEXT, row_id INTEGER,
            text_sha TEXT, score REAL, flag TEXT, decision TEXT, reason TEXT, note TEXT,
            reviewer TEXT, model TEXT, threshold REAL, decided_at REAL);""")
        for i, r in enumerate(rows):
            c.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (f"k{i}", r["source"], r["row_id"], r["text_sha"], r["score"],
                       r["flag"], r["decision"], "口语闲聊", "", "u", r["model"],
                       r["threshold"], r["decided_at"]))
    return p


# ---------------------------------------------------------------- 读取
def test_missing_db_gives_actionable_message(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="先跑"):
        load_decisions(tmp_path / "nope.sqlite3")


def test_load_decisions_returns_rows(tmp_path: Path):
    db = _db(tmp_path, [_rec(0.9, "BLOCK", "benign", "a")])
    assert len(load_decisions(db)) == 1


# ---------------------------------------------------------------- 语料
def test_false_positives_only_takes_benign_blocked_and_dedupes():
    rows = [_rec(0.9, "BLOCK", "benign", "a"), _rec(0.9, "BLOCK", "benign", "a"),
            _rec(0.8, "WATCH", "poison", "b"), _rec(0.1, "PASS", "benign", "c")]
    fps = false_positives(rows)
    assert len(fps) == 1 and fps[0]["text_sha"] == "a", "只取被拦下且判 benign 的，去重"
    assert false_positives(rows, min_score=0.95) == []


def test_confirmed_positives_only_poison():
    rows = [_rec(0.9, "BLOCK", "poison", "p1"), _rec(0.9, "BLOCK", "benign", "b1")]
    pos = confirmed_positives(rows)
    assert [r["text_sha"] for r in pos] == ["p1"]


def test_export_corpus_writes_labels(tmp_path: Path):
    p = export_corpus([_rec(0.9, "BLOCK", "benign", "a", text="难负样本")],
                      tmp_path / "hn.jsonl", kind="hardneg")
    row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == 0 and row["text"] == "难负样本"
    assert row["source"] == "/d/f.txt" and row["model"] == "m"

    p2 = export_corpus([_rec(0.9, "BLOCK", "poison", "b")],
                       tmp_path / "pos.jsonl", kind="positive")
    assert json.loads(p2.read_text(encoding="utf-8").splitlines()[0])["label"] == 1


# ---------------------------------------------------------------- 统计
def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0), "无样本时必须给全区间而不是 0"
    lo, hi = wilson_interval(3, 100)
    assert lo < 0.03 < hi and 0.0 <= lo <= hi <= 1.0
    lo2, hi2 = wilson_interval(30, 100)
    assert hi2 - lo2 > 0, "区间非退化"


def test_fp_rate_counts_only_blocked():
    rows = [_rec(0.9, "BLOCK", "benign", "a"), _rec(0.8, "WATCH", "poison", "b"),
            _rec(0.1, "PASS", "benign", "c"), _rec(0.7, "BLOCK", "", "d")]
    r = fp_rate(rows)
    assert r["n_blocked"] == 3 and r["n_fp"] == 1 and r["n_pending"] == 1
    assert abs(r["fp_rate"] - 1 / 3) < 1e-9


def test_small_sample_never_alarms():
    r = fp_rate([_rec(0.9, "BLOCK", "benign", f"s{i}") for i in range(5)])
    a = drift_alarm(r, 0.01, min_support=30)
    assert a["status"] == "insufficient" and a["alarm"] is False, "5 条样本不许下结论"


def test_alarm_needs_ci_lower_bound_above_target():
    # 40 条里 30 条误报：区间下界远高于 1% → 必须告警
    rows = [_rec(0.9, "BLOCK", "benign", f"b{i}") for i in range(30)] + \
           [_rec(0.9, "BLOCK", "poison", f"p{i}") for i in range(10)]
    r = fp_rate(rows)
    a = drift_alarm(r, 0.01, min_support=30)
    assert a["status"] == "drifting" and a["alarm"] is True


def test_point_estimate_above_target_but_ci_covers_it_is_only_watch():
    # 100 条里 2 条误报：点估计 2% > 1%，但区间覆盖 1% → 不告警
    rows = [_rec(0.9, "BLOCK", "benign", f"b{i}") for i in range(2)] + \
           [_rec(0.9, "BLOCK", "poison", f"p{i}") for i in range(98)]
    a = drift_alarm(fp_rate(rows), 0.01, min_support=30)
    assert a["status"] == "watch" and a["alarm"] is False


def test_daily_stats_groups_by_local_day():
    t0 = 1_700_000_000.0
    rows = [_rec(0.9, "BLOCK", "benign", "a", t0), _rec(0.9, "BLOCK", "poison", "b", t0),
            _rec(0.9, "BLOCK", "benign", "c", t0 + 86400)]
    days = daily_stats(rows)
    assert len(days) == 2 and days[0]["n_fp"] == 1 and days[0]["n_blocked"] == 2


# ---------------------------------------------------------------- 影子阈值
def test_shadow_threshold_raises_threshold_when_fp_are_high_scored():
    clean = np.linspace(0, 0.5, 200)              # 干净分数很小
    fp = np.array([0.80, 0.82, 0.85, 0.88])       # 误报分数很高
    s = shadow_threshold(clean, fp, 0.01, mode="pooled")
    assert s["new_threshold"] >= s["old_threshold"], "并进高分干净样本，阈值只能升不能降"
    assert s["fp_still_blocked"] < 4, "并进这批误报后，仍被拦下的应少于 4 条"
    assert s["fp_blocked_before"] == 4
    assert s["ref_used"] == "pooled" and s["n_fp"] == 4


def test_shadow_threshold_replace_mode_needs_support():
    clean = np.linspace(0, 0.5, 200)
    fp = np.array([0.8, 0.85])                    # 只有 2 条 → 退回 pooled
    s = shadow_threshold(clean, fp, 0.01, mode="replace", min_support=10)
    assert s["ref_used"] == "pooled", "样本不足时不许用 replace"
    assert shadow_threshold(clean, np.linspace(0.7, 0.9, 50), 0.01,
                            mode="replace", min_support=10)["ref_used"] == "fp_only"


def test_shadow_threshold_is_pure():
    clean = np.linspace(0, 0.5, 50)
    fp = np.array([0.9])
    before = clean.copy()
    shadow_threshold(clean, fp, 0.01)
    assert np.array_equal(clean, before), "重算阈值不能改调用方的数组"


def test_apply_shadow_counts_flag_changes_only():
    recs = [_rec(0.9, "BLOCK", "benign", "a"), _rec(0.4, "BLOCK", "poison", "b")]
    # 阈值 0.3 → 0.5：0.4 那条从拦截变放过，0.9 仍是拦截
    out = apply_shadow(recs, {"old_threshold": 0.3, "new_threshold": 0.5})
    assert out["n_flag_changes"] == 1, "只有 0.4 那条会翻转"
    assert "未改动线上" in out["note"]


def test_apply_shadow_uses_online_threshold_and_reports_recall_cost():
    """回放必须以线上阈值为基准，并且把"放过多少确认有毒"算出来。"""
    recs = [_rec(0.90, "BLOCK", "benign", "a"), _rec(0.85, "BLOCK", "poison", "b"),
            _rec(0.20, "BLOCK", "poison", "c")]
    out = apply_shadow(recs, {"online_threshold": 0.5, "old_threshold": 0.9,
                              "new_threshold": 0.95})
    assert out["n_flag_changes"] == 2, "0.90 和 0.85 都从拦截变放过"
    assert out["dropped_by_decision"] == {"benign": 1, "poison": 1}
    assert out["from_threshold"] == 0.5 and out["to_threshold"] == 0.95


def test_run_feedback_surfaces_threshold_gap(tmp_path: Path):
    """线上阈值 0.5、校准阈值 0.95 → 必须把口径差写进结果，否则人会以为阈值没差。"""
    rows = [_rec(0.90, "BLOCK", "benign", "b1"), _rec(0.90, "BLOCK", "poison", "p1")]
    db = _db(tmp_path, rows)
    res = run_feedback(db, clean_scores=np.linspace(0, 0.95, 500), target_fpr=0.01,
                       min_support=1, corpus_dir=None)
    sh = res["shadow"]
    assert sh["online_threshold"] == 0.5
    assert sh["new_threshold"] > 0.85, "99 分位应落在干净分布的高端"
    assert sh["threshold_gap"] > 0.3, "口径差必须被显式记录"
    assert sh["fp_blocked_online"] == 1 and sh["fp_still_blocked"] == 0
    assert res["replay"]["n_flag_changes"] >= 1


def test_shadow_requires_clean_set():
    with pytest.raises(ValueError, match="干净校准集为空"):
        shadow_threshold(np.array([]), np.array([0.9]), 0.01)
    with pytest.raises(ValueError, match="mode"):
        shadow_threshold(np.linspace(0, 1, 20), np.array([0.9]), 0.01, mode="nope")


# ---------------------------------------------------------------- 端到端
def test_run_feedback_end_to_end(tmp_path: Path):
    rows = [_rec(0.9, "BLOCK", "benign", f"b{i}") for i in range(12)] + \
           [_rec(0.9, "BLOCK", "poison", f"p{i}") for i in range(20)]
    db = _db(tmp_path, rows)
    res = run_feedback(db, out_path=tmp_path / "fb.json",
                       clean_scores=np.linspace(0, 0.5, 300),
                       target_fpr=0.01, min_support=10,
                       corpus_dir=tmp_path / "corpus",
                       report_path=tmp_path / "day.md")
    assert (tmp_path / "fb.json").exists()
    assert res["fp_rate"]["n_fp"] == 12 and res["fp_rate"]["n_blocked"] == 32
    assert res["shadow"]["new_threshold"] >= res["shadow"]["old_threshold"]
    assert (tmp_path / "corpus" / "hardneg_feedback.jsonl").exists()
    assert (tmp_path / "corpus" / "confirmed_poison.jsonl").exists()
    text = (tmp_path / "day.md").read_text(encoding="utf-8")
    assert "误报率" in text and "影子阈值" in text
    assert "拦截精度" in text and "漏报" in text, "报告必须写明口径与选择偏差"


def test_report_without_shadow_still_renders():
    res = {"db": "x", "target_fpr": 0.01, "n_decided": 2,
           "fp_rate": {"n_blocked": 1, "n_fp": 0, "fp_rate": 0.0, "ci95": [0.0, 0.9],
                       "n_poison": 1, "n_unsure": 0, "n_pending": 0},
           "alarm": {"status": "insufficient", "alarm": False, "message": "样本不足"},
           "daily": [], "corpora": {}, "shadow": None, "replay": None}
    text = build_report(res)
    assert "拦截精度" in text and "样本不足" in text
