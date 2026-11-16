"""环节 30-3 合规审计导出测试：台账 → JSON/MD/CSV 报告。"""

from __future__ import annotations

import json

from poisonguard.compliance import export_compliance, _aggregate
from poisonguard.ingest import IngestLedger, file_sha256

FLAGS = ("PASS", "WATCH", "BLOCK")


def _seed_ledger(tmp_path, db="ledger.sqlite3"):
    """造一个台账：2 个文件、若干行级记录（含 BLOCK/WATCH/PASS）。"""
    led = IngestLedger(tmp_path / db)
    # 文件1：3 行
    led.begin("shaAAAA", str(tmp_path / "a.csv"), model="stu-v1",
              pkg_sha="pkg1", threshold=0.5)
    led.add_records([
        ("shaAAAA", 0, "t0", 0.92, "BLOCK"),
        ("shaAAAA", 1, "t1", 0.47, "WATCH"),
        ("shaAAAA", 2, "t2", 0.10, "PASS"),
    ], decided_at=100.0)
    led.finish("shaAAAA", 3, "ok")
    # 文件2：2 行（同批，另一模型）
    led.begin("shaBBBB", str(tmp_path / "b.jsonl"), model="stu-v2",
              pkg_sha="pkg2", threshold=0.6)
    led.add_records([
        ("shaBBBB", 0, "u0", 0.88, "BLOCK"),
        ("shaBBBB", 1, "u1", 0.03, "PASS"),
    ], decided_at=200.0)
    led.finish("shaBBBB", 2, "ok")
    return led


def test_aggregate_counts(tmp_path):
    """_aggregate 汇总：BLOCK/WATCH/PASS + 状态 + 分文件。"""
    led = _seed_ledger(tmp_path)
    meta = led.files_meta()
    records = led.records_range(since=0, until=1e18)
    agg = _aggregate(meta, records)
    assert agg["n_files"] == 2
    assert agg["n_rows_total"] == 5
    assert agg["n_records"] == 5
    assert agg["flag_breakdown"] == {"BLOCK": 2, "WATCH": 1, "PASS": 2}
    assert agg["status_breakdown"] == {"ok": 2}
    # 分文件：shaAAAA 有 1 BLOCK / 1 WATCH / 1 PASS
    per = {e["file_sha"]: e for e in agg["per_file_flag_breakdown"]}
    assert per["shaAAAA"]["BLOCK"] == 1
    assert per["shaAAAA"]["WATCH"] == 1
    # BLOCK 多的排前面
    assert agg["per_file_flag_breakdown"][0]["file_sha"] == "shaAAAA"


def test_export_compliance_json_md(tmp_path):
    """export_compliance 产出 JSON（含 records）+ MD（含表/样例），不碰模型。"""
    led = _seed_ledger(tmp_path)
    out = export_compliance(
        ledger_path=tmp_path / "ledger.sqlite3",
        out_dir=tmp_path / "rep",
        since=0.0, until=0.0,  # 0 = 不限
        render_md=True, render_csv=True, limit_examples=2,
    )
    assert out["n_records"] == 5
    rj = json.loads(open(out["report_json"], encoding="utf-8").read())
    assert rj["summary"]["flag_breakdown"]["BLOCK"] == 2
    assert len(rj["records"]) == 5
    assert len(rj["files"]) == 2
    # 首个 BLOCK 样例在 records 里可回查
    md = open(out["report_md"], encoding="utf-8").read()
    assert "合规审计报告" in md
    assert "BLOCK" in md and "text_sha" in md
    # csv 存在且带表头
    csv_head = open(out["report_csv"], encoding="utf-8-sig").readline().strip()
    assert csv_head.split(",") == ["file_sha", "row_id", "text_sha", "score", "flag", "decided_at"]


def test_export_time_window(tmp_path):
    """时间窗过滤：只导 decided_at 在窗内的记录。"""
    led = _seed_ledger(tmp_path)
    out = export_compliance(
        ledger_path=tmp_path / "ledger.sqlite3",
        out_dir=tmp_path / "rep2",
        since=150.0, until=300.0,
        render_md=False, render_csv=False,
    )
    # 窗 [150,300)：只有文件2的 2 条
    assert out["n_records"] == 2
    rj = json.loads(open(out["report_json"], encoding="utf-8").read())
    assert rj["summary"]["flag_breakdown"]["BLOCK"] == 1
    assert all(r["decided_at"] >= 150.0 and r["decided_at"] < 300.0
               for r in rj["records"])


def test_export_empty_ledger(tmp_path):
    """空台账也能导出一个合法的空报告（不崩）。"""
    lc = IngestLedger(tmp_path / "empty.sqlite3")
    assert lc.n_files() == 0
    out = export_compliance(
        ledger_path=tmp_path / "empty.sqlite3",
        out_dir=tmp_path / "rep3", render_md=True, render_csv=False,
    )
    assert out["n_files"] == 0 and out["n_records"] == 0
    rj = json.loads(open(out["report_json"], encoding="utf-8").read())
    assert rj["summary"]["flag_breakdown"] == {"BLOCK": 0, "WATCH": 0, "PASS": 0}
    # md 里不应崩溃、应显式说明无 BLOCK
    md = open(out["report_md"], encoding="utf-8").read()
    assert "该时间窗内无 BLOCK 记录" in md


def test_ingest_ledger_files_meta_records_range(tmp_path):
    """台账查询接口：files_meta / records_range 形状正确。"""
    led = _seed_ledger(tmp_path)
    meta = led.files_meta()
    assert meta[0]["file_sha"] == "shaAAAA"
    assert meta[0]["model"] == "stu-v1"
    assert meta[0]["pkg_sha"] == "pkg1"
    assert meta[0]["threshold"] == 0.5
    assert meta[0]["status"] == "ok"
    rec = led.records_range(since=0, until=1e18)
    assert len(rec) == 5
    assert rec[0]["flag"] in FLAGS
    assert "text_sha" in rec[0] and "decided_at" in rec[0]
