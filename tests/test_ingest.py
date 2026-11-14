"""入库钩子（C2）的测试：全部用桩 scorer，不加载模型、不碰真实语料。

这里验的是"门禁语义"和"幂等"，因为这两件事错了不会报错，只会悄悄把数据放错地方：
- 重复处理 = 同一批数据入账两次（下游统计全错）；
- 分档边界写错 = 该隔离的进了放行区（安全事故），或全进隔离区（不可用）。
"""

from __future__ import annotations

import json
from pathlib import Path

from poisonguard.ingest import IngestLedger, Ingestor, file_sha256, text_sha


class StubScorer:
    """分数由关键词决定：含「毒」=0.9（BLOCK），含「边」=0.47（WATCH），其余 0.1（PASS）。"""

    version = "stub-v1"
    threshold = 0.5

    def __init__(self) -> None:
        self.calls = 0

    def score(self, texts, explain=False):
        self.calls += 1
        out = []
        for t in texts:
            v = 0.9 if "毒" in t else (0.47 if "边" in t else 0.1)
            rec = {"score": v, "flag": v > self.threshold}
            if explain:
                rec["views"] = {"student": v}
            out.append(rec)
        return {"results": out, "n": len(texts),
                "threshold": self.threshold, "model": self.version}


def _ing(tmp_path: Path, **kw) -> Ingestor:
    return Ingestor(StubScorer(), out_dir=tmp_path / "out",
                    ledger_path=tmp_path / "db.sqlite3", **kw)


def _write(p: Path, lines) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


# ---------------------------------------------------------------- 基础
def test_file_sha256_stable_and_distinct(tmp_path: Path):
    a = _write(tmp_path / "a.txt", ["一", "二"])
    b = _write(tmp_path / "b.txt", ["一", "二", "三"])
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)
    assert len(text_sha("同一句话")) == 16


# ---------------------------------------------------------------- 分档
def test_decide_three_bands(tmp_path: Path):
    ing = _ing(tmp_path, watch_band=0.05)
    assert ing.decide(0.10) == "PASS"
    assert ing.decide(0.44) == "PASS", "带外（< 阈值-band）必须放行"
    assert ing.decide(0.45) == "WATCH", "带的下沿（含）算 WATCH"
    assert ing.decide(0.50) == "WATCH", "等于阈值不算 BLOCK（与 scorer.flag 口径一致）"
    assert ing.decide(0.51) == "BLOCK"


def test_decide_binary_when_band_zero(tmp_path: Path):
    ing = _ing(tmp_path, watch_band=0.0)
    assert ing.decide(0.49) == "PASS"
    assert ing.decide(0.51) == "BLOCK"


# ---------------------------------------------------------------- 文件处理
def test_ingest_file_routes_and_writes_jsonl(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["正常的一句话", "网传：这条有毒", "边缘样本", ""])
    res = _ing(tmp_path).ingest_file(f)

    assert res["status"] == "ok"
    assert (res["block"], res["watch"], res["pass"]) == (1, 1, 1)
    assert res["n"] == 3, "空行被过滤，不进入打分"
    for sub, flag in (("quarantine", "BLOCK"), ("review", "WATCH"), ("released", "PASS")):
        files = list((tmp_path / "out" / sub).glob("*.jsonl"))
        assert files, f"{sub} 应该有产出"
        rows = [json.loads(line) for p in files for line in p.read_text(encoding="utf-8").splitlines()]
        assert {r["flag"] for r in rows} == {flag}
        assert all(r["text_sha"] and r["model"] == "stub-v1" and r["threshold"] == 0.5 for r in rows)


def test_ingest_file_is_idempotent_by_sha(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["正常", "有毒"])
    ing = _ing(tmp_path)
    first = ing.ingest_file(f)
    second = ing.ingest_file(f)
    assert first["status"] == "ok" and first["n"] == 2
    assert second["status"] == "skipped"
    assert ing.ledger.counts() == {"PASS": 1, "WATCH": 0, "BLOCK": 1}


def test_restart_ledger_still_skips(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["有毒"])
    _ing(tmp_path).ingest_file(f)
    again = _ing(tmp_path).ingest_file(f)  # 模拟进程重启：新对象、同一个 sqlite
    assert again["status"] == "skipped"


def test_ledger_records_provenance(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["有毒"])
    ing = _ing(tmp_path, pkg_sha="abc123")
    ing.ingest_file(f)
    rows = ing.ledger.counts()
    assert rows["BLOCK"] == 1
    import sqlite3

    with sqlite3.connect(tmp_path / "db.sqlite3") as c:
        model, pkg, thr = c.execute(
            "SELECT model, pkg_sha, threshold FROM files").fetchone()
        score, flag, decided = c.execute(
            "SELECT score, flag, decided_at FROM records").fetchone()
    assert (model, pkg, thr) == ("stub-v1", "abc123", 0.5)
    assert flag == "BLOCK" and abs(score - 0.9) < 1e-9 and decided > 0


def test_row_ids_follow_source_lines(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["第一行", "", "第三行有毒"])
    ing = _ing(tmp_path)
    ing.ingest_file(f)
    import sqlite3

    with sqlite3.connect(tmp_path / "db.sqlite3") as c:
        ids = [r[0] for r in c.execute("SELECT row_id FROM records ORDER BY row_id")]
    assert ids == [0, 2], "空行被过滤，但 row_id 必须保留源文件行号（可回连业务行）"


def test_missing_file_returns_error_not_raise(tmp_path: Path):
    res = _ing(tmp_path).ingest_file(tmp_path / "nope.txt")
    assert res["status"] == "error" and "不存在" in res["error"]


def test_ingest_inbox_patterns_and_max_files(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _write(inbox / "a.txt", ["有毒"])
    _write(inbox / "b.txt", ["正常"])
    _write(inbox / "c.md", ["不是数据文件"])
    ing = _ing(tmp_path)
    rows = ing.ingest_inbox(inbox, patterns=["*.txt"], max_files=1)
    assert len(rows) == 1, "max_files 必须真的截断"
    all_rows = ing.ingest_inbox(inbox, patterns=["*.txt"])
    assert {Path(r["path"]).name for r in all_rows} >= {"a.txt", "b.txt"}
    assert not any(Path(r["path"]).suffix == ".md" for r in all_rows)


def test_empty_file_marked_empty(tmp_path: Path):
    f = _write(tmp_path / "empty.txt", [""])
    res = _ing(tmp_path).ingest_file(f)
    assert res["status"] == "empty" and res["n"] == 0


def test_summary_shape(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["有毒", "正常"])
    ing = _ing(tmp_path)
    ing.ingest_file(f)
    s = ing.summary()
    assert s["files"] == 1 and s["records"]["BLOCK"] == 1 and s["records"]["PASS"] == 1
    assert s["threshold"] == 0.5 and s["model"] == "stub-v1"
