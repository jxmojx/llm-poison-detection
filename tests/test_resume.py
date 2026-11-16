"""断点续跑（环节 30）的测试：行级台账去重、chunked 落盘、--resume 续跑语义。

全部用桩 scorer，不加载模型。核心验证三件事：
- 同一文件处理两遍：第二次要么整文件 skipped（status=ok 命中台账），
  要么 resume 后只处理"台账里还没有的行"（幂等 = 行级 PK，不是内存快照）；
- 用 flush_every=1 强制分块：每块打分后即时写回台账，模拟长文件中断场景；
- 处理了一半的文件（status='running'）再次 ingest（--resume）只补剩余行，
  计数是"历史 + 本次"合并后的全文件口径，不会重复入账。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from poisonguard.ingest import IngestLedger, Ingestor, file_sha256


class DummyScorer:
    """分数：含「毒」=0.9（BLOCK），含「边」=0.47（WATCH），其余 0.1（PASS）。"""

    version = "stub-v1"
    threshold = 0.5

    def __init__(self) -> None:
        self.total = 0  # 累计打过分数的行数（用于断言"没重复打分"）

    def score(self, texts, explain=False):
        self.total += len(texts)
        out = []
        for t in texts:
            v = 0.9 if "毒" in t else (0.47 if "边" in t else 0.1)
            out.append({"score": v, "flag": v > self.threshold,
                        "views": {"student": v} if explain else {}})
        return {"results": out, "n": len(out), "threshold": self.threshold,
                "model": self.version}


def _write(p: Path, lines) -> Path:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def _ing(tmp_path: Path, **kw) -> Ingestor:
    return Ingestor(DummyScorer(), out_dir=tmp_path / "out",
                    ledger_path=tmp_path / "db.sqlite3", **kw)


# ---------------------------------------------------------------- 行级幂等
def test_resume_done_rows_matches_ledger(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["有毒", "正常"])
    ing = _ing(tmp_path)
    ing.ingest_file(f)
    sha = file_sha256(f)
    done = ing.ledger.done_rows(sha)
    assert done == {0, 1}, "台账里的已判决行 = 全部文本行 row_id"


def test_resume_running_file_skips_done_rows(tmp_path: Path):
    """模拟中断现场：台账里已有 row 0/1 的判决但文件 status='running'。"""
    f = _write(tmp_path / "batch.txt", ["第一句毒", "第二句", "第三句毒"])
    ing = _ing(tmp_path, resume=True, flush_every=1)
    res = ing.ingest_file(f)
    assert res["status"] == "ok" and res["n"] == 3

    sha = file_sha256(f)
    # 手工把文件标为 running（模拟进程死掉，没走到 finish('ok') 就被杀掉）
    with sqlite3.connect(tmp_path / "db.sqlite3") as c:
        c.execute("UPDATE files SET status='running' WHERE file_sha=?", (sha,))
    # 再处理：--resume 下应只补剩余行；此时全部行已在台账 => 不重复打分
    again = ing.ingest_file(f)
    assert again["status"] == "resumed"
    assert again["resumed"] == 3 and again["new"] == 0
    # 台账没有新增，计数仍是全文件的
    assert ing.ledger.counts() == {"PASS": 1, "WATCH": 0, "BLOCK": 2}


def test_resume_partial_ledger_adds_only_missing(tmp_path: Path):
    """台账里已有部分行（模拟上次只处理到一半）时，只补未判决的行。"""
    f = _write(tmp_path / "batch.txt", ["有毒A", "正常B", "也有毒C"])
    ing = _ing(tmp_path, resume=True, flush_every=1)
    sha = file_sha256(f)

    # 手工预置：只有 row 0 被判决，文件标记 running（模拟中断现场）
    ing.ledger.begin(sha, str(f), model="stub-v1", pkg_sha="", threshold=0.5)
    ing.ledger.add_records(
        [(sha, 0, "abcde12345", 0.9, "BLOCK")], 123.0)

    res = ing.ingest_file(f)
    assert res["status"] == "ok"
    assert res["resumed"] == 1 and res["new"] == 2
    assert ing.ledger.counts() == {"PASS": 1, "WATCH": 0, "BLOCK": 2}
    # 打分器只收到 2 条（row 0 没被重复打分）
    assert ing.scorer.total == 2


def test_resume_off_does_not_skip(tmp_path: Path):
    """没有 --resume 时，running 文件仍整文件重打（旧行为，幂等靠行级 PK 兜底）。"""
    f = _write(tmp_path / "batch.txt", ["有毒A", "正常B"])
    ing = _ing(tmp_path, flush_every=1)  # resume=False
    sha = file_sha256(f)
    ing.ledger.begin(sha, str(f), model="stub-v1", pkg_sha="", threshold=0.5)
    ing.ledger.add_records([(sha, 0, "abcde12345", 0.9, "BLOCK")], 123.0)
    res = ing.ingest_file(f)
    assert res["status"] == "ok"
    assert res["resumed"] == 0
    # 整文件重打（幂等靠 PK，不重复入账）
    assert ing.ledger.counts() == {"PASS": 1, "WATCH": 0, "BLOCK": 1}


# ---------------------------------------------------------------- chunked 落盘
def test_flush_chunks_commit_partial_records(tmp_path: Path):
    """flush_every=2：每一块打分后就写台账。模拟中断时，已处理块被保留。"""
    f = _write(tmp_path / "batch.txt", [f"毒句{i}" for i in range(5)])
    ing = _ing(tmp_path, flush_every=2)
    sha = file_sha256(f)
    ing.ledger.begin(sha, str(f), model="stub-v1", pkg_sha="", threshold=0.5)
    res = ing.ingest_file(f)
    assert res["status"] == "ok" and res["n"] == 5
    # 5 行全 BLOCK（都含「毒」）
    assert ing.ledger.counts() == {"PASS": 0, "WATCH": 0, "BLOCK": 5}


def test_flush_every_is_optional_and_defaults(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["正常", "有毒"])
    ing = _ing(tmp_path)
    res = ing.ingest_file(f)
    assert res["status"] == "ok"
    assert ing.ledger.counts() == {"PASS": 1, "WATCH": 0, "BLOCK": 1}


# ---------------------------------------------------------------- 文档化的产物
def test_resume_result_shape_and_jsonl(tmp_path: Path):
    f = _write(tmp_path / "batch.txt", ["正常", "有毒"])
    ing = _ing(tmp_path, resume=True, flush_every=1)
    res = ing.ingest_file(f)
    assert res["status"] == "ok"
    assert set(("n", "new", "resumed", "block", "watch", "pass")) <= set(res)
    # 落盘 jsonl 照常产出（每档都有）
    rows = []
    for sub in ("quarantine", "review", "released"):
        for p in (tmp_path / "out" / sub).glob("*.jsonl"):
            rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert {r["flag"] for r in rows} == {"BLOCK", "PASS"}
