"""人审台（C1）的测试：队列读取 + 判决落库 + HTTP 契约。

不测浏览器渲染，只测"数据进得去、判得下、回流得出来"这三件事——
UI 可以改版，这三条变了就是回归。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poisonguard.review import (DECISIONS, REASONS, ReviewStore, create_review_app,
                                item_key, load_items)


def _rec(source: str, row_id: int, text: str, score: float, flag: str,
         tsha: str = "abc", model: str = "stub") -> dict:
    return {"source": source, "row_id": row_id, "text": text, "text_sha": tsha,
            "score": score, "flag": flag, "views": {"student": score},
            "model": model, "threshold": 0.5, "decided_at": 1.0}


def _write_queue(out: Path, block: list[dict], watch: list[dict]) -> Path:
    for sub, rows in (("quarantine", block), ("review", watch)):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "20261115.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8", newline="\n")
    return out


# ---------------------------------------------------------------- 队列
def test_load_items_reads_both_flags_and_sorts_by_score(tmp_path: Path):
    out = _write_queue(tmp_path,
                       [_rec("/d/a.txt", 0, "低分", 0.60, "BLOCK")],
                       [_rec("/d/b.txt", 1, "高分", 0.95, "WATCH", tsha="def")])
    items = load_items(out)
    assert [i["text"] for i in items] == ["高分", "低分"], "必须按分数降序"
    assert {i["flag"] for i in items} == {"BLOCK", "WATCH"}
    assert all(i["key"] for i in items)


def test_load_items_dedupes_same_source_row_text(tmp_path: Path):
    rec = _rec("/d/a.txt", 3, "同一条", 0.7, "BLOCK")
    out = _write_queue(tmp_path, [rec, dict(rec)], [])
    assert len(load_items(out)) == 1, "同一 (source,row,text_sha) 只能出现一次"


def test_load_items_filters_flag_and_decided(tmp_path: Path):
    out = _write_queue(tmp_path,
                       [_rec("/d/a.txt", 0, "BLOCK 项", 0.9, "BLOCK")],
                       [_rec("/d/b.txt", 1, "WATCH 项", 0.6, "WATCH", tsha="d2")])
    assert len(load_items(out, flags=("BLOCK",))) == 1
    only = load_items(out, flags=("WATCH",))[0]
    assert only["text"] == "WATCH 项"
    assert load_items(out, decided={item_key(only)}) == [] or \
        all(i["key"] != item_key(only) for i in load_items(out, decided={item_key(only)}))


def test_load_items_pagination(tmp_path: Path):
    block = [_rec("/d/a.txt", i, f"第{i}条", 0.9 - i * 0.01, "BLOCK", tsha=f"t{i}")
             for i in range(5)]
    out = _write_queue(tmp_path, block, [])
    page1 = load_items(out, limit=2, offset=0)
    page2 = load_items(out, limit=2, offset=2)
    assert [p["row_id"] for p in page1] == [0, 1]
    assert [p["row_id"] for p in page2] == [2, 3]


def test_missing_dirs_is_empty_not_crash(tmp_path: Path):
    assert load_items(tmp_path / "nope") == []


# ---------------------------------------------------------------- 判决
def test_decide_upserts_and_counts(tmp_path: Path):
    store = ReviewStore(tmp_path / "r.sqlite3")
    item = _rec("/d/a.txt", 0, "一条", 0.9, "BLOCK")
    item["key"] = item_key(item)
    store.decide(item, "benign", reason="口语闲聊")
    store.decide(item, "poison", reason="内容篡改(事实)", note="改判")
    s = store.stats()
    assert s["total"] == 1, "同一个 key 只能有一条最新判决"
    assert s["by_decision"]["poison"] == 1 and s["by_decision"]["benign"] == 0
    rows = store.all_rows()
    assert rows[0]["note"] == "改判" and rows[0]["reason"] == "内容篡改(事实)"


def test_decide_rejects_unknown_decision_and_reason(tmp_path: Path):
    store = ReviewStore(tmp_path / "r.sqlite3")
    item = _rec("/d/a.txt", 0, "x", 0.5, "BLOCK")
    with pytest.raises(ValueError, match="decision"):
        store.decide(item, "maybe")
    with pytest.raises(ValueError, match="reason"):
        store.decide(item, "benign", reason="内容篡改(事实)")  # 属于 poison 的码


def test_reason_codes_are_disjoint_and_required_shape():
    assert set(REASONS) == set(DECISIONS)
    for k, v in REASONS.items():
        assert v and all(isinstance(x, str) and x for x in v), f"{k} 理由码不能为空"


def test_export_csv_has_header_and_rows(tmp_path: Path):
    store = ReviewStore(tmp_path / "r.sqlite3")
    item = _rec("/d/a.txt", 0, "一条", 0.9, "BLOCK")
    item["key"] = item_key(item)
    store.decide(item, "poison", reason="模板投毒")
    p = store.export_csv(tmp_path / "out.csv")
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("key,source,row_id")
    assert len(lines) == 2 and "模板投毒" in lines[1]


# ---------------------------------------------------------------- HTTP
def test_http_queue_and_decision_contract(tmp_path: Path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    out = _write_queue(tmp_path,
                       [_rec("/d/a.txt", 0, "可疑句", 0.93, "BLOCK")],
                       [_rec("/d/b.txt", 1, "边缘句", 0.55, "WATCH", tsha="d2")])
    app = create_review_app(out, tmp_path / "r.sqlite3")
    c = TestClient(app)

    r = c.get("/api/queue")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2 and len(data["items"]) == 2
    assert data["reasons"] == REASONS

    first = data["items"][0]
    ok = c.post("/api/decision", json={"key": first["key"], "decision": "poison",
                                       "reason": "内容篡改(事实)", "item": first})
    assert ok.status_code == 200, ok.text
    assert ok.json()["decision"] == "poison"

    after = c.get("/api/queue").json()
    assert after["total"] == 1, "判过的必须从队列里消失"
    assert after["stats"]["by_decision"]["poison"] == 1

    bad = c.post("/api/decision", json={"key": first["key"], "decision": "nope", "item": first})
    assert bad.status_code == 400

    csv = c.get("/api/export.csv")
    assert csv.status_code == 200 and "decision" in csv.text


def test_index_page_renders(tmp_path: Path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    app = create_review_app(tmp_path, tmp_path / "r.sqlite3", title="测试人审台")
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "测试人审台" in r.text and "/api/decision" in r.text
