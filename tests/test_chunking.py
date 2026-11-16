"""长文本分块（P1）纯逻辑测试：不依赖模型/编码器。"""

from __future__ import annotations

import pytest

from poisonguard.chunking import (Chunk, _NUM_UNIT_RE, _hard_split_sentence,
                                  _protected_end, chunk_document, chunk_text,
                                  split_sentences)


def test_split_sentences_basic():
    s = split_sentences("今天天气好。适合出门散步！再写一句？")
    assert len(s) == 3
    assert s[0] == "今天天气好。"


def test_split_sentences_ignores_empty():
    s = split_sentences("第一句。\n\n第二句。")
    assert all(p for p in s)  # 无空段


def test_chunk_text_respects_max_chars():
    text = "。".join(f"第{i}句话内容" for i in range(1, 30)) + "。"
    chunks = chunk_text(text, max_chars=30, overlap_chars=0)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 30 or _hard_split_sentence(c.text, 30)


def test_chunk_text_factpoint_intact():
    # 数字+单位不应被拆到两块
    text = ("公司今年营收达到1300万元，同比增长30%。"
            "这是非常非常重要的一则新闻事实信息。")
    chunks = chunk_text(text, max_chars=16, overlap_chars=2)
    joined = "|".join(c.text for c in chunks)
    assert "1300万" in joined


def test_chunk_text_overlap_carries_tail():
    text = "今天天气很好。明天要下雨了。后天出太阳。大后天台风。"
    chunks = chunk_text(text, max_chars=12, overlap_chars=4)
    assert len(chunks) >= 2
    # 下一块文本应包含上一块尾部
    chunks_raw = [c.text for c in chunks]
    assert chunks_raw[1].startswith(chunks_raw[0][-4:])


def test_split_single_long_sentence_hard_split():
    long = "小明已经连续工作了三十二小时还没有下班非常疲惫。"
    pieces = _hard_split_sentence(long, 10)
    assert len(pieces) >= 2
    assert all(0 < len(p) <= 15 for p in pieces)


def test_protected_end_rides_number_unit():
    text = "营收为1300万元人民币整"
    # 切点在数字+单位中间时右移到单位结束
    idx = text.index("1300万")
    cut = _protected_end(text, idx + 2)
    assert cut >= idx + 4  # 至少跨过整个 1300万
    m = _NUM_UNIT_RE.search("1300万")
    assert m and m.group() == "1300万"


def test_chunk_document_flat_dicts():
    lines = ["第一行：今天天气好。", "第二行：适合出门散步！"]
    out = chunk_document(lines, max_chars=40, overlap_chars=0)
    assert isinstance(out, list)
    assert all(isinstance(c, dict) for c in out)
    assert all("text" in c and "index" in c for c in out)


def test_chunk_document_max_chunks():
    lines = [f"第{i}行句子内容。" for i in range(1, 50)]
    out = chunk_document(lines, max_chars=20, overlap_chars=0, max_chunks=3)
    assert len(out) <= 3


def test_chunk_empty_text():
    assert chunk_text("") == []


def test_chunk_factpoint_count():
    text = "营收1300万元，共32家，占比5%，时间2024年。"
    chunks = chunk_text(text, max_chars=100, overlap_chars=0)
    assert sum(len(c.fact_points) for c in chunks) >= 4