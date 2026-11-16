"""长文本分块（P1 向量库/事实核验适配）：按句切分 + 重叠，保住事实点。

为什么需要：
  - CoROM 编码器 max_length=64 字符，bge=512；给向量库/核验喂长文本时
    若不切块，编码器会截断/丢尾，事实点（数字/日期/事实词）可能被腰斩。
  - 本模块的 chunk 保证：
      1. 优先在句边界（。！？；…换行）切开，块长 ≤ max_chars；
      2. 句子超过单块上限时，在**事实点外**的标点/空格处硬切（不把"1300万"
        这类数字+单位拆到两块）；
      3. 相邻块带 overlap_chars 重叠，保证跨句事实不被漏检。

纯逻辑：不依赖编码器/模型，CI 可直接测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# 中文/英文句子边界（不含消耗型标点本身）
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])")
# 数字+单位（chunk 内需要整体保住，如 1300万 / 30% / 2024年 / 5.2% / 32家）
_NUM_UNIT_RE = re.compile(r"\d+(?:\.\d+)?(?:%|％|万|亿|元|美元|岁|年|月|日|天|小时|分钟|公里|千米|米|吨|公斤|倍|家|户|辆|套|件|个)")


@dataclass
class Chunk:
    index: int          # chunk 序号
    text: str           # 块文本
    start_char: int     # 在原文中的起始字符偏移
    end_char: int       # 结束字符偏移（不含）
    fact_points: list = field(default_factory=list)   # 本块包含的事实计数（数字等）

    def as_dict(self) -> dict:
        return {"index": self.index, "text": self.text,
                "start_char": self.start_char, "end_char": self.end_char,
                "n_factpoints": len(self.fact_points)}


def split_sentences(text: str) -> list[str]:
    """按句边界切分（保留标点于句尾）。空段丢弃。"""
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text or "")]
    return [p for p in parts if p]


def _protected_end(text: str, end: int) -> int:
    """把切点右推到安全位置：不在数字+单位中间。

    return：>= end 的安全切点；找不到安全点则返回 len(text)（整句硬切）。
    """
    for m in _NUM_UNIT_RE.finditer(text):
        s, e = m.start(), m.end()
        if s < end < e:
            # 目标切点落在数字+单位中间 → 右推到单位结束
            return min(e, len(text))
    return end


def _ride_over_facttext(text: str, cut: int) -> int:
    """若切点 cut 落在数字+单位中间，右移到单位结束；否则原样。"""
    return _protected_end(text, cut)


def _hard_split_sentence(text: str, max_chars: int) -> list[str]:
    """单句超长时的兜底切分：括号/标点/空格处切，数字+单位不拆。"""
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end >= n:
            pieces.append(text[start:])
            break
        # 在 [start, end] 里找尽量靠右、且不切在数字+单位上的位置
        cut = end
        for i in range(end - 1, start, -1):
            if text[i] in "，,、:：；;。． \t\n":
                cut = i + 1
                break
        cut = _ride_over_facttext(text, cut)
        if cut <= start:
            cut = end  # 保底：整块硬切
        piece = text[start:cut]
        if piece.strip():
            pieces.append(piece.strip())
        start = cut
    return [p for p in pieces if p.strip()]


def chunk_text(text: str, max_chars: int = 60, overlap_chars: int = 8,
               min_chars: int = 12) -> list[Chunk]:
    """长文本 → 按句切块（度身适配 CoROM 64 字 / bge 512 字窗口）。

    逻辑：
      1. 先按句切；
      2. 句子逐句累积到 > max_chars 就收块（保证单块 ≤ max_chars）；
      3. 单句本身超过 max_chars → 硬切（安全点，数字+单位不拆）；
      4. 下一块先携带上一块尾部 overlap_chars 字符，再继续累积，
         保证跨句事实点不因边界漏检。
    """
    sents = split_sentences(text)
    # 句子（绝对偏移）
    cursor = 0
    sent_spans: list[tuple[str, int]] = []
    for s in sents:
        idx = text.find(s, cursor)
        if idx < 0:
            idx = cursor
        sent_spans.append((s, idx))
        cursor = idx + len(s)

    chunks: list[Chunk] = []
    carry: list[tuple[str, int]] = []
    carry_len = 0
    cur: list[tuple[str, int]] = []
    cur_len = 0
    chunk_start = 0
    for s, off in sent_spans:
        # 单句超长 → 先硬切成段（段首可能携带上一块的重叠尾巴）
        pieces = _hard_split_sentence(s, max_chars) if len(s) > max_chars else [s]
        base_off = off
        for piece in pieces:
            if len(piece) > max_chars:
                # 硬切后仍超长（极端连续长词）：强制 max_chars 切
                piece = piece[:max_chars]
            if cur and cur_len + len(piece) > max_chars:
                chunks.append(_make_chunk(chunks, cur, chunk_start))
                # 下一块携带上一块尾部重叠
                tail = chunks[-1].text[-overlap_chars:] if overlap_chars else ""
                carry = [(tail, max(0, chunk_start + cur_len - len(tail)))][0:1] if tail else []
                carry_len = len(tail)
                cur = list(carry)
                cur_len = carry_len
                chunk_start = base_off - len(tail) if tail else base_off
            cur.append((piece, base_off))
            cur_len += len(piece)
            base_off += len(piece)
    if cur:
        chunks.append(_make_chunk(chunks, cur, chunk_start))
    return chunks


def _make_chunk(chunks: list[Chunk], sents: list[tuple[str, int]],
                start: int) -> Chunk:
    text = "".join(s for s, _ in sents)
    end = start + len(text)
    cp = [m.group(0) for m in _NUM_UNIT_RE.finditer(text)]
    return Chunk(index=len(chunks), text=text, start_char=start,
                 end_char=end, fact_points=cp)


def _take_overlap(chunks: list[Chunk], next_sent: str, overlap: int = 8) -> int:
    """从上一块尾部取 overlap 字符并入下一块（不切事实点）。"""
    if not chunks:
        return 0
    prev = chunks[-1].text
    if overlap >= len(prev):
        return 0
    tail = prev[-overlap:]
    return len(tail)


def chunk_document(lines: Iterable[str], max_chars: int = 60,
                   overlap_chars: int = 8, min_chars: int = 12,
                   max_chunks: int | None = None) -> list[dict]:
    """文档级入口：lines（解析器输出）→ 扁平 chunk 列表（字典）。

    max_chunks 截断（None=不限），便于压测/向量入库前限制量。
    """
    text = "\n".join(l for l in lines if l and l.strip())
    chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars,
                        min_chars=min_chars)
    out = [c.as_dict() for c in chunks]
    if max_chunks is not None:
        out = out[:max_chunks]
    return out
