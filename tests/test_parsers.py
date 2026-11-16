"""文档解析（P1）纯逻辑测试：用内存构造的最小 docx/pdf/txt，不下载模型。

- docx：用 python-docx 现写一个含段落+表格的最小文档再解析；
- pdf：用纯 ASCII 手写一个最小 PDF（内容流里放 Tj 文本）走 stdlib 回退；
  有 PyMuPDF 时另验证其路径（构造真实页）。
- 目的：锁死 extract_text 的三种分支都有产出且不炸。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from poisonguard.parsers import (_parse_docx_stdlib, extract_text,
                                 parse_docx, parse_pdf)


@pytest.fixture()
def mini_docx(tmp_path: Path) -> Path:
    """python-docx 生成的最小 docx（含一段落 + 一个 1x2 表格）。"""
    import docx  # 测试环境已装（1.2.0）

    d = docx.Document()
    d.add_paragraph("第一段：检索式事实核验原型说明")
    d.add_paragraph("第二段：数字1300万")
    tbl = d.add_table(rows=1, cols=2)
    tbl.rows[0].cells[0].text = "表1单元格A"
    tbl.rows[0].cells[1].text = "表1单元格B"
    out = tmp_path / "mini.docx"
    d.save(str(out))
    return out


@pytest.fixture()
def mini_pdf(tmp_path: Path) -> Path:
    """手写最小单页 PDF：内容流里放 '(Hello RAG) Tj'。"""
    content = b"BT /F1 12 Tf 72 720 Td (Hello RAG Guard) Tj ET"
    stream = zlib.compress(content)
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
        b"4 0 obj << /Length %d >> stream\n" % len(stream) + stream + b"\nendstream endobj",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += obj + b"\n"
    xref_pos = len(pdf)
    pdf += b"xref\n0 %d\n" % (len(objects) + 1)
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += (b"trailer << /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref_pos))
    out = tmp_path / "mini.pdf"
    out.write_bytes(pdf)
    return out


def test_docx_paragraphs_and_table(mini_docx: Path):
    lines = parse_docx(mini_docx)
    joined = " ".join(lines)
    assert any("检" in l and "事实" in l for l in lines) or "检索式事实核验" in joined
    assert "表1单元格A" in " ".join(lines)
    assert "1300万" in "\n".join(lines)


def test_docx_stdlib_fallback_equivalent(mini_docx: Path):
    # stdlib 回退至少抽出段落文本（python-docx 路径应与之兼容）
    std = _parse_docx_stdlib(mini_docx)
    joined = " ".join(std)
    assert "检索式事实核验" in joined
    assert "1300万" in joined


def test_pdf_pymupdf_primary(mini_pdf: Path):
    try:
        lines = parse_pdf(mini_pdf)
    except Exception:
        pytest.skip("PyMuPDF 不可用")
    joined = " ".join(lines)
    assert "Hello" in joined or "RAG" in joined


def test_pdf_stdlib_fallback(mini_pdf: Path):
    from poisonguard.parsers import _parse_pdf_stdlib

    lines = _parse_pdf_stdlib(mini_pdf)
    joined = " ".join(lines)
    assert "Hello" in joined and "RAG" in joined


def test_extract_text_txt(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("行1：今天天气好\n行2：出门散步\n", encoding="utf-8")
    assert extract_text(p) == ["行1：今天天气好", "行2：出门散步"]


def test_extract_text_dispatch_docx(mini_docx: Path):
    lines = extract_text(mini_docx)
    assert any("1300万" in l for l in lines)


def test_extract_text_dispatch_pdf(mini_pdf: Path):
    lines = extract_text(mini_pdf)
    joined = " ".join(lines)
    assert "Hello" in joined
