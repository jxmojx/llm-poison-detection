"""PDF/Word 文档解析（P1 文档入库前置）：把 docx/pdf/txt 统一抽成纯文本。

- `.docx`：优先 python-docx（段落+表格）；不可用时回退纯 stdlib 的 zipfile+xml
  （只解 word/document.xml 的 <w:t> 文本，按 <w:p> 分段——不依赖任何第三方包）。
- `.pdf`：优先 PyMuPDF（`import pymupdf`，fitz 已弃用改名）；回退用 re 扫
  PDF 内容流里的 `(...) Tj/TJ` 文本操作符（只覆盖简单生成型 PDF，用于无重编译环境）。
- 其它扩展名/无扩展名：按 UTF-8 文本直读；解码失败则 latin-1 保底。

所有路径都返回 list[str]（段落级），方便直接喂分块器/事实核验/向量库。
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

# 命名空间：word/document.xml 的段落与文本节点
_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

# PDF 简单文本操作符：(...) Tj 与 [...] TJ
_PDF_TEXT_RE = re.compile(rb"\((?:\\.|[^()\\])*\)\s*Tj|\[(?:\\.|[^()\\])*\]\s*TJ")
_PDF_UNESCAPE = {b"\\(": b"(", b"\\)": b")", b"\\\\": b"\\", b"\\n": b"\n", b"\\t": b"\t"}


def _parse_docx_python_docx(path: Path) -> list[str]:
    """python-docx 路径：段落 + 表格单元格都抽出来。"""
    import docx

    d = docx.Document(str(path))
    lines: list[str] = []
    for para in d.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            row_text = " ".join(c for c in cells if c)
            if row_text:
                lines.append(row_text)
    return lines


def _parse_docx_stdlib(path: Path) -> list[str]:
    """纯 stdlib 回退：只解 document.xml 的 w:t 文本，按 w:p 分段。"""
    lines: list[str] = []
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            root = ET.fromstring(f.read())
    for para in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        text = "".join(
            t.text or "" for t in para.iter(
                "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
        ).strip()
        if text:
            lines.append(text)
    return lines


def parse_docx(path: Path) -> list[str]:
    """docx → 段落列表。

    优先 python-docx；无该依赖时回退 zipfile+xml（stdlib only）。
    """
    try:
        import docx  # noqa: F401

        return _parse_docx_python_docx(path)
    except Exception:
        try:
            return _parse_docx_stdlib(path)
        except Exception as e:
            raise ValueError(f"docx 解析失败（python-docx 与 stdlib 回退均失败）: {e}") from e


def _parse_pdf_pymupdf(path: Path, max_pages: int | None = None) -> list[str]:
    """PyMuPDF 路径（`import pymupdf` 优先；旧名 fitz 兜底）。"""
    try:
        import pymupdf as pdfmod        # PyMuPDF >= 1.23 的新名
    except Exception:
        import fitz as pdfmod           # noqa: F401  # 旧名（弃用但兼容）
    doc = pdfmod.open(str(path))
    lines: list[str] = []
    pages = doc if max_pages is None else doc[:max_pages]
    for page in pages:
        text = page.get_text("text")
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    doc.close()
    return lines


def _parse_pdf_stdlib(path: Path) -> list[str]:
    """纯 stdlib 回退：抓简单 PDF 内容流里的 Tj/TJ 文本操作符。

    只覆盖文本流未压缩的生成型 PDF；压缩流（FlateDecode）无法解时给出
    清晰的报错（提示装 PyMuPDF）。
    """
    try:
        import zlib
    except Exception:
        zlib = None

    raw = path.read_bytes()
    # PDF 里内容流 = 介于 stream ... endstream 之间的块（可能被 FlateDecode 压缩）
    bodies = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", raw, re.DOTALL)
    text_parts: list[str] = []
    for body in bodies:
        data = body
        if zlib is not None:
            try:
                data = zlib.decompress(data)
            except Exception:
                pass
        for m in _PDF_TEXT_RE.finditer(data):
            tok = m.group(0)
            # 去掉操作符后缀，只留括号内容
            if b"Tj" in tok:
                inner = tok[: tok.rfind(b")") + 1]
            else:
                inner = tok[: tok.rfind(b"]") + 1]
            s = inner[1:-1].decode("latin-1", errors="replace")
            for k, v in _PDF_UNESCAPE.items():
                s = s.replace(k.decode("latin-1"), v.decode("latin-1"))
            s = s.strip()
            if s:
                text_parts.append(s)
    return text_parts


def parse_pdf(path: Path, max_pages: int | None = None) -> list[str]:
    """pdf → 段落列表。优先 PyMuPDF；回退 stdlib 简单解析。

    PyMuPDF 对畸形/简单 PDF 可能不抛异常但返回空（内容流句法告警），
    这时也落到 stdlib 路径兜底，避免"主解析器静默吞内容"。
    """
    try:
        out = _parse_pdf_pymupdf(path, max_pages=max_pages)
        if out:
            return out
    except Exception as e1:
        last = f"PyMuPDF: {e1}"
    else:
        last = "PyMuPDF 返回空文本"
    try:
        return _parse_pdf_stdlib(path)
    except Exception as e2:
        raise ValueError(f"pdf 解析失败（PyMuPDF 与 stdlib 回退均失败）: {last} | {e2}") from e2


def extract_text(path: str | Path, *, max_pages: int | None = None) -> list[str]:
    """按扩展名抽取文档段落（统一入口，落入库前清洗）。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        return parse_docx(p)
    if suffix == ".pdf":
        return parse_pdf(p, max_pages=max_pages)
    # 其它：按文本读
    data = p.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("gb18030", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines
