import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .schemas import Segment

# A PDF page with fewer extractable characters than this is treated as scanned.
MIN_CHARS_FOR_TEXT_PAGE = 30
# DOCX text without any headings is cut into segments of about this many words,
# so citations can still point to a paragraph range instead of just the file.
MAX_HEADINGLESS_WORDS = 500


class ParseError(Exception):
    """Raised when a file cannot be parsed (corrupt, encrypted, unsupported)."""


def file_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm_for_repeat(paragraph: str) -> str:
    # "Page 3 of 10" and "Page 4 of 10" must count as the same footer.
    return re.sub(r"\d+", "#", paragraph.lower()).strip()


def _strip_repeated(
    pages: list[list[str]],
    min_pages: int = 3,
    ratio: float = 0.5,
    max_len: int = 120,
    edge: int = 2,
) -> list[list[str]]:
    """Drop running headers/footers: short paragraphs near the top or bottom of a
    page whose normalised text shows up on at least `ratio` of all pages."""
    if len(pages) < min_pages:
        return pages

    def edge_idx(n: int) -> set[int]:
        return set(range(min(edge, n))) | set(range(max(n - edge, 0), n))

    counts: Counter[str] = Counter()
    for paras in pages:
        counts.update(
            {_norm_for_repeat(paras[i]) for i in edge_idx(len(paras)) if len(paras[i]) <= max_len}
        )
    repeated = {key for key, c in counts.items() if c >= ratio * len(pages)}

    cleaned = []
    for paras in pages:
        idx = edge_idx(len(paras))
        cleaned.append(
            [
                p
                for i, p in enumerate(paras)
                if not (i in idx and len(p) <= max_len and _norm_for_repeat(p) in repeated)
            ]
        )
    return cleaned


def parse_pdf(path: Path, doc_id: str) -> list[Segment]:
    try:
        pdf = pymupdf.open(path)
    except Exception as e:
        raise ParseError(f"Cannot open PDF '{path.name}': {e}") from e

    with pdf:
        if pdf.needs_pass:
            raise ParseError(f"PDF '{path.name}' is password-protected")

        pages: list[list[str]] = []
        for page in pdf:
            # Each text block (type 0) is roughly a paragraph; sort=True gives reading order.
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [" ".join(b[4].split()) for b in blocks if b[6] == 0]
            pages.append([p for p in paragraphs if p])

    segments = []
    for page_no, paragraphs in enumerate(_strip_repeated(pages), start=1):
        text = clean_text("\n\n".join(paragraphs))
        segments.append(
            Segment(
                doc_id=doc_id,
                doc_name=path.name,
                page=page_no,
                text=text,
                needs_ocr=len(text) < MIN_CHARS_FOR_TEXT_PAGE,
            )
        )
    return segments


def _iter_blocks(doc):
    """Yield paragraphs and tables in document order (python-docx doesn't do this natively)."""
    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _table_to_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            value = " ".join(cell.text.split())
            if not cells or value != cells[-1]:  # merged cells repeat; collapse them
                cells.append(value)
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(path: Path, doc_id: str) -> list[Segment]:
    try:
        doc = Document(str(path))
    except Exception as e:
        raise ParseError(f"Cannot open DOCX '{path.name}': {e}") from e

    segments: list[Segment] = []
    heading: str | None = None
    buffer: list[str] = []
    buf_words = 0
    para_no = 0  # counts non-empty paragraphs and tables
    start_para = 1

    def flush() -> None:
        nonlocal buf_words, start_para
        text = clean_text("\n\n".join(buffer))
        if text:
            # Before the first heading (or in heading-less files) cite a paragraph range.
            section = heading or f"paras {start_para}-{para_no}"
            segments.append(
                Segment(doc_id=doc_id, doc_name=path.name, section=section, text=text)
            )
        buffer.clear()
        buf_words = 0
        start_para = para_no + 1

    for block in _iter_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            style = block.style.name if block.style is not None else ""
            if style.startswith("Heading") or style == "Title":
                flush()
                heading = text
            para_no += 1
            buffer.append(text)
            buf_words += len(text.split())
        else:
            table_text = _table_to_text(block)
            if not table_text.strip():
                continue
            para_no += 1
            buffer.append(table_text)
            buf_words += len(table_text.split())
        if heading is None and buf_words >= MAX_HEADINGLESS_WORDS:
            flush()
    flush()
    return segments


PARSERS = {".pdf": parse_pdf, ".docx": parse_docx}


def parse_document(path: str | Path) -> list[Segment]:
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"File not found: {path}")
    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ParseError(f"Unsupported file type: '{path.suffix}'")
    return parser(path, file_id(path))
