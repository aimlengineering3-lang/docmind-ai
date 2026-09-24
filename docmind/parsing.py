import hashlib
import re
import unicodedata
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .schemas import Segment

# A PDF page with fewer extractable characters than this is treated as scanned.
MIN_CHARS_FOR_TEXT_PAGE = 30


class ParseError(Exception):
    """Raised when a file cannot be parsed (corrupt, encrypted, unsupported)."""


def file_id(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(path: Path, doc_id: str) -> list[Segment]:
    try:
        pdf = pymupdf.open(path)
    except Exception as e:
        raise ParseError(f"Cannot open PDF '{path.name}': {e}") from e

    with pdf:
        if pdf.needs_pass:
            raise ParseError(f"PDF '{path.name}' is password-protected")

        segments = []
        for page_no, page in enumerate(pdf, start=1):
            # Each text block (type 0) is roughly a paragraph; sort=True gives reading order.
            blocks = page.get_text("blocks", sort=True)
            paragraphs = [" ".join(b[4].split()) for b in blocks if b[6] == 0]
            text = clean_text("\n\n".join(p for p in paragraphs if p))
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

    def flush() -> None:
        text = clean_text("\n\n".join(buffer))
        if text:
            segments.append(
                Segment(doc_id=doc_id, doc_name=path.name, section=heading, text=text)
            )
        buffer.clear()

    for block in _iter_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            style = block.style.name if block.style is not None else ""
            if style.startswith("Heading") or style == "Title":
                flush()
                heading = text or heading
            if text:
                buffer.append(text)
        else:
            table_text = _table_to_text(block)
            if table_text.strip():
                buffer.append(table_text)
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