from typing import Literal

from pydantic import BaseModel


class Segment(BaseModel):
    """A unit of extracted text with a citable location: a PDF page or a DOCX section."""

    doc_id: str
    doc_name: str
    page: int | None = None  # 1-based; None for formats without pages (DOCX)
    section: str | None = None  # nearest heading, used for DOCX
    text: str
    method: Literal["text", "ocr"] = "text"
    needs_ocr: bool = False  # page had (almost) no extractable text


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int | None = None
    section: str | None = None
    chunk_index: int  # position within the document
    text: str
    word_count: int
    method: Literal["text", "ocr"] = "text"

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.doc_name}, p. {self.page}"
        if self.section:
            return f"{self.doc_name}, § {self.section}"
        return self.doc_name