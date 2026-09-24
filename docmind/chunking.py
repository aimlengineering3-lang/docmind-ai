import re

from .schemas import Chunk, Segment


def _split_long(words: list[str], max_words: int, overlap: int) -> list[str]:
    """Sliding window for a single paragraph that exceeds max_words."""
    step = max_words - overlap
    windows = []
    for start in range(0, len(words), step):
        windows.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return windows


def chunk_segment(
    text: str, max_words: int = 180, overlap_words: int = 30
) -> list[str]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    units: list[str] = []
    for p in paragraphs:
        words = p.split()
        if len(words) > max_words:
            units.extend(_split_long(words, max_words, overlap_words))
        else:
            units.append(p)

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0
    for unit in units:
        n = len(unit.split())
        if buffer and buffer_words + n > max_words:
            chunks.append("\n\n".join(buffer))
            buffer, buffer_words = [], 0
        buffer.append(unit)
        buffer_words += n
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def chunk_segments(
    segments: list[Segment],
    max_words: int = 180,
    overlap_words: int = 30,
    min_words: int = 5,
) -> list[Chunk]:
    """Chunk each segment independently so a chunk never spans pages/sections."""
    chunks: list[Chunk] = []
    for seg in segments:
        if not seg.text:
            continue  # e.g. scanned page awaiting OCR
        for text in chunk_segment(seg.text, max_words, overlap_words):
            word_count = len(text.split())
            if word_count < min_words:
                continue  # page numbers, stray labels
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{seg.doc_id}:{idx:04d}",
                    doc_id=seg.doc_id,
                    doc_name=seg.doc_name,
                    page=seg.page,
                    section=seg.section,
                    chunk_index=idx,
                    text=text,
                    word_count=word_count,
                    method=seg.method,
                )
            )
    return chunks