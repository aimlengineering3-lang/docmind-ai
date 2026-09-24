import argparse
import logging

from .chunking import chunk_segments
from .parsing import ParseError, parse_document

log = logging.getLogger("docmind.ingest")


def ingest_file(path):
    segments = parse_document(path)
    ocr_pages = [s.page for s in segments if s.needs_ocr]
    if ocr_pages:
        log.warning("Pages needing OCR (skipped for now): %s", ocr_pages)
    return segments, chunk_segments(segments)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Parse and chunk a document")
    ap.add_argument("path")
    ap.add_argument("--show", type=int, default=3, help="chunks to preview")
    args = ap.parse_args()

    try:
        segments, chunks = ingest_file(args.path)
    except ParseError as e:
        raise SystemExit(f"error: {e}")

    avg = sum(c.word_count for c in chunks) / len(chunks) if chunks else 0
    print(f"segments={len(segments)} chunks={len(chunks)} avg_words={avg:.0f}")
    for c in chunks[: args.show]:
        print(f"\n[{c.chunk_id}] {c.citation} ({c.word_count} words)")
        print(c.text[:300] + ("..." if len(c.text) > 300 else ""))


if __name__ == "__main__":
    main()