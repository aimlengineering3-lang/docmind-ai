from docmind.chunking import chunk_segments
from docmind.schemas import Segment


def seg(text, page=1):
    return Segment(doc_id="d1", doc_name="a.pdf", page=page, text=text)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_long_paragraph_splits_with_overlap():
    chunks = chunk_segments([seg(words(500))], max_words=100, overlap_words=20)
    assert len(chunks) == 6
    assert all(c.word_count <= 100 for c in chunks)
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]


def test_paragraphs_are_packed_up_to_limit():
    text = "\n\n".join(words(40, p) for p in "abc")
    chunks = chunk_segments([seg(text)], max_words=100)
    assert [c.word_count for c in chunks] == [80, 40]


def test_chunks_never_cross_pages():
    chunks = chunk_segments([seg(words(50, "a"), page=1), seg(words(50, "b"), page=2)])
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].text.startswith("a0") and chunks[1].text.startswith("b0")


def test_tiny_and_empty_segments_are_dropped():
    chunks = chunk_segments([seg("12"), seg(""), seg(words(30), page=3)])
    assert len(chunks) == 1 and chunks[0].page == 3


def test_chunk_ids_unique_and_citation():
    chunks = chunk_segments([seg(words(500))], max_words=100, overlap_words=20)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].citation == "a.pdf, p. 1"