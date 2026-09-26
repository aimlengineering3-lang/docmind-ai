import re

import numpy as np
import pytest

from docmind.index import Index
from docmind.schemas import Chunk


class FakeEmbedder:
    DIM = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            v[hash(tok) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, query):
        return self._vec(query)[None, :]


class FakeVectorStore:
    """In-memory Pinecone stand-in: cosine search over stored vectors, no network."""

    def __init__(self):
        self._vecs: dict[str, np.ndarray] = {}
        self._docs: dict[str, str] = {}

    def upsert(self, ids, vectors, doc_ids):
        for cid, v, d in zip(ids, vectors, doc_ids):
            self._vecs[cid] = v
            self._docs[cid] = d

    def query(self, vector, top_k, doc_ids=None):
        scored = [
            (cid, float(np.dot(v, vector)))
            for cid, v in self._vecs.items()
            if not doc_ids or self._docs[cid] in doc_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, ids):
        for cid in ids:
            self._vecs.pop(cid, None)
            self._docs.pop(cid, None)


TEXTS = [
    "Payment is due within 30 days of the invoice date.",
    "Either party may terminate this agreement with 60 days written notice.",
    "The invoice number is INV-2024-0042 issued by Acme Corp.",
    "The office is closed on public holidays.",
]


def make_chunks(doc_id="d1", name="a.pdf", texts=TEXTS):
    return [
        Chunk(chunk_id=f"{doc_id}:{i:04d}", doc_id=doc_id, doc_name=name, page=i + 1,
              chunk_index=i, text=t, word_count=len(t.split()))
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def idx():
    index = Index(FakeEmbedder(), FakeVectorStore(), ":memory:")
    index.add_chunks(make_chunks())
    return index


def test_keyword_finds_exact_identifier(idx):
    hits = idx.search("INV-2024-0042", k=3, mode="keyword")
    assert hits[0].chunk.page == 3
    assert hits[0].keyword_score is not None


def test_semantic_finds_overlapping_text(idx):
    hits = idx.search("terminate agreement written notice", k=1, mode="semantic")
    assert hits[0].chunk.page == 2


def test_hybrid_fills_cosine_for_every_hit(idx):
    hits = idx.search("payment due days", k=3)
    assert hits[0].chunk.page == 1
    assert all(h.semantic_score is not None for h in hits)


def test_doc_filter(idx):
    idx.add_chunks(make_chunks("d2", "b.pdf", ["Payment terms are net 45 days for all invoices here."]))
    hits = idx.search("payment days", k=5, doc_ids=["d2"])
    assert hits and {h.chunk.doc_id for h in hits} == {"d2"}


def test_delete_removes_from_both_indexes(idx):
    assert idx.delete_document("d1")
    assert idx.search("payment", mode="keyword") == []
    assert idx.search("payment", mode="semantic") == []
    assert not idx.delete_document("d1")


def test_fts_syntax_in_query_does_not_crash(idx):
    idx.search('payment" OR ( NEAR', mode="keyword")


def test_document_text_concatenates_in_order(idx):
    text = idx.document_text("d1")
    assert text.index("Payment is due") < text.index("terminate this agreement")


def test_empty_index_returns_nothing():
    assert Index(FakeEmbedder(), FakeVectorStore(), ":memory:").search("anything") == []