import logging
from typing import Protocol

import numpy as np

log = logging.getLogger("docmind.vector_store")


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, doc_ids: list[str]) -> None: ...
    def query(self, vector: np.ndarray, top_k: int, doc_ids: set[str] | None) -> list[tuple[str, float]]: ...
    def delete(self, ids: list[str]) -> None: ...


class PineconeVectorStore:
    """chunk_id is used as the Pinecone vector id, so results map straight back
    to SQLite rows with no separate id-translation layer."""

    def __init__(self, api_key: str, index_name: str, dimension: int = 384, batch_size: int = 100):
        from pinecone import Pinecone, ServerlessSpec

        self.batch_size = batch_size
        pc = Pinecone(api_key=api_key)
        existing = {i["name"] for i in pc.list_indexes()}
        if index_name not in existing:
            log.info("Creating Pinecone index '%s' (dim=%d)", index_name, dimension)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = pc.Index(index_name)

    def upsert(self, ids: list[str], vectors, doc_ids: list[str]) -> None:
        vecs = [
            {"id": cid, "values": v.tolist(), "metadata": {"doc_id": d}}
            for cid, v, d in zip(ids, vectors, doc_ids)
        ]
        for i in range(0, len(vecs), self.batch_size):
            self.index.upsert(vectors=vecs[i : i + self.batch_size])

    def query(self, vector, top_k: int, doc_ids=None) -> list[tuple[str, float]]:
        flt = {"doc_id": {"$in": sorted(doc_ids)}} if doc_ids else None
        res = self.index.query(vector=vector.tolist(), top_k=top_k, filter=flt)
        return [(m["id"], float(m["score"])) for m in res.get("matches", [])]

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.index.delete(ids=ids)