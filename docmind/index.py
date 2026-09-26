import argparse
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel

from .config import get_settings
from .ingest import ingest_file
from .parsing import ParseError
from .schemas import Chunk
from .vector_store import VectorStore

log = logging.getLogger("docmind.index")

DEFAULT_DB = os.environ.get("DOCMIND_DB", "data/docmind.db")
RRF_K = 60  # standard reciprocal-rank-fusion constant

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id   TEXT PRIMARY KEY,
    doc_name TEXT NOT NULL,
    n_chunks INTEGER NOT NULL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id    TEXT NOT NULL UNIQUE,
    doc_id      TEXT NOT NULL,
    doc_name    TEXT NOT NULL,
    page        INTEGER,
    section     TEXT,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    word_count  INTEGER NOT NULL,
    method      TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""

CHUNK_COLUMNS = "chunk_id, doc_id, doc_name, page, section, chunk_index, text, word_count, method"

STOPWORDS = frozenset(
    "a an and are as at be by did do does for from how in is it of on or that the this to "
    "was were what when where which who with".split()
)

Mode = Literal["hybrid", "semantic", "keyword"]


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


class Hit(BaseModel):
    chunk: Chunk
    score: float
    semantic_score: float | None = None
    keyword_score: float | None = None


class Index:
    """SQLite holds chunk text, metadata, the FTS5 keyword index, and a local
    embedding cache. Pinecone holds the same vectors as the managed semantic
    store. SQLite stays the source of truth for text/citations; Pinecone is
    swappable (interview point: only vector_store.py would change)."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, db_path: str | Path = DEFAULT_DB):
        self.embedder = embedder
        self.vector_store = vector_store
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def has_document(self, doc_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        import time

        t0 = time.perf_counter()
        vecs = self.embedder.embed_passages([c.text for c in chunks]).astype(np.float32)
        t1 = time.perf_counter()
        log.info("Embedding %d chunks took %.2fs", len(chunks), t1 - t0)
        first = chunks[0]
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO documents(doc_id, doc_name, n_chunks) VALUES (?,?,?)",
                (first.doc_id, first.doc_name, len(chunks)),
            )
            self.db.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, doc_name, page, section, chunk_index,"
                " text, word_count, method, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (c.chunk_id, c.doc_id, c.doc_name, c.page, c.section, c.chunk_index,
                     c.text, c.word_count, c.method, v.tobytes())
                    for c, v in zip(chunks, vecs)
                ],
            )
        t2 = time.perf_counter()
        self.vector_store.upsert(
            ids=[c.chunk_id for c in chunks], vectors=vecs, doc_ids=[c.doc_id for c in chunks]
        )
        log.info("Pinecone upsert took %.2fs", time.perf_counter() - t2)

    def add_file(self, path: str | Path) -> tuple[str, int, str]:
        """status is one of: 'indexed', 'duplicate', 'no_text' (likely scanned)."""
        segments, chunks = ingest_file(path)
        if not segments:
            raise ParseError(f"No content found in '{Path(path).name}'")
        doc_id = segments[0].doc_id
        if self.has_document(doc_id):
            return doc_id, 0, "duplicate"
        if not chunks:
            log.warning("'%s' has no extractable text (scanned? OCR comes later)", Path(path).name)
            return doc_id, 0, "no_text"
        self.add_chunks(chunks)
        return doc_id, len(chunks), "indexed"

    def delete_document(self, doc_id: str) -> bool:
        rows = self.db.execute("SELECT chunk_id FROM chunks WHERE doc_id=?", (doc_id,)).fetchall()
        chunk_ids = [r["chunk_id"] for r in rows]
        with self.db:
            cur = self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        if chunk_ids:
            self.vector_store.delete(chunk_ids)
        return cur.rowcount > 0

    def document_text(self, doc_id: str) -> str:
        rows = self.db.execute(
            "SELECT text FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return "\n\n".join(r["text"] for r in rows)

    def documents(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT doc_id, doc_name, n_chunks, added_at FROM documents ORDER BY added_at, doc_name"
        ).fetchall()

    def _semantic(self, qvec: np.ndarray, n: int, doc_ids: set[str] | None) -> list[tuple[str, float]]:
        return self.vector_store.query(qvec[0], top_k=n, doc_ids=doc_ids)

    def _keyword(self, query: str, n: int, doc_ids: set[str] | None) -> list[tuple[str, float]]:
        tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
        sql = (
            "SELECT c.chunk_id AS chunk_id, -bm25(chunks_fts) AS score FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
        )
        params: list = [match]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += sorted(doc_ids)
        sql += " ORDER BY score DESC LIMIT ?"
        params.append(n)
        return [(r["chunk_id"], float(r["score"])) for r in self.db.execute(sql, params)]

    def _cosines(self, qvec: np.ndarray, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(f"SELECT chunk_id, embedding FROM chunks WHERE chunk_id IN ({marks})", chunk_ids)
        return {r["chunk_id"]: float(np.dot(np.frombuffer(r["embedding"], dtype=np.float32), qvec[0])) for r in rows}

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE chunk_id IN ({marks})", chunk_ids)
        return {r["chunk_id"]: Chunk(**dict(r)) for r in rows}

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        doc_ids: list[str] | None = None,
        pool: int = 20,
    ) -> list[Hit]:
        allowed = set(doc_ids) if doc_ids else None
        qvec = None
        if mode != "keyword":
            qvec = np.ascontiguousarray(self.embedder.embed_query(query), dtype=np.float32)
        sem = self._semantic(qvec, pool, allowed) if qvec is not None else []
        kw = self._keyword(query, pool, allowed) if mode != "semantic" else []
        sem_map, kw_map = dict(sem), dict(kw)

        if mode == "semantic":
            ranked = sem
        elif mode == "keyword":
            ranked = kw
        else:
            fused: dict[str, float] = defaultdict(float)
            for lst in (sem, kw):
                for rank, (cid, _) in enumerate(lst, start=1):
                    fused[cid] += 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:k]

        if qvec is not None:
            sem_map.update(self._cosines(qvec, [cid for cid, _ in ranked if cid not in sem_map]))
        chunks = self._fetch_chunks([cid for cid, _ in ranked])
        return [
            Hit(chunk=chunks[cid], score=score, semantic_score=sem_map.get(cid), keyword_score=kw_map.get(cid))
            for cid, score in ranked
            if cid in chunks
        ]


def build_index(db: str = DEFAULT_DB) -> Index:
    from .embeddings import BgeEmbedder
    from .vector_store import PineconeVectorStore

    settings = get_settings()
    vs = PineconeVectorStore(api_key=settings.pinecone_api_key, index_name=settings.pinecone_index)
    return Index(BgeEmbedder(), vs, db)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m docmind.index")
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="parse + index files")
    add.add_argument("paths", nargs="+")
    sub.add_parser("list", help="list indexed documents")
    dele = sub.add_parser("delete", help="remove a document by doc_id")
    dele.add_argument("doc_id")
    srch = sub.add_parser("search", help="retrieve chunks for a query")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=5)
    srch.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    ask = sub.add_parser("ask", help="ask a grounded question")
    ask.add_argument("query")
    ask.add_argument("-k", type=int, default=5)
    ext = sub.add_parser("extract", help="structured extraction from a document")
    ext.add_argument("doc_id")
    ext.add_argument("doc_type", choices=["invoice", "resume", "contract"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = build_index(args.db)

    if args.cmd == "add":
        for path in args.paths:
            try:
                doc_id, n, status = index.add_file(path)
            except ParseError as e:
                print(f"error: {e}")
                continue
            label = f"{n} chunks indexed" if status == "indexed" else status
            print(f"{Path(path).name} [{doc_id}]: {label}")
    elif args.cmd == "list":
        for d in index.documents():
            print(f"{d['doc_id']}  {d['n_chunks']:>4} chunks  {d['doc_name']}")
    elif args.cmd == "delete":
        print("deleted" if index.delete_document(args.doc_id) else "no such document")
    elif args.cmd == "ask":
        from .config import get_settings
        from .llm import GeminiProvider
        from .qa import answer_question

        settings = get_settings()
        llm = GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        result = answer_question(index, llm, args.query, k=args.k)
        print(result.answer)
        if result.citations:
            print("\nSources: " + ", ".join(result.citations))
    elif args.cmd == "extract":
        from .config import get_settings
        from .extraction import SCHEMAS
        from .llm import GeminiProvider

        text = index.document_text(args.doc_id)
        if not text:
            print("error: no such document, or it has no extracted text")
            return
        settings = get_settings()
        llm = GeminiProvider(settings.gemini_api_key, settings.gemini_model)
        try:
            result = llm.extract_structured(text, SCHEMAS[args.doc_type])
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"error: extraction failed ({e})")
    else:
        for rank, h in enumerate(index.search(args.query, args.k, args.mode), start=1):
            sem = f"{h.semantic_score:.3f}" if h.semantic_score is not None else "-"
            kw = f"{h.keyword_score:.2f}" if h.keyword_score is not None else "-"
            print(f"\n#{rank} score={h.score:.4f} cos={sem} bm25={kw}  {h.chunk.citation}")
            print("   " + h.chunk.text[:220].replace("\n", " ") + ("..." if len(h.chunk.text) > 220 else ""))


if __name__ == "__main__":
    main()