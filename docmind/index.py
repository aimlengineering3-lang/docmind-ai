import argparse
import logging
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal, Protocol

import faiss
import numpy as np
from pydantic import BaseModel

from .ingest import ingest_file
from .parsing import ParseError
from .schemas import Chunk

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

CHUNK_COLUMNS = "id, chunk_id, doc_id, doc_name, page, section, chunk_index, text, word_count, method"

# Dropped from the keyword query; they only add noise to an OR-style BM25 match.
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
    score: float  # ranking score: RRF (hybrid), cosine (semantic) or -bm25 (keyword)
    semantic_score: float | None = None  # cosine similarity, filled for every hit unless mode=keyword
    keyword_score: float | None = None  # -bm25, only if the chunk matched the keyword query


class Index:
    """SQLite is the single source of truth (chunks, FTS5 keyword index, embeddings).
    FAISS is an in-memory exact index rebuilt from SQLite, so deletes and re-indexing
    never leave the two stores out of sync."""

    def __init__(self, embedder: Embedder, db_path: str | Path = DEFAULT_DB):
        self.embedder = embedder
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._faiss = None
        self._row_ids: list[int] = []
        self._row_docs: list[str] = []
        self._rebuild()

    # ---------- write path ----------
    def _rebuild(self) -> None:
        rows = self.db.execute("SELECT id, doc_id, embedding FROM chunks ORDER BY id").fetchall()
        if not rows:
            self._faiss, self._row_ids, self._row_docs = None, [], []
            return
        mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        self._faiss = faiss.IndexFlatIP(mat.shape[1])
        self._faiss.add(np.ascontiguousarray(mat))
        self._row_ids = [r["id"] for r in rows]
        self._row_docs = [r["doc_id"] for r in rows]

    def has_document(self, doc_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Index the chunks of ONE document."""
        if not chunks:
            return
        vecs = self.embedder.embed_passages([c.text for c in chunks]).astype(np.float32)
        first = chunks[0]
        with self.db:  # single transaction: documents + chunks + FTS (via trigger)
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
        self._rebuild()

    def add_file(self, path: str | Path) -> tuple[str, int]:
        """Parse, chunk, embed and index a file. Returns (doc_id, chunks_added);
        chunks_added is 0 when the document was already indexed or has no extractable text."""
        segments, chunks = ingest_file(path)
        if not segments:
            raise ParseError(f"No content found in '{Path(path).name}'")
        doc_id = segments[0].doc_id
        if self.has_document(doc_id):
            return doc_id, 0
        if not chunks:
            log.warning("'%s' has no extractable text (scanned? OCR comes later)", Path(path).name)
            return doc_id, 0
        self.add_chunks(chunks)
        return doc_id, len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        with self.db:
            cur = self.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self.db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        if cur.rowcount:
            self._rebuild()
        return cur.rowcount > 0

    def documents(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT doc_id, doc_name, n_chunks, added_at FROM documents ORDER BY added_at, doc_name"
        ).fetchall()

    # ---------- read path ----------
    def _semantic(self, qvec: np.ndarray, n: int, doc_ids: set[str] | None) -> list[tuple[int, float]]:
        if self._faiss is None:
            return []
        # Flat index scores every row anyway, so with a doc filter we score all rows and filter after.
        fetch = self._faiss.ntotal if doc_ids else min(n, self._faiss.ntotal)
        scores, idx = self._faiss.search(qvec, fetch)
        out: list[tuple[int, float]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0 or (doc_ids and self._row_docs[i] not in doc_ids):
                continue
            out.append((self._row_ids[i], float(s)))
            if len(out) >= n:
                break
        return out

    def _keyword(self, query: str, n: int, doc_ids: set[str] | None) -> list[tuple[int, float]]:
        tokens = [t for t in re.findall(r"\w+", query.lower()) if t not in STOPWORDS]
        if not tokens:
            return []
        # Quote every token so user input can never be parsed as FTS5 syntax.
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))
        sql = (
            "SELECT c.id AS id, -bm25(chunks_fts) AS score FROM chunks_fts "
            "JOIN chunks c ON c.id = chunks_fts.rowid WHERE chunks_fts MATCH ?"
        )
        params: list = [match]
        if doc_ids:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += sorted(doc_ids)
        sql += " ORDER BY score DESC LIMIT ?"
        params.append(n)
        return [(r["id"], float(r["score"])) for r in self.db.execute(sql, params)]

    def _cosines(self, qvec: np.ndarray, ids: list[int]) -> dict[int, float]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT id, embedding FROM chunks WHERE id IN ({marks})", ids)
        return {r["id"]: float(np.dot(np.frombuffer(r["embedding"], dtype=np.float32), qvec[0])) for r in rows}

    def _fetch_chunks(self, ids: list[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(f"SELECT {CHUNK_COLUMNS} FROM chunks WHERE id IN ({marks})", ids)
        return {r["id"]: Chunk(**{k: r[k] for k in r.keys() if k != "id"}) for r in rows}

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Mode = "hybrid",
        doc_ids: list[str] | None = None,
        pool: int = 20,
    ) -> list[Hit]:
        """Hybrid = reciprocal-rank fusion of the semantic and keyword top-`pool` lists.
        RRF works on ranks, so it needs no score normalisation between cosine and BM25."""
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
            fused: dict[int, float] = defaultdict(float)
            for lst in (sem, kw):
                for rank, (cid, _) in enumerate(lst, start=1):
                    fused[cid] += 1.0 / (RRF_K + rank)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        ranked = ranked[:k]

        if qvec is not None:  # cosine for every returned chunk (needed for abstention thresholds later)
            sem_map.update(self._cosines(qvec, [cid for cid, _ in ranked if cid not in sem_map]))
        chunks = self._fetch_chunks([cid for cid, _ in ranked])
        return [
            Hit(chunk=chunks[cid], score=score, semantic_score=sem_map.get(cid), keyword_score=kw_map.get(cid))
            for cid, score in ranked
        ]


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
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from .embeddings import BgeEmbedder

    index = Index(BgeEmbedder(), args.db)

    if args.cmd == "add":
        for path in args.paths:
            try:
                doc_id, n = index.add_file(path)
            except ParseError as e:
                print(f"error: {e}")
                continue
            status = f"{n} chunks indexed" if n else "skipped (already indexed or no text)"
            print(f"{Path(path).name} [{doc_id}]: {status}")
    elif args.cmd == "list":
        for d in index.documents():
            print(f"{d['doc_id']}  {d['n_chunks']:>4} chunks  {d['doc_name']}")
    elif args.cmd == "delete":
        print("deleted" if index.delete_document(args.doc_id) else "no such document")
    else:
        for rank, h in enumerate(index.search(args.query, args.k, args.mode), start=1):
            sem = f"{h.semantic_score:.3f}" if h.semantic_score is not None else "-"
            kw = f"{h.keyword_score:.2f}" if h.keyword_score is not None else "-"
            print(f"\n#{rank} score={h.score:.4f} cos={sem} bm25={kw}  {h.chunk.citation}")
            print("   " + h.chunk.text[:220].replace("\n", " ") + ("..." if len(h.chunk.text) > 220 else ""))


if __name__ == "__main__":
    main()
