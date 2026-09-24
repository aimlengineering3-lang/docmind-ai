import logging

import numpy as np

log = logging.getLogger("docmind.embeddings")


class BgeEmbedder:
    """BAAI/bge-small-en-v1.5 on CPU. Vectors are L2-normalised, so inner product = cosine.

    The model loads lazily, so commands that never embed (list, delete) start instantly.
    """

    # bge-v1.5 retrieval convention: prefix the *query* only, never the passages.
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading embedding model %s (first run downloads ~130 MB)", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return vecs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [self.QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True
        )
        return vec.astype(np.float32)
