"""Local embedding model wrapper (sentence-transformers).

Handles the query/passage prefix conventions that bge- and e5-family models
need for good retrieval, and caches a single loaded model per process.
"""
from __future__ import annotations

import functools

import config


@functools.lru_cache(maxsize=2)
def _load(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


class Embedder:
    """Encapsulates prefixing rules so callers just say encode_queries / _passages."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or config.EMBED_MODEL
        self.model = _load(self.model_name)

    def _encode(self, texts, prefix=""):
        payload = [prefix + t for t in texts] if prefix else list(texts)
        vecs = self.model.encode(
            payload,
            batch_size=config.EMBED_BATCH,
            normalize_embeddings=config.EMBED_NORMALIZE,
            show_progress_bar=len(payload) > 64,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def encode_passages(self, texts):
        prefix = "passage: " if _is_e5(self.model_name) else ""
        return self._encode(texts, prefix)

    def encode_queries(self, texts):
        if _is_e5(self.model_name):
            prefix = "query: "
        else:  # bge-family: instruction prefix on queries only
            prefix = config.QUERY_INSTRUCTION
        return self._encode(texts, prefix)
