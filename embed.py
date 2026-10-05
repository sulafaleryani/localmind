"""Singleton wrapper around SentenceTransformer."""
from __future__ import annotations

import threading

import numpy as np

from localmind.config import get_settings


class Embedder:
    _instance: Embedder | None = None
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    @classmethod
    def get(cls) -> Embedder:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(get_settings().embed_model)
            return cls._instance

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        )


def embed_documents(texts: list[str]) -> np.ndarray:
    return Embedder.get().encode(texts)


def embed_query(text: str) -> np.ndarray:
    # bge models work best with this instruction on queries
    q = "Represent this sentence for searching relevant passages: " + text
    return Embedder.get().encode([q])[0]
