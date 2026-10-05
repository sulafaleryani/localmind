"""ChromaDB wrapper."""
from __future__ import annotations

import re

import chromadb

from localmind.config import get_settings


def normalize_name(name: str) -> str:
    """Chroma requires 3-63 chars, alphanumeric start/end."""
    n = re.sub(r"[^a-zA-Z0-9._-]", "-", name.strip()).strip("-._")[:63]
    return n if len(n) >= 3 else (n + "-col")[:63]


class Store:
    def __init__(self, path: str | None = None):
        self.client = chromadb.PersistentClient(path=path or get_settings().chroma_dir)

    def _col(self, name: str):
        return self.client.get_or_create_collection(
            normalize_name(name), metadata={"hnsw:space": "cosine"}
        )

    def add(self, collection, documents, embeddings, metadatas, ids) -> None:
        self._col(collection).upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(map(float, e)) for e in embeddings],
            metadatas=list(metadatas),
        )

    def query(self, collection, query_embedding, top_k: int = 5) -> list[dict]:
        col = self._col(collection)
        n = col.count()
        if n == 0:
            return []
        res = col.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": t, "metadata": m, "score": 1.0 - d}
            for t, m, d in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def delete_collection(self, collection) -> None:
        try:
            self.client.delete_collection(normalize_name(collection))
        except Exception:  # noqa: BLE001 - already gone
            pass

    def list_collections(self) -> list[str]:
        return sorted(c if isinstance(c, str) else c.name
                      for c in self.client.list_collections())  # fmt: skip

    def count(self, collection) -> int:
        return self._col(collection).count()
