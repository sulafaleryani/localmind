"""Settings loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    chroma_dir: str = "./.chroma"
    default_top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        d = cls()
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", d.ollama_host),
            ollama_model=os.getenv("OLLAMA_MODEL", d.ollama_model),
            embed_model=os.getenv("EMBED_MODEL", d.embed_model),
            chroma_dir=os.getenv("CHROMA_DIR", d.chroma_dir),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", d.default_top_k)),
            chunk_size=int(os.getenv("CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", d.chunk_overlap)),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
