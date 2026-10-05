"""File loaders and chunking."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from localmind.config import get_settings

TEXT_EXTS = {".txt", ".md"}
CODE_EXTS = {
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h",
    ".json", ".yaml", ".yml", ".toml",
}  # fmt: skip
SUPPORTED_EXTS = {".pdf"} | TEXT_EXTS | CODE_EXTS

_LANG_MAP = {
    ".py": "PYTHON", ".js": "JS", ".ts": "TS", ".java": "JAVA", ".go": "GO",
    ".rs": "RUST", ".cpp": "CPP", ".c": "CPP", ".h": "CPP", ".md": "MARKDOWN",
}  # fmt: skip
_CODE_SEPS = ["\nclass ", "\ndef ", "\nfunc ", "\nfn ", "\n\n", "\n", " ", ""]
_TEXT_SEPS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------- splitting
def _merge(pieces: list[str], sep: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    for p in pieces:
        if cur and len(sep.join(cur)) + len(sep) + len(p) > size:
            chunks.append(sep.join(cur))
            while cur and (
                len(sep.join(cur)) > overlap
                or len(sep.join(cur)) + len(sep) + len(p) > size
            ):
                cur.pop(0)
        cur.append(p)
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def simple_split(text: str, size: int, overlap: int, seps: list[str]) -> list[str]:
    """Minimal recursive character splitter (fallback)."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    for i, sep in enumerate(seps):
        if sep == "":
            step = max(1, size - overlap)
            return [text[j : j + size] for j in range(0, len(text), step)]
        if sep in text:
            pieces: list[str] = []
            for part in filter(None, text.split(sep)):
                if len(part) > size:
                    pieces.extend(simple_split(part, size, overlap, seps[i + 1 :]))
                else:
                    pieces.append(part)
            return _merge(pieces, sep, size, overlap)
    return [text]


def split_text(text: str, ext: str = ".txt", size: int | None = None,
               overlap: int | None = None) -> list[str]:  # fmt: skip
    s = get_settings()
    size = size or s.chunk_size
    overlap = s.chunk_overlap if overlap is None else overlap
    try:
        from langchain_text_splitters import (
            Language,
            RecursiveCharacterTextSplitter,
        )

        lang = _LANG_MAP.get(ext)
        if lang:
            splitter = RecursiveCharacterTextSplitter.from_language(
                Language[lang], chunk_size=size, chunk_overlap=overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size, chunk_overlap=overlap
            )
        return splitter.split_text(text)
    except ImportError:
        seps = _CODE_SEPS if ext in CODE_EXTS else _TEXT_SEPS
        return simple_split(text, size, overlap, seps)


# ----------------------------------------------------------------- loading
def _line_range(text: str, chunk: str, cursor: int) -> tuple[int, int, int]:
    idx = text.find(chunk[:80], cursor)
    if idx == -1:
        idx = cursor
    start = text.count("\n", 0, idx) + 1
    return start, start + chunk.count("\n"), idx


def load_file(name: str, data: bytes) -> list[Document]:
    """Parse + chunk one file (given as bytes) into Documents."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext or name}")
    now = datetime.now(timezone.utc).isoformat()
    docs: list[Document] = []

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        idx = 0
        for pno, page in enumerate(reader.pages, start=1):
            for chunk in split_text(page.extract_text() or "", ext):
                docs.append(Document(chunk, {
                    "source": name, "page": pno, "chunk_index": idx,
                    "filetype": "pdf", "ingested_at": now,
                }))  # fmt: skip
                idx += 1
    else:
        text = data.decode("utf-8", errors="replace")
        cursor = 0
        for idx, chunk in enumerate(split_text(text, ext)):
            ls, le, pos = _line_range(text, chunk, cursor)
            cursor = max(cursor, pos)
            docs.append(Document(chunk, {
                "source": name, "chunk_index": idx, "filetype": ext.lstrip("."),
                "line_start": ls, "line_end": le, "ingested_at": now,
            }))  # fmt: skip
    return docs
