"""retrieve -> prompt -> stream."""
from __future__ import annotations

from collections.abc import Iterator

from localmind import embed, llm
from localmind.llm import LocalMindError
from localmind.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from localmind.store import Store


class EmptyCollectionError(LocalMindError):
    pass


def retrieve(store: Store, collection: str, question: str, top_k: int = 5) -> list[dict]:
    if store.count(collection) == 0:
        raise EmptyCollectionError("This collection is empty. Upload a file to get started.")
    return store.query(collection, embed.embed_query(question), top_k)


def citation_label(meta: dict) -> str:
    label = meta.get("source", "unknown")
    if "page" in meta:
        return f"{label} p.{meta['page']}"
    if "line_start" in meta:
        return f"{label} L{meta['line_start']}-{meta['line_end']}"
    return label


def _header(i: int, meta: dict) -> str:
    h = f"[{i}] source={meta.get('source', 'unknown')}"
    if "page" in meta:
        h += f" page={meta['page']}"
    elif "line_start" in meta:
        h += f" lines={meta['line_start']}-{meta['line_end']}"
    return h


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n---\n".join(
        f"{_header(i, c['metadata'])}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]


def stream_answer(store: Store, collection: str, question: str, model: str,
                  top_k: int = 5) -> Iterator[tuple[str, list[dict]]]:  # fmt: skip
    chunks = retrieve(store, collection, question, top_k)
    citations = [
        {"id": i, "label": citation_label(c["metadata"]), "text": c["text"],
         "score": c["score"], "metadata": c["metadata"]}
        for i, c in enumerate(chunks, 1)
    ]  # fmt: skip
    for token in llm.stream_chat(model, build_prompt(question, chunks)):
        yield token, citations
