"""Ollama client with streaming."""
from __future__ import annotations

from collections.abc import Iterator

from localmind.config import get_settings


class LocalMindError(Exception):
    """User-facing error."""


def stream_chat(model: str, messages: list[dict]) -> Iterator[str]:
    import httpx
    import ollama

    client = ollama.Client(host=get_settings().ollama_host)
    try:
        for part in client.chat(model=model, messages=messages, stream=True):
            tok = part["message"]["content"]
            if tok:
                yield tok
    except (ConnectionError, httpx.ConnectError) as e:
        raise LocalMindError(
            "Ollama isn't running. Start it with `ollama serve` "
            "or run `make run-docker`."
        ) from e
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404 or "not found" in str(e).lower():
            raise LocalMindError(
                f"Model '{model}' isn't pulled. Run `ollama pull {model}` "
                "(or `make pull-model`)."
            ) from e
        raise LocalMindError(f"Ollama error: {e}") from e
