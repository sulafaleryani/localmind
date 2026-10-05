# LocalMind

> Chat with your documents. 100% local. No API keys. No cloud.

<!-- PLACEHOLDER: 10-second demo GIF here -->
![demo](docs/demo.gif)

## Why

Your documents are private. Your questions are too. LocalMind runs the LLM, the
embeddings, and the vector database on your own machine, so your data never leaves
your laptop. No signup, no API key, no telemetry.

## Features

- Drag-and-drop PDF, Markdown, text, and code files (py, js, ts, java, go, rs, c/cpp, json, yaml, toml)
- Token-by-token streaming answers from a local Ollama model
- Inline citations `[1]`, `[2]` with filename and page / line range
- Expandable **Sources** under every answer showing the raw retrieved chunk
- Multiple collections ("work", "research", "personal")
- Language-aware chunking for code
- Friendly errors: Ollama down, model not pulled, empty collection
- One-command Docker setup, configurable via `.env`

## Quickstart (60 seconds)

```bash
git clone https://github.com/<you>/localmind
cd localmind
make setup
make pull-model
make run
```

Requires Python 3.11+ and [Ollama](https://ollama.com/download). Open http://localhost:8501.

## Docker

```bash
make run-docker
# first run only, pull a model into the ollama container:
docker compose exec ollama ollama pull llama3.2
```

## Configuration

Copy `.env.example` to `.env` (done by `make setup`).

| Variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `CHROMA_DIR` | `./.chroma` |
| `DEFAULT_TOP_K` | `5` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` |

The embedding model downloads once from Hugging Face on first ingest, then runs offline.

## How it works

`ingest` (load + chunk) → `embed` (sentence-transformers) → `store` (ChromaDB, cosine) →
`rag` (retrieve, build numbered-context prompt) → `llm` (Ollama streaming) → `app` (Streamlit).

## Development

```bash
make test    # pytest
make lint    # ruff + black --check
make format  # auto-fix
pre-commit install
```

## License

MIT
