import importlib


def test_import_modules_and_app():
    for m in ("config", "ingest", "embed", "store", "llm", "rag", "prompts"):
        importlib.import_module(f"localmind.{m}")
    app = importlib.import_module("app")
    assert callable(app.main)


def test_prompt_format():
    from localmind.rag import build_prompt

    msgs = build_prompt("q?", [
        {"text": "A", "metadata": {"source": "foo.pdf", "page": 3}},
        {"text": "B", "metadata": {"source": "bar.md"}},
    ])  # fmt: skip
    ctx = msgs[1]["content"]
    assert "[1] source=foo.pdf page=3" in ctx and "[2] source=bar.md" in ctx
    assert "---" in ctx and msgs[0]["role"] == "system"
