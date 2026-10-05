from localmind.ingest import load_file, simple_split, split_text


def test_chunks_overlap_and_size():
    text = " ".join(f"word{i}" for i in range(600))
    chunks = split_text(text, ".txt", size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()) & set(b.split()), "adjacent chunks should overlap"


def test_simple_splitter_fallback():
    text = "\n\n".join("sentence number %d. " % i * 10 for i in range(30))
    chunks = simple_split(text, 300, 50, ["\n\n", "\n", " ", ""])
    assert len(chunks) > 1 and all(len(c) <= 300 for c in chunks)


def test_metadata_text_file():
    data = ("line of text\n" * 200).encode()
    docs = load_file("notes.md", data)
    assert docs
    for i, d in enumerate(docs):
        m = d.metadata
        assert m["source"] == "notes.md" and m["chunk_index"] == i
        assert m["filetype"] == "md" and m["ingested_at"]
        assert m["line_start"] >= 1 and m["line_end"] >= m["line_start"]
