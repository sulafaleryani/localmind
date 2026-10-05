"""LocalMind: Streamlit UI."""
import hashlib

import streamlit as st

from localmind import embed, ingest, rag
from localmind.config import get_settings
from localmind.llm import LocalMindError
from localmind.store import Store

S = get_settings()
NEW = "+ New collection"


@st.cache_resource
def get_store() -> Store:
    return Store(S.chroma_dir)


def render_sources(citations: list[dict]) -> None:
    with st.expander("Sources"):
        for c in citations:
            st.markdown(f"**[{c['id']}] {c['label']}**  · score {c['score']:.2f}")
            st.code(c["text"], language=None)


def ingest_files(store: Store, collection: str, files) -> None:
    bar = st.progress(0.0, text="Starting…")
    total = 0
    for i, f in enumerate(files):
        bar.progress(i / len(files), text=f"Ingesting {f.name}…")
        try:
            docs = ingest.load_file(f.name, f.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"{f.name}: {e}")
            continue
        if not docs:
            st.warning(f"{f.name}: no text found.")
            continue
        vecs = embed.embed_documents([d.page_content for d in docs])
        h = hashlib.sha1(f.getvalue()).hexdigest()[:10]
        store.add(
            collection,
            [d.page_content for d in docs],
            vecs,
            [d.metadata for d in docs],
            [f"{f.name}-{h}-{d.metadata['chunk_index']}" for d in docs],
        )
        total += len(docs)
    bar.progress(1.0, text="Done")
    st.toast(f"Ingested {total} chunks into '{collection}'", icon="✅")


def sidebar(store: Store) -> tuple[str, str, int]:
    with st.sidebar:
        st.title("LocalMind")
        st.caption("100% local RAG")
        existing = store.list_collections() or ["default"]
        choice = st.selectbox("Collection", existing + [NEW])
        if choice == NEW:
            choice = st.text_input("New collection name", value="").strip() or "default"
        st.session_state["collection"] = choice

        files = st.file_uploader(
            "Upload documents",
            type=sorted(e.lstrip(".") for e in ingest.SUPPORTED_EXTS),
            accept_multiple_files=True,
        )
        if st.button("Ingest", disabled=not files, use_container_width=True):
            ingest_files(store, choice, files)
        st.metric("Chunks in collection", store.count(choice))

        st.divider()
        model = st.text_input("Model", value=S.ollama_model)
        top_k = st.slider("Top-K", 1, 10, S.default_top_k)
        confirm = st.checkbox("Confirm clear")
        if st.button("Clear collection", disabled=not confirm, use_container_width=True):
            store.delete_collection(choice)
            st.session_state.setdefault("history", {}).pop(choice, None)
            st.rerun()
        st.caption("Running locally. No data leaves your machine.")
    return choice, model, top_k


def main() -> None:
    st.set_page_config(page_title="LocalMind", page_icon="🧠", layout="wide")
    store = get_store()
    collection, model, top_k = sidebar(store)
    history = st.session_state.setdefault("history", {}).setdefault(collection, [])

    if store.count(collection) == 0:
        st.info("Upload a file to get started.")

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("citations"):
                render_sources(m["citations"])

    if question := st.chat_input("Ask about your documents…"):
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            box, text, cites = st.empty(), "", []
            try:
                for token, cites in rag.stream_answer(store, collection, question, model, top_k):
                    text += token
                    box.markdown(text + "▌")
                box.markdown(text)
                if cites:
                    render_sources(cites)
                history.append({"role": "assistant", "content": text, "citations": cites})
            except LocalMindError as e:
                box.error(str(e))


if __name__ == "__main__":
    main()
