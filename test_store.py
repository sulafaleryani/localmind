import numpy as np

from localmind.store import Store


def test_add_query_order_delete(tmp_path):
    st = Store(str(tmp_path))
    dim = 16
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    st.add("work", [f"doc{i}" for i in range(5)], vecs,
           [{"source": f"f{i}.txt"} for i in range(5)], [f"id{i}" for i in range(5)])  # fmt: skip
    assert st.count("work") == 5 and "work" in st.list_collections()

    res = st.query("work", vecs[3], top_k=3)
    assert res[0]["text"] == "doc3" and res[0]["score"] > 0.99
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)

    st.delete_collection("work")
    assert "work" not in st.list_collections()
