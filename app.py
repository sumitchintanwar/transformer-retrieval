import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search
from bm25_search import BM25Retriever
from hybrid_search import SemanticRetriever, HybridRetriever
import config


@st.cache
def read_data(data=config.PROCESSED_DATA_FILE):
    """Read the preprocessed passage data."""
    return pd.read_csv(data)


@st.cache(allow_output_mutation=True)
def load_model(name=config.MODEL_NAME):
    """Instantiate the sentence transformer model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss=config.FAISS_INDEX_FILE):
    """Load and deserialize the FAISS index."""
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


@st.cache(allow_output_mutation=True)
def load_bm25_retriever():
    """Instantiate the BM25 retriever."""
    return BM25Retriever()


@st.cache(allow_output_mutation=True)
def load_semantic_retriever():
    """Instantiate the semantic retriever."""
    return SemanticRetriever()


@st.cache(allow_output_mutation=True)
def load_hybrid_retriever():
    """Instantiate the hybrid retriever."""
    bm25 = load_bm25_retriever()
    semantic = load_semantic_retriever()
    return HybridRetriever(bm25, semantic, alpha=config.HYBRID_ALPHA)


def display_result(rank, result, mode):
    """Display a single search result based on the retrieval mode."""
    passage_id = result["passage_id"]
    passage_text = result["passage_text"]
    word_count = result.get("word_count", "")

    if mode == "Semantic":
        score = result.get("score", 0.0)
        header = f"**#{rank}** | Cosine Similarity: `{score:.4f}`"
    elif mode == "BM25":
        score = result.get("score", 0.0)
        header = f"**#{rank}** | BM25 Score: `{score:.4f}`"
    else:
        final = result.get("final_score", 0.0)
        bm25_s = result.get("bm25_score", 0.0)
        sem_s = result.get("semantic_score", 0.0)
        header = (
            f"**#{rank}** | Combined: `{final:.4f}` "
            f"| BM25: `{bm25_s:.4f}` "
            f"| Semantic: `{sem_s:.4f}`"
        )

    if word_count:
        header += f" | Words: {word_count}"

    st.markdown(header)
    st.write(passage_text)
    st.divider()


def main():
    data = read_data()

    st.title("Semantic Search Engine")
    st.caption("Powered by Sentence Transformers + FAISS | MS MARCO Passages")

    user_input = st.text_area("Search box", "how does photosynthesis work")

    st.sidebar.markdown("**Filters**")
    search_mode = st.sidebar.selectbox("Search Mode", ["Semantic", "BM25", "Hybrid"])
    num_results = st.sidebar.slider("Number of search results", 5, 50, 10)

    if user_input:
        passage_lookup = data.set_index("passage_id")

        if search_mode == "Semantic":
            model = load_model()
            faiss_index = load_faiss_index()
            D, I = vector_search([user_input], model, faiss_index, num_results)
            for rank, (dist, pid) in enumerate(
                zip(D.flatten().tolist(), I.flatten().tolist()), 1
            ):
                if pid not in passage_lookup.index:
                    continue
                row = passage_lookup.loc[pid]
                result = {
                    "passage_id": int(pid),
                    "score": float(dist),
                    "passage_text": row["passage_text"],
                    "word_count": row["word_count"],
                }
                display_result(rank, result, "Semantic")

        elif search_mode == "BM25":
            bm25 = load_bm25_retriever()
            results = bm25.search(user_input, top_k=num_results)
            for rank, r in enumerate(results, 1):
                pid = r["passage_id"]
                if pid not in passage_lookup.index:
                    continue
                row = passage_lookup.loc[pid]
                r["word_count"] = row["word_count"]
                r["passage_text"] = row["passage_text"]
                display_result(rank, r, "BM25")

        else:
            hybrid = load_hybrid_retriever()
            results = hybrid.search(user_input, top_k=num_results)
            for rank, r in enumerate(results, 1):
                pid = r["passage_id"]
                if pid not in passage_lookup.index:
                    continue
                row = passage_lookup.loc[pid]
                r["word_count"] = row["word_count"]
                r["passage_text"] = row["passage_text"]
                display_result(rank, r, "Hybrid")


if __name__ == "__main__":
    main()
