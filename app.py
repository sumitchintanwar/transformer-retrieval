import streamlit as st

from api import SearchEngineAPI


@st.cache(allow_output_mutation=True)
def get_api():
    """Instantiate and cache the Search Engine API."""
    # Pre-loading internal retrievers so they are ready for the user
    api = SearchEngineAPI()
    api._load_lookup()
    api._get_bm25()
    api._get_semantic()
    api._get_hybrid()
    return api


def display_result(rank, result, mode):
    """Display a single search result based on the retrieval mode."""
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
    api = get_api()

    st.title("Semantic Search Engine")
    st.caption("Powered by Sentence Transformers + FAISS | MS MARCO Passages")

    user_input = st.text_area("Search box", "how does photosynthesis work")

    st.sidebar.markdown("**Filters**")
    search_mode = st.sidebar.selectbox("Search Mode", ["Semantic", "BM25", "Hybrid"])
    num_results = st.sidebar.slider("Number of search results", 5, 50, 10)

    if user_input:
        if search_mode == "Semantic":
            results = api.search_semantic(user_input, top_k=num_results)
            for rank, r in enumerate(results, 1):
                display_result(rank, r, "Semantic")

        elif search_mode == "BM25":
            results = api.search_bm25(user_input, top_k=num_results)
            for rank, r in enumerate(results, 1):
                display_result(rank, r, "BM25")

        else:
            results = api.search_hybrid(user_input, top_k=num_results)
            for rank, r in enumerate(results, 1):
                display_result(rank, r, "Hybrid")


if __name__ == "__main__":
    main()
