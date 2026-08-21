from bm25_search import BM25Retriever
from hybrid_search import HybridRetriever, SemanticRetriever


def print_results(label, results, max_text=100):
    """Pretty-print a list of search results."""
    for rank, r in enumerate(results, 1):
        score_key = "final_score" if "final_score" in r else "score"
        score = r[score_key]
        extra = ""
        if "bm25_score" in r:
            extra = f" (bm25={r['bm25_score']:.4f}, sem={r['semantic_score']:.4f})"
        print(f"  #{rank} [{label}] score={score:.4f}{extra}  id={r['passage_id']}")
        print(f"     {r['passage_text'][:max_text]}...")
    print()


def main():
    print("Loading BM25 retriever...")
    bm25 = BM25Retriever()

    print("Loading semantic retriever...")
    semantic = SemanticRetriever()

    print("Loading hybrid retriever (alpha=0.5)...")
    hybrid = HybridRetriever(bm25, semantic, alpha=0.5)

    queries = [
        "photosynthesis",
        "machine learning",
        "reserve bank of australia",
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        bm25_results = bm25.search(query, top_k=5)
        semantic_results = semantic.search(query, top_k=5)
        hybrid_results = hybrid.search(query, top_k=5)

        print("--- BM25 (lexical) ---")
        print_results("BM25", bm25_results)

        print("--- Semantic (dense) ---")
        print_results("SEM", semantic_results)

        print("--- Hybrid (alpha=0.5) ---")
        print_results("HYB", hybrid_results)


if __name__ == "__main__":
    main()
