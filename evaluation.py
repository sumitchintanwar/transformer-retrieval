import json
import numpy as np
from bm25_search import BM25Retriever
from hybrid_search import SemanticRetriever, HybridRetriever


EVAL_FILE = "evaluation_dataset.json"
TOP_K = 10


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of top-k retrieved items that are relevant.

    Args:
        retrieved_ids: List of passage IDs returned by the retriever.
        relevant_ids: Set of ground-truth relevant passage IDs.
        k: Number of top results to consider.

    Returns:
        float: Precision@k score in [0, 1].
    """
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(1 for pid in retrieved_at_k if pid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Fraction of relevant items that appear in top-k results.

    Args:
        retrieved_ids: List of passage IDs returned by the retriever.
        relevant_ids: Set of ground-truth relevant passage IDs.
        k: Number of top results to consider.

    Returns:
        float: Recall@k score in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    retrieved_at_k = retrieved_ids[:k]
    hits = sum(1 for pid in retrieved_at_k if pid in relevant_ids)
    return hits / len(relevant_ids)


def mrr_at_k(retrieved_ids, relevant_ids, k):
    """Mean Reciprocal Rank — 1/rank of the first relevant result.

    If the first relevant result is at position r (1-indexed),
    the score is 1/r. If no relevant result is in top-k, score is 0.

    Args:
        retrieved_ids: List of passage IDs returned by the retriever.
        relevant_ids: Set of ground-truth relevant passage IDs.
        k: Number of top results to consider.

    Returns:
        float: MRR@k score in [0, 1].
    """
    retrieved_at_k = retrieved_ids[:k]
    for rank, pid in enumerate(retrieved_at_k, 1):
        if pid in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retriever(retriever, eval_data, k=TOP_K):
    """Run a retriever on all evaluation queries and compute metrics.

    Args:
        retriever: Object with a .search(query, top_k) method.
        eval_data: List of dicts with 'query' and 'relevant_passage_ids'.
        k: Cutoff for precision, recall, and MRR.

    Returns:
        dict: Average precision@k, recall@k, mrr@k across all queries.
    """
    precisions = []
    recalls = []
    mrrs = []

    for item in eval_data:
        query = item["query"]
        relevant = set(item["relevant_passage_ids"])

        results = retriever.search(query, top_k=k)
        retrieved_ids = [r["passage_id"] for r in results]

        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        mrrs.append(mrr_at_k(retrieved_ids, relevant, k))

    return {
        "precision@k": np.mean(precisions),
        "recall@k": np.mean(recalls),
        "mrr@k": np.mean(mrrs),
    }


def main():
    with open(EVAL_FILE) as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} evaluation queries\n")

    print("Loading BM25 retriever...")
    bm25 = BM25Retriever()

    print("Loading semantic retriever...")
    semantic = SemanticRetriever()

    print("Loading hybrid retriever...")
    hybrid = HybridRetriever(bm25, semantic, alpha=0.5)

    retrievers = {
        "BM25": bm25,
        "Semantic": semantic,
        "Hybrid": hybrid,
    }

    all_results = {}
    for name, retriever in retrievers.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_retriever(retriever, eval_data, k=TOP_K)
        all_results[name] = metrics
        print(
            f"  P@{TOP_K}={metrics['precision@k']:.4f}  "
            f"R@{TOP_K}={metrics['recall@k']:.4f}  "
            f"MRR@{TOP_K}={metrics['mrr@k']:.4f}"
        )

    # Find best method per metric
    best_p = max(all_results, key=lambda m: all_results[m]["precision@k"])
    best_r = max(all_results, key=lambda m: all_results[m]["recall@k"])
    best_m = max(all_results, key=lambda m: all_results[m]["mrr@k"])

    # Generate report
    report = f"""# Evaluation Report

## Setup

- **Corpus**: MS MARCO passages (10K)
- **Queries**: {len(eval_data)}
- **Top-K**: {TOP_K}
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Hybrid alpha**: 0.5

## Results

| Method | Precision@{TOP_K} | Recall@{TOP_K} | MRR@{TOP_K} |
|--------|:{'':>13s}|:{'':>12s}|:{'':>10s}|
"""

    for name in ["BM25", "Semantic", "Hybrid"]:
        m = all_results[name]
        report += f"| {name} | {m['precision@k']:.4f} | {m['recall@k']:.4f} | {m['mrr@k']:.4f} |\n"

    report += f"""
## Best Per Metric

- **Precision@{TOP_K}**: {best_p} ({all_results[best_p]['precision@k']:.4f})
- **Recall@{TOP_K}**: {best_r} ({all_results[best_r]['recall@k']:.4f})
- **MRR@{TOP_K}**: {best_m} ({all_results[best_m]['mrr@k']:.4f})

## Analysis

"""

    # Add analysis based on actual results
    scores = {name: all_results[name]["precision@k"] + all_results[name]["recall@k"] + all_results[name]["mrr@k"] for name in all_results}
    overall_best = max(scores, key=scores.get)

    report += f"""The **{overall_best}** retriever achieves the strongest overall performance across all three metrics.

"""
    if all_results["Semantic"]["mrr@k"] > all_results["BM25"]["mrr@k"]:
        report += """Semantic search tends to rank relevant documents higher (better MRR), meaning users find what they need faster in the result list.

"""
    if all_results["BM25"]["recall@k"] > all_results["Semantic"]["recall@k"]:
        report += """BM25 achieves higher recall, retrieving more of the total relevant documents within the top-10 cutoff.

"""
    if all_results["Hybrid"]["precision@k"] > all_results["BM25"]["precision@k"] and all_results["Hybrid"]["precision@k"] > all_results["Semantic"]["precision@k"]:
        report += """Hybrid fusion improves precision by combining lexical matching (BM25) with semantic understanding, filtering out noise from each individual method.

"""
    report += """### Recommendations

1. **Use Semantic search** for natural language queries where meaning matters more than exact keywords.
2. **Use BM25** for keyword-heavy queries where exact term matching is critical.
3. **Use Hybrid** as the default for general-purpose retrieval — it balances precision and recall.
4. Consider tuning the `alpha` parameter in HybridRetriever to optimize for your specific query distribution.
"""

    with open("evaluation_report.md", "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"{'Method':<12} {'P@10':>8} {'R@10':>8} {'MRR@10':>8}")
    print("-" * 60)
    for name in ["BM25", "Semantic", "Hybrid"]:
        m = all_results[name]
        print(f"{name:<12} {m['precision@k']:>8.4f} {m['recall@k']:>8.4f} {m['mrr@k']:>8.4f}")
    print("=" * 60)
    print(f"\nBest overall: {overall_best}")
    print("Report saved to evaluation_report.md")


if __name__ == "__main__":
    main()
