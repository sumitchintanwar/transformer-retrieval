import time
import random
import numpy as np
import pandas as pd

from bm25_search import BM25Retriever
from hybrid_search import SemanticRetriever, HybridRetriever


NUM_QUERIES = 50
TOP_K = 10
DATA_FILE = "data/msmarco_passages.csv"


def get_sample_queries(n):
    """Sample n random passages from the corpus to use as queries.

    This is a standard benchmarking practice: use actual corpus passages
    as queries so we know relevant documents exist in the index.
    """
    df = pd.read_csv(DATA_FILE)
    samples = df.sample(n=n, random_state=42)
    return samples["passage_text"].tolist()


def benchmark_retriever(retriever, queries, label):
    """Measure average query latency for a retriever.

    Runs each query once and records wall-clock time in milliseconds.
    Returns the average latency across all queries.

    Args:
        retriever: Object with a .search(query, top_k) method.
        queries (list[str]): List of query strings.
        label (str): Name of the method (for logging).

    Returns:
        float: Average latency in milliseconds.
    """
    latencies = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        retriever.search(query, top_k=TOP_K)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            avg = np.mean(latencies)
            print(f"  [{label}] {i + 1}/{len(queries)} done, running avg: {avg:.1f} ms")

    avg_latency = np.mean(latencies)
    return avg_latency


def main():
    print(f"Generating {NUM_QUERIES} sample queries...")
    queries = get_sample_queries(NUM_QUERIES)

    print("\nLoading BM25 retriever...")
    bm25 = BM25Retriever()

    print("Loading semantic retriever...")
    semantic = SemanticRetriever()

    print("Loading hybrid retriever...")
    hybrid = HybridRetriever(bm25, semantic, alpha=0.5)

    print(f"\nBenchmarking with {NUM_QUERIES} queries, top_k={TOP_K}:\n")

    print("Running BM25...")
    bm25_avg = benchmark_retriever(bm25, queries, "BM25")

    print("\nRunning Semantic...")
    sem_avg = benchmark_retriever(semantic, queries, "Semantic")

    print("\nRunning Hybrid...")
    hyb_avg = benchmark_retriever(hybrid, queries, "Hybrid")

    results = [
        ("BM25", bm25_avg),
        ("Semantic", sem_avg),
        ("Hybrid", hyb_avg),
    ]

    print("\n" + "=" * 50)
    print(f"{'Method':<15} {'Avg Latency (ms)':>20}")
    print("-" * 50)
    for method, latency in results:
        print(f"{method:<15} {latency:>18.1f} ms")
    print("=" * 50)

    with open("benchmark_results.md", "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"- **Queries**: {NUM_QUERIES}\n")
        f.write(f"- **Top-K**: {TOP_K}\n")
        f.write(f"- **Corpus**: MS MARCO passages (10K)\n\n")
        f.write("| Method | Average Latency (ms) |\n")
        f.write("|--------|---------------------:|\n")
        for method, latency in results:
            f.write(f"| {method} | {latency:.1f} |\n")

    print("\nResults saved to benchmark_results.md")


if __name__ == "__main__":
    main()
