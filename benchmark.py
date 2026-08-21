import time
import json
import numpy as np
import pandas as pd

from bm25_search import BM25Retriever
from hybrid_search import SemanticRetriever, HybridRetriever
import config


def run_benchmark(retriever_name, retriever, queries, top_k=config.DEFAULT_TOP_K, warmup=2, reps=5):
    """Measure average query latency for a retriever.

    Runs each query once and records wall-clock time in milliseconds.
    Returns the average latency across all queries.
    """
    # Warmup
    for query in queries[:warmup]:
        retriever.search(query, top_k=top_k)

    latencies = []
    for i, query in enumerate(queries):
        start = time.perf_counter()
        retriever.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        if (i + 1) % 10 == 0:
            avg = np.mean(latencies)
            print(f"  [{retriever_name}] {i + 1}/{len(queries)} done, running avg: {avg:.1f} ms")

    avg_latency = np.mean(latencies)
    return avg_latency


def main():
    print(f"Loading {config.BENCHMARK_QUERIES} queries for benchmarking...")
    with open(config.EVAL_FILE) as f:
        eval_data = json.load(f)
    queries = [q["query"] for q in eval_data[:config.BENCHMARK_QUERIES]]

    print("Loading retrievers...")
    bm25 = BM25Retriever(data_path=config.PROCESSED_DATA_FILE)
    semantic = SemanticRetriever(
        model_name=config.MODEL_NAME,
        index_path=config.FAISS_INDEX_FILE,
        data_path=config.PROCESSED_DATA_FILE,
    )
    hybrid = HybridRetriever(bm25, semantic, alpha=config.HYBRID_ALPHA)

    print(f"\nBenchmarking with {len(queries)} queries, top_k={config.DEFAULT_TOP_K}:\n")

    print("Running BM25...")
    bm25_avg = run_benchmark("BM25", bm25, queries)

    print("\nRunning Semantic...")
    sem_avg = run_benchmark("Semantic", semantic, queries)

    print("\nRunning Hybrid...")
    hyb_avg = run_benchmark("Hybrid", hybrid, queries)

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
        f.write(f"- **Queries**: {len(queries)}\n")
        f.write(f"- **Top-K**: {config.DEFAULT_TOP_K}\n")
        f.write(f"- **Corpus Size**: {len(bm25.passage_ids)}\n\n")
        f.write("| Method | Average Latency (ms) |\n")
        f.write("|--------|---------------------:|\n")
        for method, latency in results:
            f.write(f"| {method} | {latency:.1f} |\n")

    print("\nResults saved to benchmark_results.md")


if __name__ == "__main__":
    main()
