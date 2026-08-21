# Prompt 9 — Verify resume claims

## Claim Verification

| Resume Claim | Verdict | Evidence | Safe Wording |
|---|---|---|---|
| A: Built a semantic and hybrid search engine using transformer embeddings, FAISS dense retrieval, BM25 lexical retrieval, and score fusion. | SUPPORTED | `api.py`, `hybrid_search.py` (Min-Max fusion), `bm25_search.py` (Okapi BM25). | Built a semantic and hybrid search engine using transformer embeddings, FAISS exact dense retrieval, BM25 lexical retrieval, and min-max score fusion. |
| B: Implemented FAISS vector retrieval over approximately 10,000 MS MARCO passages using normalized transformer embeddings and inner-product similarity. | SUPPORTED | `build_index.py` (`normalize_embeddings=True`, `faiss.IndexFlatIP`), `config.py` (`NUM_PASSAGES = 10000`). | Implemented FAISS exact inner-product vector retrieval over 10,000 MS MARCO passages using normalized transformer embeddings. |
| C: Evaluated BM25, semantic, and hybrid retrieval using Precision@10, Recall@10, and MRR@10. | SUPPORTED | `evaluation.py`, `test_metrics.py`. Evaluation artifact `evaluation_report.md` reports P@10, R@10, MRR@10. | Evaluated BM25, semantic, and hybrid retrieval on a sample query set using Precision@10, Recall@10, and MRR@10. |
| D: Achieved low-latency retrieval. | QUALIFIED | `benchmark.py` times end-to-end latency (averaging ~100-300ms on CPU). 100ms is low latency for typical web apps, but this is on a very small 10K corpus with an exhaustive index. | Achieved ~140ms average semantic retrieval latency on a 10K-passage corpus using CPU inference. |

## 1. Best three resume bullets today
- Built an end-to-end hybrid search engine combining exact FAISS dense retrieval (`all-MiniLM-L6-v2`) and Okapi BM25 lexical retrieval, using min-max score fusion.
- Implemented an automated evaluation pipeline measuring Precision@10, Recall@10, and MRR@10, demonstrating hybrid retrieval effectiveness on MS MARCO.
- Engineered a robust, reproducible batch-indexing pipeline using pandas data chunking to maintain constant memory footprint during FAISS vector indexing.

## 2. Claims that must not be made
- **"Large-scale search" or "Scalable ANN"**: The system uses `IndexFlatIP` (exact exhaustive search) which does not scale sublinearly, and the corpus is only 10,000 passages.
- **"Production-ready"**: While it has tests and logging, it lacks persistence layers (e.g., PostgreSQL), concurrent query handling, and API rate limiting.

## 3. Interview defense questions

1. **Why FAISS?** It provides a highly optimized C++ implementation for exact and approximate nearest neighbor search that works seamlessly with numpy.
2. **Why IndexFlatIP?** We had a small 10K corpus where exact exhaustive search is fast enough. By normalizing embeddings, inner product equals cosine similarity, making it mathematically correct and fast.
3. **Why normalize embeddings?** Cosine similarity measures directional alignment regardless of magnitude. Normalizing vectors ensures that inner-product search exactly computes cosine similarity.
4. **Why use BM25?** Dense embeddings can miss exact keyword matches (like specific IDs, acronyms, or rare names). BM25 handles sparse lexical matching efficiently.
5. **Why Min-Max normalization?** BM25 scores are unbounded (often >10), while cosine similarity scores are bounded [-1, 1]. Without normalization, BM25 scores would dominate the fusion.
6. **Why hybrid retrieval?** It combines the semantic understanding of dense embeddings with the exact keyword matching of sparse retrieval, generally leading to higher overall relevance (MRR).
7. **How was MRR calculated?** Mean Reciprocal Rank calculates the reciprocal of the rank of the *first* relevant document retrieved for each query, averaged across all queries.
8. **What are the limitations of the 10K corpus?** It is a toy dataset. Metrics computed on it give signal but are not statistically robust enough to generalize to real-world performance.
9. **What does benchmark latency include?** It measures end-to-end wall-clock latency: passing the query string to the sentence transformer, encoding it, FAISS retrieval, and ID-to-document mapping.
10. **What would you change at 1M documents?** I would replace `IndexFlatIP` with an Approximate Nearest Neighbor (ANN) index like `IndexHNSWFlat`, move ID-to-document mapping to a real database, and potentially separate the embedding server from the search server.

## 4. Final decision
**RESUME READY WITH QUALIFIED WORDING**
The codebase is clean, well-tested, handles edge cases, and processes data out-of-core. The engineering is strong, but the actual scale (10K documents, Flat FAISS) means claims must specify the exact constraints to avoid failing an interview.
