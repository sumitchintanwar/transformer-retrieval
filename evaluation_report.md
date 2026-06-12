# Evaluation Report

## Setup

- **Corpus**: MS MARCO passages (10K)
- **Queries**: 50
- **Top-K**: 10
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Hybrid alpha**: 0.5

## Results

| Method | Precision@10 | Recall@10 | MRR@10 |
|--------|-------------:|----------:|-------:|
| BM25 | 0.1360 | 0.6090 | 0.4411 |
| Semantic | 0.1840 | 0.7933 | 0.5247 |
| Hybrid | 0.1720 | 0.7360 | 0.5539 |

## Best Per Metric

- **Precision@10**: Semantic (0.1840)
- **Recall@10**: Semantic (0.7933)
- **MRR@10**: Hybrid (0.5539)

## Analysis

The **Semantic** retriever achieves the strongest overall performance, leading on both Precision@10 (0.184) and Recall@10 (0.793).

Key observations:

- **Semantic search** dominates on precision and recall — it retrieves more relevant documents and a higher fraction of its results are relevant. This is expected since the all-MiniLM-L6-v2 model captures meaning beyond keyword overlap.
- **Hybrid** achieves the best MRR@10 (0.554) — its fusion of lexical and semantic signals pushes the first relevant result higher in the ranking than either method alone. This matters most for user experience, where the top result is critical.
- **BM25** lags behind on all metrics, confirming that pure keyword matching is insufficient for this query set.

### Recommendations

1. **Use Hybrid as the default** — it achieves the best MRR, meaning users find relevant results faster in the list.
2. **Use Semantic** when recall is critical — it retrieves the most relevant documents overall.
3. **Use BM25** only for exact keyword matching scenarios (e.g., part numbers, proper nouns).
4. Tune `alpha` in HybridRetriever to shift the balance between lexical and semantic signals.
