# Semantic Search Engine

A robust, configurable search engine implementing lexical, dense semantic, and hybrid retrieval. It leverages Sentence Transformers for generating dense vectors, FAISS for efficient inner-product vector search, and `rank-bm25` for sparse lexical retrieval. A hybrid fusion approach combines both modalities using Min-Max normalization to provide highly relevant search results over a curated MS MARCO passage dataset.

## Architecture

The system follows a scalable pipeline:

```text
Documents
  ↓
Preprocessing (Word count filtering, cleanup)
  ↓
Transformer Embeddings (all-MiniLM-L6-v2)
  ↓
FAISS Dense Index (IndexFlatIP with normalized vectors)
  ↓
Query Embedding
  ↓
Dense Retrieval ──────┐
                      ├→ Hybrid Score Fusion (Min-Max norm) → Ranked Results
BM25 Retrieval ───────┘
```

## Technical Stack

- **Core Search**: FAISS (for exact Inner Product dense retrieval), `rank-bm25` (Okapi BM25 implementation)
- **Deep Learning Embeddings**: `sentence-transformers` (under the hood utilizes `transformers` and `torch`)
- **Data Handling**: `pandas`, `numpy`, `datasets`
- **Application**: `streamlit`
- **Testing**: `pytest`

## Retrieval Methods

1. **BM25 Retrieval**: Traditional keyword/lexical search based on Okapi BM25. Excellent for exact term matching.
2. **Dense Semantic Retrieval**: Uses `all-MiniLM-L6-v2` (384-dimensional) embeddings to perform nearest neighbor search in a FAISS index (`IndexFlatIP`). Catches contextual meaning and synonyms.
3. **Hybrid Retrieval**: Retrieves top candidates independently from both BM25 and Semantic methods. The raw scores are min-max normalized to `[0, 1]`, and a weighted average (`alpha * bm25 + (1 - alpha) * semantic`) determines the final ranking.

## Evaluation & Benchmarking

The system is evaluated on a sample of MS MARCO queries. The evaluation compares Precision@10, Recall@10, and MRR@10 across all three methods.
Latency benchmarks measure end-to-end query time (including embedding generation and mapping) averaged across the query set.

### Known Limitations
- The current default corpus contains approximately **10,000 passages**.
- The FAISS index uses exact search (`IndexFlatIP`) rather than Approximate Nearest Neighbors (ANN) like HNSW, which is optimal for small corpora but may not scale to billions of vectors without changing the index type.
- The evaluation dataset consists of a small query subset, giving directional performance metrics rather than statistically significant generalized metrics.

## Reproducing Results

Configuration is centralized in `config.py`.

### 1. Installation
```bash
python -m pip install -r requirements.txt
```

### 2. Data Preparation & Indexing
```bash
python dataset_download.py    # Download passages
python preprocess.py          # Clean and enrich
python build_index.py         # Batch encode and build FAISS index
```

### 3. Evaluation & Benchmarking
```bash
python evaluation.py          # Compute P@10, R@10, MRR@10
python benchmark.py           # Measure end-to-end latency
```

### 4. Run Tests
```bash
python -m pytest tests/       # Run unit and integration tests
```

### 5. Application
```bash
streamlit run app.py          # Launch interactive UI
```
