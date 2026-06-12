# Project Progress — Semantic Search Engine

## Current State

A fully functional semantic search engine with three retrieval modes (Semantic, BM25, Hybrid) over 10K MS MARCO passages, with a Streamlit UI, benchmarking, and evaluation dataset.

---

## What Was Built

### 1. Dataset Pipeline (replaced original toy dataset)

| Script | Purpose | Status |
|--------|---------|--------|
| `dataset_download.py` | Streams 10K passages from MS MARCO v1.1 via HuggingFace `datasets` | Done |
| `preprocess.py` | Deduplicates, filters short passages (<5 words), adds `word_count` | Done |
| `build_index.py` | Encodes with `all-MiniLM-L6-v2`, builds FAISS `IndexFlatIP` + `IndexIDMap` | Done |

**Run order**: `dataset_download.py` → `preprocess.py` → `build_index.py`

**Data files**:
- `data/msmarco_passages_raw.csv` — raw downloaded passages
- `data/msmarco_passages.csv` — preprocessed (passage_id, passage_text, word_count)
- `models/faiss_index.pickle` — serialized FAISS index

### 2. Retrieval Engines

**Semantic Search** (`vector_engine/utils.py`):
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- FAISS index: `IndexFlatIP` with L2-normalized embeddings (cosine similarity)
- `vector_search(query, model, index, num_results)` → D (scores), I (passage_ids)

**BM25 Search** (`bm25_search.py`):
- `BM25Retriever` class with `search(query, top_k)` → list of dicts
- Uses `rank_bm25.BM25Okapi` with lowercase whitespace tokenization
- Returns passage_id, score, passage_text

**Hybrid Search** (`hybrid_search.py`):
- `SemanticRetriever` class — wraps SentenceTransformer + FAISS
- `HybridRetriever` class — combines BM25 + Semantic via score fusion
- Pipeline: retrieve top 20 from each → merge by passage_id → min-max normalize → `0.5 * bm25 + 0.5 * semantic`
- `search(query, top_k)` → list of dicts with passage_id, final_score, bm25_score, semantic_score, passage_text

### 3. Streamlit App (`app.py`)

- Sidebar: Search Mode selector (Semantic / BM25 / Hybrid) + result count slider
- Semantic mode: shows Cosine Similarity score
- BM25 mode: shows BM25 score
- Hybrid mode: shows Combined, BM25, and Semantic scores
- All retrievers cached via `@st.cache`

### 4. Benchmarking (`benchmark.py`)

- Measures average query latency across 50 random queries
- Results saved to `benchmark_results.md`

**Results (10K passages, CPU)**:
| Method | Avg Latency (ms) |
|--------|------------------:|
| Semantic | 138.9 |
| Hybrid | 298.4 |
| BM25 | 299.1 |

### 5. Evaluation Dataset (`evaluation_dataset.json`)

- 50 queries with relevant passage IDs
- 10 queries per category: Science, Finance, Health, Technology, Geography
- Sourced from actual MS MARCO subset passages

### 6. Documentation

| File | Content |
|------|---------|
| `PROJECT_ARCHITECTURE.md` | Pipeline diagrams, data flow, indexing flow, search flow |
| `MIGRATION_NOTES.md` | Schema changes, model changes, FAISS changes, app changes |
| `README.md` | Setup, indexing pipeline, run instructions, Docker, BM25 usage |
| `benchmark_results.md` | Latency benchmarks |
| `PROGRESS.md` | This file |

---

## File Tree

```
semantic-search-engine/
├── app.py                      # Streamlit UI (3 search modes)
├── benchmark.py                # Latency benchmarking script
├── bm25_search.py              # BM25Retriever class
├── build_index.py              # FAISS index builder
├── dataset_download.py         # MS MARCO downloader
├── evaluation_dataset.json     # 50 queries with ground truth
├── hybrid_search.py            # SemanticRetriever + HybridRetriever
├── preprocess.py               # Data preprocessing
├── test_hybrid.py              # Hybrid search comparison test
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── Dockerfile                  # Docker build
├── Dockerrun.aws.json          # AWS EB config
├── PROJECT_ARCHITECTURE.md     # Architecture docs
├── MIGRATION_NOTES.md          # Migration changelog
├── README.md                   # Project readme
├── benchmark_results.md        # Benchmark output
├── PROGRESS.md                 # This file
├── data/
│   ├── msmarco_passages_raw.csv
│   └── msmarco_passages.csv
├── models/
│   └── faiss_index.pickle
├── vector_engine/
│   ├── __init__.py
│   └── utils.py                # vector_search(), id2details() removed
└── notebooks/
    └── 001_vector_search.ipynb # Original notebook (updated model refs)
```

---

## Key Decisions Made

1. **Model**: `all-MiniLM-L6-v2` (384-dim) — faster and better for retrieval than original `distilbert-base-nli-stsb-mean-tokens`
2. **Index**: `IndexFlatIP` with normalized embeddings — cosine similarity, scores in [0, 1]
3. **Corpus**: 10K MS MARCO passages — lightweight, realistic web text
4. **Fusion**: Equal weight (alpha=0.5) for hybrid — min-max normalized scores
5. **Tokenization**: Lowercase whitespace split for BM25

---

## What's NOT Done Yet

- [ ] Evaluate retrieval quality using `evaluation_dataset.json` (precision, recall, MRR)
- [ ] Tune hybrid alpha parameter
- [ ] Add cross-encoder re-ranking stage
- [ ] Persistent BM25 index (currently built in-memory at startup)
- [ ] Dockerfile updated for new dependencies
- [ ] CI/CD or automated testing
- [ ] Deploy to production

---

## How to Continue

### To run the app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### To rebuild index from scratch:
```bash
python dataset_download.py
python preprocess.py
python build_index.py
```

### To run benchmarks:
```bash
python benchmark.py
```

### To test hybrid search:
```bash
python test_hybrid.py
```

### Next logical steps:
1. Use `evaluation_dataset.json` to compute precision@k, recall@k, MRR for all three modes
2. Experiment with different alpha values in HybridRetriever
3. Add cross-encoder re-ranking (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
4. Update `Dockerfile` for new dependencies
