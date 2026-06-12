# Project Architecture

## Current Search Pipeline

### Semantic Search (Dense Retrieval)
- **Offline**: CSV → SentenceTransformer encode (L2-normalized) → FAISS IndexFlatIP + IndexIDMap → serialize to pickle
- **Online**: User query → SentenceTransformer encode → FAISS cosine search → display rank, score, passage text

### BM25 Search (Sparse Retrieval)
- **Offline**: CSV → tokenize → BM25Okapi index (in-memory, built at startup)
- **Online**: User query → tokenize → BM25Okapi.get_scores → display rank, score, passage text

## Data Flow

```
data/msmarco_passages.csv (10K passages, 3 columns)
  ├── passage_id (int): unique identifier, used as FAISS index key
  ├── passage_text (str): web passage text from MS MARCO
  └── word_count (int): computed during preprocessing
```

## Embedding Generation Flow

- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Offline** (`build_index.py`): `model.encode(passages, normalize_embeddings=True)` → float32 array → FAISS IndexFlatIP
- **Runtime** (`vector_engine/utils.py`): `model.encode([query], normalize_embeddings=True)` → query vector → FAISS search

## FAISS Indexing Flow (`build_index.py`)

1. Load preprocessed passages CSV
2. Encode all passages with L2 normalization using all-MiniLM-L6-v2
3. `faiss.IndexFlatIP(384)` — inner product (cosine similarity with normalized vectors)
4. `faiss.IndexIDMap(index)` — map vectors to passage IDs
5. `index.add_with_ids(embeddings, passage_ids)` → `pickle.dump` → `models/faiss_index.pickle`

## BM25 Indexing Flow (`bm25_search.py`)

1. Load preprocessed passages CSV
2. Tokenize each passage (lowercase + whitespace split)
3. `BM25Okapi(tokenized_corpus)` — builds inverted index with TF-IDF-like scoring
4. At query time: tokenize query → `get_scores()` → sort by descending score

## Search Flow

### Semantic (app.py + vector_engine/utils.py)
1. Load CSV, SentenceTransformer model, FAISS index (cached via `@st.cache`)
2. User enters query text
3. `vector_search()` encodes query → FAISS `index.search(k=num_results)` → returns scores `D`, IDs `I`
4. Results displayed with rank, similarity score, word count, passage text

### BM25 (bm25_search.py)
1. `BM25Retriever` loads CSV and builds BM25Okapi index at init
2. `search(query, top_k)` tokenizes query → `get_scores()` → sort → return top_k results
3. Each result contains `passage_id`, `score`, `passage_text`

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `dataset_download.py` | Stream 10K passages from MS MARCO v1.1 via HuggingFace |
| `preprocess.py` | Deduplicate, filter short passages, add word_count |
| `build_index.py` | Encode passages, build FAISS cosine-similarity index |
| `bm25_search.py` | BM25 keyword retriever (standalone, importable class) |
