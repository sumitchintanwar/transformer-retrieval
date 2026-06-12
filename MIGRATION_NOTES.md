# Migration Notes: MS MARCO Dataset + Model Upgrade

## Overview

Migrated from a toy dataset of 8,430 misinformation academic papers to 10,000 passages from the MS MARCO Passage Ranking dataset. Upgraded the embedding model and switched to cosine similarity search.

---

## Schema Changes

### Data File

| Aspect | Before | After |
|--------|--------|-------|
| File | `data/misinformation_papers.csv` | `data/msmarco_passages.csv` |
| Rows | 8,430 | 10,000 |
| Columns | `original_title`, `abstract`, `year`, `citations`, `id`, `is_EN` | `passage_id`, `passage_text`, `word_count` |
| Content | Academic paper metadata | Web passage text |

### New Columns

- `passage_id` — Integer ID, used as FAISS index key
- `passage_text` — Raw passage text from MS MARCO
- `word_count` — Computed during preprocessing

---

## Model Changes

| Aspect | Before | After |
|--------|--------|-------|
| Model | `distilbert-base-nli-stsb-mean-tokens` | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dim | 768 | 384 |
| Quality | Decent on STS tasks | Strong on semantic retrieval (MS MARCO trained) |

---

## FAISS Index Changes

| Aspect | Before | After |
|--------|--------|-------|
| Index type | `IndexFlatL2` | `IndexFlatIP` (inner product) |
| Similarity | L2 distance (lower = better) | Cosine similarity (higher = better) |
| Normalization | None | L2-normalized embeddings |
| Wrapper | `IndexIDMap` | `IndexIDMap` (unchanged) |

With normalized vectors, inner product equals cosine similarity, giving scores in [0, 1].

---

## App Changes (`app.py`)

- Model name updated to `all-MiniLM-L6-v2`
- Data path updated to `msmarco_passages.csv`
- Removed sidebar filters: year, citations, language
- Kept: number of results slider
- Display now shows: rank number, similarity score, word count, passage text
- Uses `passage_id` index for O(1) lookup instead of filtering

---

## Pipeline Scripts

Three new scripts replace the manual notebook workflow:

1. **`dataset_download.py`** — Streams 10K unique passages from MS MARCO v1.1 via HuggingFace `datasets`, saves to `data/msmarco_passages_raw.csv`
2. **`preprocess.py`** — Deduplicates, removes empty/short passages (<5 words), computes `word_count`, saves to `data/msmarco_passages.csv`
3. **`build_index.py`** — Loads `all-MiniLM-L6-v2`, encodes all passages with L2 normalization, builds `IndexFlatIP` + `IndexIDMap`, serializes to `models/faiss_index.pickle`

### Running the pipeline

```bash
python dataset_download.py
python preprocess.py
python build_index.py
```

---

## Removed

- `data/misinformation_papers.csv` (replaced)
- `folium` dependency (was unused)
- `id2details()` function from `vector_engine/utils.py`
- Year/citations/language filters from the sidebar
- Pinned version numbers in `requirements.txt` (now unpinned for compatibility)

---

## Files Modified

| File | Action |
|------|--------|
| `app.py` | Rewritten for new schema |
| `vector_engine/utils.py` | Updated for normalized embeddings, removed `id2details` |
| `requirements.txt` | Added `datasets`, removed `folium`, unpinned versions |
| `dataset_download.py` | New |
| `preprocess.py` | New |
| `build_index.py` | New |
