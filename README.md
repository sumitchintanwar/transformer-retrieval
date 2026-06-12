# Semantic Search Engine

Semantic search over text passages using Sentence Transformers and FAISS, with an optional BM25 keyword retrieval mode.

## Setup

```bash
pip install -r requirements.txt
```

## Indexing Pipeline

```bash
python dataset_download.py    # Download 10K MS MARCO passages
python preprocess.py          # Clean and enrich passages
python build_index.py         # Build FAISS cosine-similarity index
```

## Run the App

```bash
streamlit run app.py
```

## Docker

```bash
docker build -t semantic-search .
docker run -p 8501:8501 semantic-search
```

Then open http://localhost:8501/

## BM25 Search

```python
from bm25_search import BM25Retriever

retriever = BM25Retriever()
results = retriever.search("how does photosynthesis work", top_k=5)
```
