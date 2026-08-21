# Baseline State

## Repository Commit/Working-Tree State
The working tree contains a basic semantic search engine implementation. No automated tests exist.

## Environment and Dependency Information
Dependencies in `requirements.txt`:
`torch`, `transformers`, `sentence-transformers`, `pandas`, `faiss-cpu`, `numpy`, `streamlit`, `datasets`, `rank-bm25`.

## Exact Commands Used
- Download Data: `python dataset_download.py`
- Preprocess Data: `python preprocess.py`
- Build FAISS Index: `python build_index.py`
- Run BM25 Retrieval: `python bm25_search.py`
- Run Hybrid/Semantic Search test: `python test_hybrid.py`
- Run Evaluation: `python evaluation.py`
- Run Benchmarks: `python benchmark.py`
- Launch Streamlit: `streamlit run app.py`

## State
- **Corpus Size**: 10,000 passages (MS MARCO).
- **Model Name and Embedding Dimension**: `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions.
- **FAISS Index Type**: `faiss.IndexFlatIP` combined with `faiss.IndexIDMap`.

## Baseline Retrieval Metrics
From `evaluation_report.md` (Top-K=10):
- **BM25**: P@10 = 0.1360, R@10 = 0.6090, MRR@10 = 0.4411
- **Semantic**: P@10 = 0.1840, R@10 = 0.7933, MRR@10 = 0.5247
- **Hybrid**: P@10 = 0.1720, R@10 = 0.7360, MRR@10 = 0.5539

## Baseline Latency Measurements
From `benchmark_results.md` (average latency):
- **BM25**: 299.1 ms
- **Semantic**: 138.9 ms
- **Hybrid**: 298.4 ms

## Known Failures, Limitations, and Issues
- **Hardcoded Paths**: Most file paths and model names are hardcoded.
- **Tests**: Zero automated unit tests or integration tests.
- **Reproducibility**: No random seeds set explicitly for the evaluation set generation, though `benchmark.py` uses seed 42.
- **Error Handling**: Very little error checking (e.g. for missing files).
- **Scale**: The "large-scale" claim is unsupported as it only uses 10,000 passages with an exhaustive exact search index.

## Summary
The pipeline is currently reproducible via sequentially running scripts. The metrics and latency match the reported artifacts. However, it completely lacks test coverage, configuration management, and robust error handling.
