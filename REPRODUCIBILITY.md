# Reproducibility Guide

All hardcoded values and paths have been centralized into a single configuration module (`config.py`).
This allows you to change the behavior of the entire pipeline simply by modifying `config.py` or by passing environment variables.

## Configuration Variables
The following environment variables (with their defaults) are available:
- `MODEL_NAME`: The sentence transformer model to use (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `DEVICE`: Compute device (default: `cpu`, can be set to `cuda`)
- `BATCH_SIZE`: Batch size for FAISS index creation (default: `256`)
- `NUM_PASSAGES`: Target number of passages to download from MS MARCO (default: `10000`)
- `TOP_K`: Number of retrieval results (default: `10`)
- `HYBRID_ALPHA`: BM25 weight in hybrid search (default: `0.5`)
- `BENCHMARK_QUERIES`: Number of queries to run during benchmarking (default: `50`)

## Pipeline Execution
To reproduce the pipeline, execute the scripts sequentially:
1. `python dataset_download.py`
2. `python preprocess.py`
3. `python build_index.py`
4. `python benchmark.py`
5. `python evaluation.py`
6. `streamlit run app.py`

This ensures that the exact same files and constants are used across all stages of the semantic search engine.
