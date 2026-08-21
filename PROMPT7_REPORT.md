# Prompt 7 — Implement true full-corpus batch indexing

## Implementation
Modified `build_index.py` to stream the processed CSV using `pd.read_csv(chunksize=...)`. The FAISS index is initialized first, and each chunk of size 100,000 is encoded and incrementally added using `index.add_with_ids(...)`.

## Memory Footprint Analysis
- **Previous implementation**: `df = pd.read_csv(...)` loads the entire CSV (millions of rows ideally) into memory. Then `df["passage_text"].tolist()` creates a massive Python list of strings. Finally, `model.encode(...)` processes everything, requiring massive host RAM and GPU VRAM if not batched.
- **New implementation**: By processing in chunks of 100,000 rows, the memory usage is bounded by:
  - FAISS index size (grows linearly, but highly optimized in C++)
  - One chunk of pandas DataFrame + Python list (small and constant)
  - PyTorch inference overhead for the current batch (constant)
  
This ensures the script memory consumed is mostly constrained by the FAISS index itself and allows scaling to datasets that are larger than the host RAM.
