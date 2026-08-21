# Prompt 2 — Fix retrieval correctness and edge cases

## Issue Report

| Issue | Severity | Root Cause | Fix | Regression Test Added |
|---|---|---|---|---|
| Query is None or invalid type | HIGH | Unhandled input type passed to tokenizer/embedder | Added type checking in `search()` for string type | Yes |
| Query is empty or whitespace | MEDIUM | Creates empty embedding vectors or invalid BM25 lookup | Added `query.strip()` check in `search()` | Yes |
| Query is extremely long | MEDIUM | Exceeds transformer context window or slows down BM25 | Added length limit check (>1000 chars) | Yes |
| `top_k` is zero or negative | LOW | Array slicing with negative indices alters intent | Added explicit check for `top_k <= 0` | Yes |
| `top_k` > corpus size | MEDIUM | FAISS returns placeholder `-1` IDs | Capped `top_k` at `len(corpus)` | Yes |
| Invalid FAISS placeholder IDs (-1) | HIGH | Requesting more neighbors than exist in index | Added `if pid < 0: continue` check | Yes |
| Min-Max NaN / Div-by-zero | HIGH | Min and max scores are identical | Checked `min_val == max_val` in `min_max_normalize` | Yes |

## Remaining Edge Cases Intentionally Unchanged

- **Extremely large `top_k` performance degradation**: We cap `top_k` at the corpus size, but even returning 10,000 results will be slow. We left this unchanged because the system is designed as a search engine where `top_k` is practically small. Strict capping to a maximum limit (e.g., 100) could be added, but we preserve the user's requested `top_k` if it is within the corpus bounds.
- **Empty FAISS index**: The code assumes the FAISS index and CSV data are populated correctly before starting the app. Validating the data at query time would introduce unnecessary overhead.
