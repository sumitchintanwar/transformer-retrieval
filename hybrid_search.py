from logger import get_logger

logger = get_logger(__name__)
import os
import pickle
from typing import Any, Dict, List, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
from bm25_search import BM25Retriever


class SemanticRetriever:
    """Dense vector retriever using Sentence Transformers + FAISS.

    Encodes queries into embeddings and searches a pre-built FAISS
    cosine-similarity index over the passage corpus.

    Args:
        model_name (str): Sentence transformer model name.
        index_path (str): Path to the serialized FAISS index.
        data_path (str): Path to the preprocessed passages CSV.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        index_path: str = config.FAISS_INDEX_FILE,
        data_path: str = config.PROCESSED_DATA_FILE,
    ) -> None:
        df = pd.read_csv(data_path)
        self.passage_ids = df["passage_id"].values
        self.passage_texts = df["passage_text"].values

        self.model = SentenceTransformer(model_name, device=config.DEVICE)

        with open(index_path, "rb") as f:
            self.index = faiss.deserialize_index(pickle.load(f))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using cosine similarity over dense embeddings.

        Args:
            query (str): The search query string.
            top_k (int): Number of results to return.

        Returns:
            list[dict]: Each dict contains passage_id (int), score (float),
                        and passage_text (str), sorted by descending score.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        if len(query) > 1000:
            raise ValueError("Query is too long (exceeds 1000 characters)")

        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Cap top_k at corpus size to prevent requesting more neighbors than exist
        actual_top_k = min(top_k, len(self.passage_ids))
        if actual_top_k == 0:
            return []
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(vec, k=actual_top_k)

        results = []
        for pid, score in zip(I.flatten().tolist(), D.flatten().tolist()):
            if pid < 0:
                continue
            idx = np.searchsorted(self.passage_ids, pid)
            if idx < len(self.passage_ids) and self.passage_ids[idx] == pid:
                results.append(
                    {
                        "passage_id": int(pid),
                        "score": float(score),
                        "passage_text": self.passage_texts[idx],
                    }
                )
        return results


def min_max_normalize(scores: List[float]) -> List[float]:
    """Apply min-max normalization to a list of scores.

    Maps all scores to [0, 1] range where:
    - The highest score becomes 1.0
    - The lowest score becomes 0.0
    - All others are linearly scaled between them

    This is necessary because BM25 scores and cosine similarity scores
    live on different scales. BM25 scores can range from 0 to 30+,
    while cosine similarity is already in [0, 1]. Without normalization,
    the method with larger absolute scores would dominate the fusion.

    Args:
        scores (list[float]): Raw scores to normalize.

    Returns:
        list[float]: Normalized scores in [0, 1].
    """
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    min_val = arr.min()
    max_val = arr.max()
    if max_val == min_val:
        return [1.0] * len(scores)
    return ((arr - min_val) / (max_val - min_val)).tolist()


class HybridRetriever:
    """Combines BM25 (sparse) and semantic (dense) retrieval via score fusion.

    Pipeline:
    1. Retrieve top_k candidates from BM25 (lexical matching)
    2. Retrieve top_k candidates from semantic search (meaning matching)
    3. Merge candidates by passage_id (union of both result sets)
    4. Normalize BM25 and semantic scores independently via min-max
    5. Compute final_score = alpha * bm25_norm + (1 - alpha) * semantic_norm

    The alpha parameter controls the blend:
    - alpha=1.0 → pure BM25 (keyword matching only)
    - alpha=0.0 → pure semantic (meaning matching only)
    - alpha=0.5 → equal blend (recommended starting point)

    Args:
        bm25_retriever (BM25Retriever): The BM25 retriever instance.
        semantic_retriever (SemanticRetriever): The semantic retriever instance.
        alpha (float): Weight for BM25 score in [0, 1]. Default 0.5.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        alpha: float = 0.5,
    ) -> None:
        self.bm25 = bm25_retriever
        self.semantic = semantic_retriever
        self.alpha = alpha

    def search(
        self, query: str, top_k: int = 10, retrieve_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 and semantic retrieval.

        Args:
            query (str): The search query string.
            top_k (int): Final number of results to return.
            retrieve_k (int): Number of candidates to fetch from each
                              retriever before fusion. Should be >= top_k.

        Returns:
            list[dict]: Each dict contains passage_id, final_score,
                        bm25_score, semantic_score, and passage_text.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        if len(query) > 1000:
            raise ValueError("Query is too long (exceeds 1000 characters)")

        if not isinstance(top_k, int) or not isinstance(retrieve_k, int):
            raise TypeError("top_k and retrieve_k must be integers")
        if top_k <= 0 or retrieve_k <= 0:
            raise ValueError("top_k and retrieve_k must be greater than 0")

        # Step 1: Retrieve candidates from both retrievers independently.
        # Each returns up to retrieve_k results with raw scores.
        bm25_results = self.bm25.search(query, top_k=retrieve_k)
        semantic_results = self.semantic.search(query, top_k=retrieve_k)

        # Step 2: Merge candidates by passage_id into a unified dict.
        # For each passage, we store both raw scores (None if not retrieved
        # by that retriever) and the passage text.
        merged = {}
        for r in bm25_results:
            pid = r["passage_id"]
            merged[pid] = {
                "passage_id": pid,
                "bm25_raw": r["score"],
                "semantic_raw": None,
                "passage_text": r["passage_text"],
            }

        for r in semantic_results:
            pid = r["passage_id"]
            if pid in merged:
                # Passage found in both retrievers — fill in semantic score.
                merged[pid]["semantic_raw"] = r["score"]
            else:
                # Passage only found by semantic retriever.
                merged[pid] = {
                    "passage_id": pid,
                    "bm25_raw": None,
                    "semantic_raw": r["score"],
                    "passage_text": r["passage_text"],
                }

        # Step 3: Normalize scores independently.
        # We collect all non-None scores from each retriever, normalize
        # them to [0, 1], then re-assign them to the merged results.
        bm25_raws = [
            v["bm25_raw"] for v in merged.values() if v["bm25_raw"] is not None
        ]
        semantic_raws = [
            v["semantic_raw"] for v in merged.values() if v["semantic_raw"] is not None
        ]

        bm25_normed = min_max_normalize(bm25_raws)
        semantic_normed = min_max_normalize(semantic_raws)

        # Map normalized scores back to each passage.
        # Passages missing from a retriever get score 0 for that modality.
        bm25_idx = 0
        semantic_idx = 0
        for v in merged.values():
            if v["bm25_raw"] is not None:
                v["bm25_score"] = bm25_normed[bm25_idx]
                bm25_idx += 1
            else:
                v["bm25_score"] = 0.0

            if v["semantic_raw"] is not None:
                v["semantic_score"] = semantic_normed[semantic_idx]
                semantic_idx += 1
            else:
                v["semantic_score"] = 0.0

        # Step 4: Compute the final blended score.
        # final_score = alpha * bm25_normalized + (1 - alpha) * semantic_normalized
        for v in merged.values():
            v["final_score"] = (
                self.alpha * v["bm25_score"] + (1 - self.alpha) * v["semantic_score"]
            )

        # Step 5: Sort by final_score descending and return top_k.
        ranked = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)

        return [
            {
                "passage_id": r["passage_id"],
                "final_score": round(r["final_score"], 4),
                "bm25_score": round(r["bm25_score"], 4),
                "semantic_score": round(r["semantic_score"], 4),
                "passage_text": r["passage_text"],
            }
            for r in ranked[:top_k]
        ]


if __name__ == "__main__":
    logger.info("Loading retrievers...")
    bm25 = BM25Retriever()
    semantic = SemanticRetriever()
    hybrid = HybridRetriever(bm25, semantic, alpha=config.HYBRID_ALPHA)

    queries = [
        "how does photosynthesis work",
        "machine learning algorithms",
        "reserve bank of australia",
    ]

    for query in queries:
        logger.info(f"\n{'='*70}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*70}")
        results = hybrid.search(query, top_k=5)
        for rank, r in enumerate(results, 1):
            logger.info(
                f"  #{rank} final={r['final_score']:.4f} "
                f"(bm25={r['bm25_score']:.4f}, sem={r['semantic_score']:.4f}) "
                f"id={r['passage_id']}"
            )
            logger.info(f"     {r['passage_text'][:120]}...")
