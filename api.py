from logger import get_logger

logger = get_logger(__name__)
from typing import Any, Dict, List, Optional

import pandas as pd

import config
from bm25_search import BM25Retriever
from hybrid_search import HybridRetriever, SemanticRetriever


class SearchEngineAPI:
    """A clean Python API boundary for the Semantic Search Engine."""

    def __init__(self, data_path: str = config.PROCESSED_DATA_FILE):
        self.data_path = data_path
        self._passage_lookup = None
        self._bm25 = None
        self._semantic = None
        self._hybrid = None

    def _load_lookup(self):
        if self._passage_lookup is None:
            df = pd.read_csv(self.data_path)
            self._passage_lookup = df.set_index("passage_id")
        return self._passage_lookup

    def _get_bm25(self) -> BM25Retriever:
        if self._bm25 is None:
            self._bm25 = BM25Retriever(data_path=str(self.data_path))
        return self._bm25

    def _get_semantic(self) -> SemanticRetriever:
        if self._semantic is None:
            self._semantic = SemanticRetriever(
                model_name=config.MODEL_NAME,
                index_path=str(config.FAISS_INDEX_FILE),
                data_path=str(self.data_path),
            )
        return self._semantic

    def _get_hybrid(self) -> HybridRetriever:
        if self._hybrid is None:
            self._hybrid = HybridRetriever(
                self._get_bm25(), self._get_semantic(), alpha=config.HYBRID_ALPHA
            )
        return self._hybrid

    def get_passage(self, passage_id: int) -> Optional[Dict[str, Any]]:
        """Lookup a passage by its ID."""
        lookup = self._load_lookup()
        if passage_id in lookup.index:
            row = lookup.loc[passage_id]
            return {
                "passage_id": passage_id,
                "passage_text": row["passage_text"],
                "word_count": int(row["word_count"]),
            }
        return None

    def search_bm25(
        self, query: str, top_k: int = config.DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform a BM25 keyword search."""
        retriever = self._get_bm25()
        results = retriever.search(query, top_k=top_k)
        return self._enrich_results(results)

    def search_semantic(
        self, query: str, top_k: int = config.DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform a dense semantic vector search."""
        retriever = self._get_semantic()
        results = retriever.search(query, top_k=top_k)
        return self._enrich_results(results)

    def search_hybrid(
        self, query: str, top_k: int = config.DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """Perform a hybrid (BM25 + Semantic) search."""
        retriever = self._get_hybrid()
        results = retriever.search(query, top_k=top_k)
        return self._enrich_results(results)

    def _enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach full passage metadata (like word_count) to the raw results."""
        enriched = []
        for r in results:
            pid = r["passage_id"]
            p_data = self.get_passage(pid)
            if p_data:
                # Merge dictionaries, preserving scores from search
                merged = {**p_data, **r}
                enriched.append(merged)
        return enriched
