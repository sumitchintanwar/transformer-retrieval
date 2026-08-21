import pickle
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd
import pytest

from bm25_search import BM25Retriever
from hybrid_search import HybridRetriever, SemanticRetriever


class MockSentenceTransformer:
    def __init__(self, model_name="mock", device="cpu"):
        self.device = device

    def encode(self, sentences, normalize_embeddings=True, **kwargs):
        # Return a random normalized vector for each sentence
        vecs = np.random.rand(len(sentences), 8).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / norms
        return vecs


@pytest.fixture
def mock_data(tmp_path):
    # Create mock CSV
    data_path = tmp_path / "msmarco_passages.csv"
    df = pd.DataFrame(
        {"passage_id": [10, 20, 30], "passage_text": ["doc A", "doc B", "doc C"]}
    )
    df.to_csv(data_path, index=False)

    # Create mock FAISS index
    index_path = tmp_path / "faiss_index.pickle"
    dim = 8
    idx = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIDMap(idx)
    # Add vectors: perfectly match the mock data
    vecs = np.array(
        [
            [1.0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1.0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.0, 0, 0, 0, 0, 0],
        ],
        dtype="float32",
    )
    idx.add_with_ids(vecs, np.array([10, 20, 30], dtype="int64"))

    with open(index_path, "wb") as f:
        pickle.dump(faiss.serialize_index(idx), f)

    return str(data_path), str(index_path)


@patch("hybrid_search.SentenceTransformer", side_effect=MockSentenceTransformer)
def test_semantic_retriever(mock_st, mock_data):
    data_path, index_path = mock_data

    retriever = SemanticRetriever(
        model_name="mock", index_path=index_path, data_path=data_path
    )

    # Force the mock model to return a specific query vector (matches doc B mostly)
    retriever.model.encode = lambda q, normalize_embeddings=True: np.array(
        [[0, 1.0, 0, 0, 0, 0, 0, 0]], dtype="float32"
    )

    results = retriever.search("query", top_k=2)

    assert len(results) == 2
    assert results[0]["passage_id"] == 20
    assert results[0]["passage_text"] == "doc B"
    assert "score" in results[0]


class MockBM25Retriever:
    def search(self, query, top_k=10):
        # Hardcode some results
        return [
            {"passage_id": 10, "score": 25.0, "passage_text": "doc A"},
            {"passage_id": 20, "score": 10.0, "passage_text": "doc B"},
        ][:top_k]


class MockSemRetriever:
    def search(self, query, top_k=10):
        # Hardcode some results
        return [
            {"passage_id": 20, "score": 0.9, "passage_text": "doc B"},
            {"passage_id": 30, "score": 0.5, "passage_text": "doc C"},
        ][:top_k]


def test_hybrid_retriever():
    bm25 = MockBM25Retriever()
    sem = MockSemRetriever()

    hybrid = HybridRetriever(bm25, sem, alpha=0.5)
    results = hybrid.search("test", top_k=5, retrieve_k=2)

    # BM25 candidates: 10, 20
    # Sem candidates: 20, 30
    # Merged candidates: 10, 20, 30
    # BM25 raw: 10->25.0, 20->10.0 => norm: 10->1.0, 20->0.0
    # Sem raw: 20->0.9, 30->0.5 => norm: 20->1.0, 30->0.0
    # Passages missing get 0.0 for that modality

    # 10: bm25_norm=1.0, sem_norm=0.0 => final=0.5
    # 20: bm25_norm=0.0, sem_norm=1.0 => final=0.5
    # 30: bm25_norm=0.0, sem_norm=0.0 => final=0.0

    assert len(results) == 3
    ids = [r["passage_id"] for r in results]
    assert set(ids) == {10, 20, 30}

    # Verify no NaN or divide by zero
    for r in results:
        assert not np.isnan(r["final_score"])
        assert not np.isnan(r["bm25_score"])
        assert not np.isnan(r["semantic_score"])


def test_invalid_query_inputs():
    bm25 = MockBM25Retriever()
    sem = MockSemRetriever()
    hybrid = HybridRetriever(bm25, sem, alpha=0.5)

    with pytest.raises(TypeError):
        hybrid.search(123)

    with pytest.raises(ValueError):
        hybrid.search("")

    with pytest.raises(ValueError):
        hybrid.search("   ")

    with pytest.raises(ValueError):
        hybrid.search("a" * 1001)


def test_invalid_top_k():
    bm25 = MockBM25Retriever()
    sem = MockSemRetriever()
    hybrid = HybridRetriever(bm25, sem, alpha=0.5)

    with pytest.raises(TypeError):
        hybrid.search("query", top_k="10")

    with pytest.raises(ValueError):
        hybrid.search("query", top_k=-5)

    with pytest.raises(ValueError):
        hybrid.search("query", top_k=0)
