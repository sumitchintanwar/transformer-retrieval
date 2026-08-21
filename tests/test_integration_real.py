import pickle

import faiss
import numpy as np
import pandas as pd
import pytest
from sentence_transformers import SentenceTransformer

import config
from bm25_search import BM25Retriever
from hybrid_search import HybridRetriever, SemanticRetriever


@pytest.fixture(scope="module")
def setup_tiny_corpus(tmp_path_factory):
    """Sets up a real but tiny deterministic corpus for integration testing."""
    tmp_dir = tmp_path_factory.mktemp("real_integration")

    data_path = tmp_dir / "tiny_passages.csv"
    index_path = tmp_dir / "tiny_faiss.pickle"

    # 1. Create Data
    docs = [
        {
            "passage_id": 100,
            "passage_text": "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water.",
        },
        {
            "passage_id": 200,
            "passage_text": "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn.",
        },
        {
            "passage_id": 300,
            "passage_text": "The Reserve Bank of Australia is Australia's central bank and banknote issuing authority.",
        },
        {
            "passage_id": 400,
            "passage_text": "Plants need water and sunlight to grow. The process of photosynthesis depends heavily on these elements.",
        },
    ]
    df = pd.DataFrame(docs)
    df.to_csv(data_path, index=False)

    # 2. Create Real Index
    model = SentenceTransformer(config.MODEL_NAME, device="cpu")
    dim = model.get_sentence_embedding_dimension()

    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    passages = df["passage_text"].tolist()
    passage_ids = df["passage_id"].values

    embeddings = model.encode(passages, normalize_embeddings=True)
    index.add_with_ids(embeddings.astype("float32"), passage_ids.astype("int64"))

    with open(index_path, "wb") as f:
        pickle.dump(faiss.serialize_index(index), f)

    return str(data_path), str(index_path)


def test_real_end_to_end_pipeline(setup_tiny_corpus):
    data_path, index_path = setup_tiny_corpus

    # 1. Load Real BM25
    bm25 = BM25Retriever(data_path=data_path)

    # 2. Load Real Semantic
    semantic = SemanticRetriever(
        model_name=config.MODEL_NAME, index_path=index_path, data_path=data_path
    )

    # 3. Load Real Hybrid
    hybrid = HybridRetriever(bm25, semantic, alpha=0.5)

    # --- Test Semantic ---
    # Query related to AI
    sem_res = semantic.search("artificial intelligence and computers", top_k=2)
    assert len(sem_res) == 2
    assert sem_res[0]["passage_id"] == 200  # Should be the machine learning doc

    # --- Test BM25 ---
    # Query for exact keywords
    bm25_res = bm25.search("Reserve Bank Australia", top_k=2)
    assert len(bm25_res) > 0
    assert bm25_res[0]["passage_id"] == 300  # Should be the bank doc

    # --- Test Hybrid ---
    # Query for "plants sunlight photosynthesis"
    # Docs 100 and 400 are relevant
    hyb_res = hybrid.search("plants sunlight photosynthesis", top_k=2)
    assert len(hyb_res) == 2
    # The top 2 should be 100 and 400 in some order
    retrieved_ids = {r["passage_id"] for r in hyb_res}
    assert retrieved_ids == {100, 400}

    # Check normalization didn't produce NaN
    assert "final_score" in hyb_res[0]
    assert not np.isnan(hyb_res[0]["final_score"])


def test_real_faiss_id_mapping(setup_tiny_corpus):
    data_path, index_path = setup_tiny_corpus

    semantic = SemanticRetriever(
        model_name=config.MODEL_NAME, index_path=index_path, data_path=data_path
    )

    # Run a generic query that matches everything to some degree
    res = semantic.search("what is this", top_k=10)

    # Ensure IDs match the original dataset
    valid_ids = {100, 200, 300, 400}
    for r in res:
        assert r["passage_id"] in valid_ids
        assert len(r["passage_text"]) > 10
