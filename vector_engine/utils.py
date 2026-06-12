import numpy as np


def vector_search(query, model, index, num_results=10):
    """Encode query and search FAISS index for nearest neighbors.

    Args:
        query (list[str]): User search queries.
        model (SentenceTransformer): Sentence transformer model.
        index (faiss.Index): FAISS index to search.
        num_results (int): Number of results to return.

    Returns:
        D (numpy.ndarray): Cosine similarity scores for each result.
        I (numpy.ndarray): Passage IDs of the results.
    """
    vector = model.encode(list(query), normalize_embeddings=True)
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I
