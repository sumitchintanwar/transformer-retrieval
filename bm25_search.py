import os
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
import config


def tokenize(text):
    """Lowercase and split text into word tokens."""
    return text.lower().split()


class BM25Retriever:
    """BM25 keyword retriever over MS MARCO passages.

    Builds an inverted index using BM25Okapi and provides
    lexical search over the passage corpus.

    Args:
        data_path (str): Path to the preprocessed passages CSV.
    """

    def __init__(self, data_path=config.PROCESSED_DATA_FILE):
        df = pd.read_csv(data_path)
        self.passage_ids = df["passage_id"].values
        self.passage_texts = df["passage_text"].values

        tokenized_corpus = [tokenize(t) for t in self.passage_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=10):
        """Search the BM25 index.

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
            
        actual_top_k = min(top_k, len(self.passage_ids))
        if actual_top_k == 0:
            return []
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:actual_top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append({
                "passage_id": int(self.passage_ids[idx]),
                "score": float(scores[idx]),
                "passage_text": self.passage_texts[idx],
            })
        return results


if __name__ == "__main__":
    retriever = BM25Retriever()

    test_queries = [
        "how does photosynthesis work",
        "what is the reserve bank of australia",
        "covid-19 misinformation on social media",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retriever.search(query, top_k=3)
        for rank, r in enumerate(results, 1):
            print(f"  #{rank} (score={r['score']:.2f}) id={r['passage_id']}")
            print(f"     {r['passage_text'][:100]}...")
