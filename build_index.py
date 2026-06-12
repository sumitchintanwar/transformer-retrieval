import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_FILE = os.path.join("data", "msmarco_passages.csv")
INDEX_DIR = "models"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss_index.pickle")
BATCH_SIZE = 256


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} passages")

    print(f"Loading model: {MODEL_NAME}")
    import torch
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Encoding passages...")
    embeddings = model.encode(
        df["passage_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, df["passage_id"].values.astype("int64"))

    print(f"FAISS index size: {index.ntotal} vectors, dimension: {dim}")

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(faiss.serialize_index(index), f)
    print(f"Index saved to {INDEX_FILE}")


if __name__ == "__main__":
    main()
