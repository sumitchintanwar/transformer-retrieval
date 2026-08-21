import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import config


def main():
    if not config.PROCESSED_DATA_FILE.exists():
        print(f"Error: {config.PROCESSED_DATA_FILE} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(config.PROCESSED_DATA_FILE)
    print(f"Loaded {len(df)} passages")

    print(f"Loading model: {config.MODEL_NAME}")
    import torch
    model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

    print("Encoding passages...")
    embeddings = model.encode(
        df["passage_text"].tolist(),
        batch_size=config.BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, df["passage_id"].values.astype("int64"))

    print(f"FAISS index size: {index.ntotal} vectors, dimension: {dim}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.FAISS_INDEX_FILE, "wb") as f:
        pickle.dump(faiss.serialize_index(index), f)
    print(f"Index saved to {config.FAISS_INDEX_FILE}")


if __name__ == "__main__":
    main()
