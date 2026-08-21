from logger import get_logger

logger = get_logger(__name__)

import pickle

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

import config


def main():
    if not config.PROCESSED_DATA_FILE.exists():
        logger.error(
            f"Error: {config.PROCESSED_DATA_FILE} not found. Run preprocess.py first."
        )
        return

    logger.info(f"Loading model: {config.MODEL_NAME}")
    model = SentenceTransformer(config.MODEL_NAME, device=config.DEVICE)

    # Initialize FAISS Index
    dim = model.get_sentence_embedding_dimension()
    # We use inner product since we will normalize the embeddings (equivalent to cosine similarity)
    index = faiss.IndexFlatIP(dim)
    # Wrap in IDMap to store our passage IDs
    index = faiss.IndexIDMap(index)

    chunk_size = 100000  # Number of rows to read from CSV at a time
    logger.info(f"Starting batch encoding and indexing (chunk size: {chunk_size})")

    total_encoded = 0

    # Process in chunks to maintain constant memory
    for chunk in pd.read_csv(config.PROCESSED_DATA_FILE, chunksize=chunk_size):
        passages = chunk["passage_text"].tolist()
        passage_ids = chunk["passage_id"].values

        logger.info(f"Encoding chunk of {len(passages)} passages...")
        embeddings = model.encode(
            passages,
            batch_size=config.BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        logger.info("Adding to FAISS index...")
        index.add_with_ids(embeddings.astype("float32"), passage_ids.astype("int64"))

        total_encoded += len(passages)
        logger.info(f"Total passages indexed so far: {total_encoded}")

    logger.info(f"Final FAISS index size: {index.ntotal} vectors, dimension: {dim}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.FAISS_INDEX_FILE, "wb") as f:
        pickle.dump(faiss.serialize_index(index), f)
    logger.info(f"Index saved to {config.FAISS_INDEX_FILE}")


if __name__ == "__main__":
    main()
