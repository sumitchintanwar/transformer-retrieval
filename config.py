import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Data files
RAW_DATA_FILE = DATA_DIR / "msmarco_passages_raw.csv"
PROCESSED_DATA_FILE = DATA_DIR / "msmarco_passages.csv"
FAISS_INDEX_FILE = MODEL_DIR / "faiss_index.pickle"

# Model Parameters
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))
EMBEDDING_DIM = 384

# Data Processing
NUM_PASSAGES_TO_DOWNLOAD = int(os.getenv("NUM_PASSAGES", "10000"))
MIN_WORDS_PER_PASSAGE = 5

# Retrieval Parameters
DEFAULT_TOP_K = int(os.getenv("TOP_K", "10"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# Evaluation / Benchmarking
EVAL_FILE = BASE_DIR / "evaluation_dataset.json"
BENCHMARK_QUERIES = int(os.getenv("BENCHMARK_QUERIES", "50"))
