"""Central configuration for the Medical RAG project.

All paths are anchored to the project root so scripts behave consistently
regardless of the current working directory.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PDF_FOLDER = PROJECT_ROOT / "data" / "medical_pdfs"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env here so every setting below can read it.
load_dotenv(ENV_PATH)


def _get_int_env(name: str, default: int) -> int:
    """Return a validated integer from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name!r} must be an integer.") from exc


# -------------------------------------------------------------------
# Embeddings: Gemini API (no local torch model, so very low RAM)
# -------------------------------------------------------------------
# NOTE: if you change the embedding model or its dimensions, you MUST
# rebuild the vector store (python src/vector_store.py).
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
EMBEDDING_DIMENSIONS = _get_int_env("GEMINI_EMBEDDING_DIMENSIONS", 768)

# Chunking (Gemini accepts up to 2048 tokens, so chunks can be bigger than
# with an older local embedding model).
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", 1000)
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", 200)

# Building the vector store (stay inside the free API quota).
EMBED_BATCH_SIZE = _get_int_env("EMBED_BATCH_SIZE", 50)
EMBED_SLEEP_SECONDS = float(os.getenv("EMBED_SLEEP_SECONDS", "1.0"))

# Retrieval
TOP_K_PER_QUERY = _get_int_env("TOP_K_PER_QUERY", 6)
FINAL_TOP_K = _get_int_env("FINAL_TOP_K", 6)
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.01"))
KEYWORD_CAP = _get_int_env("KEYWORD_CAP", 40)