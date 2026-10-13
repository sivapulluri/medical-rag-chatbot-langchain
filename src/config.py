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


def _get_project_path(name: str, default: Path) -> Path:
    """Resolve an environment path, anchoring relative paths at the project root."""
    raw_value = os.getenv(name)
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# -------------------------------------------------------------------
# Gemini API embeddings (lightweight, zero RAM overhead)
# -------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_MODEL_NAME = os.getenv("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
EMBEDDING_DIMENSIONS = _get_int_env("EMBEDDING_DIMENSIONS", 768)
VECTORSTORE_PATH = _get_project_path(
    "VECTORSTORE_PATH", PROJECT_ROOT / "vectorstore" / "chroma_db"
)

# Keep chunks within the embedding model's 128-token input limit.
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", 100)

# Building the vector store.
EMBED_BATCH_SIZE = _get_int_env("EMBED_BATCH_SIZE", 50)

# Retrieval
TOP_K_PER_QUERY = _get_int_env("TOP_K_PER_QUERY", 6)
FINAL_TOP_K = _get_int_env("FINAL_TOP_K", 6)
KEYWORD_WEIGHT = float(os.getenv("KEYWORD_WEIGHT", "0.01"))
KEYWORD_CAP = _get_int_env("KEYWORD_CAP", 40)