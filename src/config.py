from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PDF_FOLDER = PROJECT_ROOT / "data" / "medical_pdfs"
VECTORSTORE_PATH = PROJECT_ROOT / "vectorstore" / "chroma_db"
ENV_PATH = PROJECT_ROOT / ".env"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K_PER_QUERY = 6
FINAL_TOP_K = 8
KEYWORD_WEIGHT = 0.01
KEYWORD_CAP = 40