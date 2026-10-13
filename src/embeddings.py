"""Local, multilingual embeddings for LangChain and Chroma."""

from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from config import (
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL_NAME,
    FASTEMBED_THREADS,
)


class LocalEmbeddings(Embeddings):
    """CPU-based multilingual embeddings using FastEmbed and ONNX Runtime."""

    def __init__(
        self,
        model_name: str,
        cache_path: Path = EMBEDDING_CACHE_PATH,
    ):
        cache_path.mkdir(parents=True, exist_ok=True)
        self._client = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_path),
            threads=FASTEMBED_THREADS,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.embed(list(texts))
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self._client.embed([text]))
        return vector.tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> LocalEmbeddings:
    model = LocalEmbeddings(EMBEDDING_MODEL_NAME)
    print(f"Local embeddings ready ({EMBEDDING_MODEL_NAME}).")
    return model


if __name__ == "__main__":
    model = get_embedding_model()
    test_embedding = model.embed_query("first aid ante enti?")
    print(f"Embedding dimension: {len(test_embedding)}")
