"""Gemini embeddings API wrapped for LangChain and Chroma."""

import math
import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL_NAME


def normalize(vector):
    """Scale a vector to length 1 (needed for cosine / L2 distance comparisons)."""
    length = math.sqrt(sum(x * x for x in vector))
    if length == 0:
        return list(vector)
    return [x / length for x in vector]


class GeminiEmbeddings(Embeddings):
    """Gemini API embeddings (low RAM, fast, no model download)."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.model_name = model_name
        self.dimensions = dimensions
        self._client = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
            output_dimensionality=dimensions,
        )
        self._query_cache = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._client.embed_documents(list(texts))
        return [normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            if len(self._query_cache) >= 512:
                self._query_cache.clear()
            self._query_cache[text] = normalize(self._client.embed_query(text))
        return self._query_cache[text]


@lru_cache(maxsize=1)
def get_embedding_model() -> GeminiEmbeddings:
    model = GeminiEmbeddings(EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS)
    print(f"Gemini embeddings ready ({EMBEDDING_MODEL_NAME}).")
    return model


if __name__ == "__main__":
    model = get_embedding_model()
    test_embedding = model.embed_query("first aid ante enti?")
    print(f"Embedding dimension: {len(test_embedding)}")

