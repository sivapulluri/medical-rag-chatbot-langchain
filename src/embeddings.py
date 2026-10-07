from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once and reuse it."""

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # normalized vectors give cleaner distances for search
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Multilingual embedding model loaded successfully.")

    return embedding_model


if __name__ == "__main__":

    model = get_embedding_model()

    test_embedding = model.embed_query("first aid ante enti?")

    print(f"Embedding dimension: {len(test_embedding)}")