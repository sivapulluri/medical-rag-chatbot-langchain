import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import GeminiEmbeddings, normalize


def test_normalize():
    vec = [3.0, 4.0]
    norm = normalize(vec)
    assert abs(norm[0] - 0.6) < 1e-5
    assert abs(norm[1] - 0.8) < 1e-5
    assert normalize([0.0, 0.0]) == [0.0, 0.0]


def test_gemini_embeddings_wrapper(monkeypatch):
    class FakeGoogleEmbeddings:
        def __init__(self, model, google_api_key, output_dimensionality):
            self.model = model
            self.api_key = google_api_key
            self.dims = output_dimensionality

        def embed_documents(self, texts):
            return [[3.0, 4.0] for _ in texts]

        def embed_query(self, text):
            return [3.0, 4.0]

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("embeddings.GoogleGenerativeAIEmbeddings", FakeGoogleEmbeddings)

    emb = GeminiEmbeddings("test-model", 768)
    docs = emb.embed_documents(["hello", "world"])
    assert len(docs) == 2
    assert abs(docs[0][0] - 0.6) < 1e-5

    q = emb.embed_query("test query")
    assert abs(q[0] - 0.6) < 1e-5
    # Query caching
    assert "test query" in emb._query_cache

