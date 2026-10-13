import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embeddings import LocalEmbeddings


def test_local_embeddings_use_cached_fastembed_model(monkeypatch, tmp_path):
    class FakeVector:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            self.options = kwargs

        def embed(self, texts):
            return (FakeVector([float(len(text)), 1.0]) for text in texts)

    monkeypatch.setattr("embeddings.TextEmbedding", FakeTextEmbedding)
    cache_path = tmp_path / "model-cache"
    embeddings = LocalEmbeddings("test-model", cache_path)

    assert embeddings.embed_documents(["one", "three"]) == [
        [3.0, 1.0],
        [5.0, 1.0],
    ]
    assert embeddings.embed_query("query") == [5.0, 1.0]
    assert embeddings._client.options == {
        "model_name": "test-model",
        "cache_dir": str(cache_path),
        "threads": 2,
    }
    assert cache_path.is_dir()
