"""API tests (no Gemini key, vector store or model download needed).

Run from the project root:  pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

import api

# Not used as a context manager, so the real startup (loading the
# model and Gemini) is skipped. Each test injects fakes instead.
client = TestClient(api.app)


def test_frontend_is_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "Carefully" in response.text


def test_frontend_assets_are_served():
    for asset in ("styles.css", "app.js"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200


def test_health_reports_not_ready_without_chain(monkeypatch):
    monkeypatch.setattr(api, "state", {})
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "rag_ready": False}


def test_ask_returns_answer(monkeypatch):
    monkeypatch.setitem(api.state, "rag_chain", lambda q: f"answer to: {q}")
    response = client.post("/ask", json={"question": "What is diabetes?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "answer to: What is diabetes?"}


def test_ask_rejects_too_short_question(monkeypatch):
    monkeypatch.setitem(api.state, "rag_chain", lambda q: "x")
    response = client.post("/ask", json={"question": "a"})
    assert response.status_code == 422


def test_ask_503_when_chain_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "state", {})
    response = client.post("/ask", json={"question": "What is diabetes?"})
    assert response.status_code == 503


def test_ask_500_when_chain_fails(monkeypatch):
    def boom(q):
        raise RuntimeError("model down")

    monkeypatch.setitem(api.state, "rag_chain", boom)
    response = client.post("/ask", json={"question": "What is diabetes?"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to generate an answer."


def test_analyze_image_ok(monkeypatch):
    monkeypatch.setattr(
        api,
        "analyze_medical_image",
        lambda b64, question, mime: f"{mime}|{question}|{len(b64) > 0}",
    )
    response = client.post(
        "/analyze-image",
        files={"image": ("strip.png", b"fake-bytes", "image/png")},
        data={"question": "what is this?"},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == "image/png|what is this?|True"


def test_analyze_image_uses_default_question(monkeypatch):
    monkeypatch.setattr(
        api, "analyze_medical_image", lambda b64, question, mime: question
    )
    response = client.post(
        "/analyze-image",
        files={"image": ("a.jpg", b"fake-bytes", "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["answer"] == api.DEFAULT_IMAGE_QUESTION


def test_analyze_image_rejects_wrong_type():
    response = client.post(
        "/analyze-image",
        files={"image": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


def test_analyze_image_rejects_empty_file():
    response = client.post(
        "/analyze-image",
        files={"image": ("a.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_analyze_image_rejects_large_file(monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 10)
    response = client.post(
        "/analyze-image",
        files={"image": ("big.png", b"x" * 100, "image/png")},
    )
    assert response.status_code == 413
