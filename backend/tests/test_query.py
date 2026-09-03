"""API tests using FastAPI's TestClient.

The heavy dependencies (embedding model + Qdrant, and the Ollama LLM) are
replaced with lightweight fakes at startup, so these tests run fast and require
neither a built vector store nor a running Ollama. The real *routing*,
*grounding gate*, and *response schema* logic is exercised end to end.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeChunk:
    def __init__(self, text, book, chapter, score):
        self.text = text
        self.book = book
        self.chapter = chapter
        self.score = score
        self.chunk_index = 0
        self.position = 0.0


class FakeRetriever:
    collection_name = "harry_potter"
    num_points = 42

    def __init__(self, settings=None):
        pass

    def retrieve(self, query, k=5):
        return [
            FakeChunk(
                "Sirius Black is Harry's godfather; he was framed and sent to Azkaban.",
                "Harry Potter and the Prisoner of Azkaban",
                "CHAPTER NINETEEN",
                0.71,
            ),
            FakeChunk(
                "Sirius offered Harry a home away from the Dursleys.",
                "Harry Potter and the Prisoner of Azkaban",
                "CHAPTER TWENTY-TWO",
                0.55,
            ),
        ]

    def ping(self):
        return True

    def close(self):
        pass


class FakeGenerator:
    model = "fake-model"
    available = True

    def __init__(self, settings=None):
        pass

    def generate(self, question, chunks):
        return (
            "Harry Potter's godfather is Sirius Black "
            "(Harry Potter and the Prisoner of Azkaban)."
        )

    def ping(self):
        return True, "ok (fake)"


@pytest.fixture
def client(monkeypatch):
    from app import main as main_module

    # Swap the heavy resources for fakes *before* the lifespan runs.
    monkeypatch.setattr(main_module, "Retriever", FakeRetriever)
    monkeypatch.setattr(main_module, "Generator", FakeGenerator)

    with TestClient(main_module.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_query_happy_path(client):
    """A valid HP question returns a grounded answer with sources."""
    resp = client.post("/query", json={"question": "Who is Harry Potter's godfather?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "hp_question"
    assert "Sirius Black" in data["answer"]
    assert len(data["sources"]) >= 1
    src = data["sources"][0]
    assert set(src) >= {"book", "chapter", "chunk_text_snippet", "score"}
    assert "Azkaban" in src["book"]


def test_query_missing_question_returns_422(client):
    """Invalid input (missing required 'question') -> 422 Unprocessable Entity."""
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_empty_question_returns_422(client):
    """An empty-string question violates min_length -> 422."""
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_greeting_route(client):
    """A greeting is handled without touching the RAG pipeline."""
    resp = client.post("/query", json={"question": "hello there!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "greeting"
    assert data["sources"] == []


def test_out_of_scope_route(client):
    """An unrelated question is politely declined, not answered."""
    resp = client.post("/query", json={"question": "What is the capital of France?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["route"] == "out_of_scope"
    assert data["sources"] == []


def test_health(client):
    """/health reports ok when both dependencies are (fake-)reachable."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vector_db" in data and "llm" in data
