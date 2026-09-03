"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A single retrieved chunk shown to the user as a citation."""

    book: str = Field(..., description="Book title the chunk came from.")
    chapter: str = Field(..., description="Chapter heading (or best-effort label).")
    chunk_text_snippet: str = Field(
        ..., description="A short excerpt of the retrieved chunk text."
    )
    score: float = Field(
        ..., description="Cosine similarity of the chunk to the query (0-1)."
    )


class QueryRequest(BaseModel):
    """Incoming user question."""

    question: str = Field(
        ...,
        min_length=1,
        description="A natural-language question, ideally about Harry Potter.",
        examples=["Who is Harry Potter's godfather?"],
    )


class QueryResponse(BaseModel):
    """Answer plus the routing decision and any supporting sources."""

    answer: str = Field(..., description="The generated (or canned) answer.")
    sources: list[Source] = Field(
        default_factory=list,
        description="Supporting chunks. Empty for greetings / out-of-scope / refusals.",
    )
    route: str = Field(
        ...,
        description="Which branch handled the query: greeting | out_of_scope | hp_question.",
    )


class HealthResponse(BaseModel):
    """Result of the /health check."""

    status: str = Field(..., description="'ok' if all dependencies are reachable.")
    vector_db: str = Field(..., description="Vector DB status detail.")
    llm: str = Field(..., description="LLM (Ollama) status detail.")
