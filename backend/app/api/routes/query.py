"""API routes: POST /query (router -> retrieve -> generate) and GET /health."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import HealthResponse, QueryResponse, QueryRequest, Source
from app.services.generation import REFUSAL
from app.services.router import (
    GREETING,
    GREETING_RESPONSE,
    HP_QUESTION,
    OUT_OF_SCOPE,
    OUT_OF_SCOPE_RESPONSE,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + "..."


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Route the question, then (for HP questions) retrieve context and answer."""
    state = request.app.state
    settings = state.settings
    question = payload.question.strip()

    route = state.router.classify(question)

    # --- Non-RAG routes: canned replies, no retrieval, no LLM. ---
    if route == GREETING:
        return QueryResponse(answer=GREETING_RESPONSE, sources=[], route=route)
    if route == OUT_OF_SCOPE:
        return QueryResponse(answer=OUT_OF_SCOPE_RESPONSE, sources=[], route=route)

    # --- hp_question: full RAG pipeline. ---
    retriever = state.retriever
    generator = state.generator
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store is not loaded. Build it via the notebook or "
                "`python -m app.services.ingest --force`. "
                f"({getattr(state, 'retriever_error', 'unknown error')})"
            ),
        )

    chunks = retriever.retrieve(question, k=settings.top_k)

    # Grounding gate: if nothing is similar enough, refuse instead of guessing.
    relevant = [c for c in chunks if c.score >= settings.min_relevance_score]
    if not relevant:
        logger.info(
            "No chunk >= min_relevance_score (%.2f) for %r; refusing.",
            settings.min_relevance_score, question,
        )
        return QueryResponse(answer=REFUSAL, sources=[], route=route)

    if generator is None or not generator.available:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM is not available. Set GROQ_API_KEY in backend/.env "
                f"(get a free key at https://console.groq.com/keys). "
                f"({getattr(state, 'generator_error', 'no API key configured')})"
            ),
        )

    answer = generator.generate(question, relevant)

    # If the model refused, don't attach (misleading) sources.
    sources: list[Source] = []
    if REFUSAL.rstrip(".").lower() not in answer.lower():
        sources = [
            Source(
                book=c.book,
                chapter=c.chapter,
                chunk_text_snippet=_snippet(c.text),
                score=round(c.score, 4),
            )
            for c in relevant
        ]

    return QueryResponse(answer=answer, sources=sources, route=route)


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report reachability of the vector DB and the LLM."""
    state = request.app.state

    retriever = state.retriever
    if retriever is not None and retriever.ping():
        vdb_ok = True
        vector_db = f"ok ({retriever.num_points} chunks in '{retriever.collection_name}')"
    else:
        vdb_ok = False
        vector_db = getattr(state, "retriever_error", None) or "vector store not loaded"

    generator = state.generator
    if generator is not None:
        llm_ok, llm_msg = generator.ping()
    else:
        llm_ok = False
        llm_msg = getattr(state, "generator_error", None) or "LLM client not loaded"

    status = "ok" if (vdb_ok and llm_ok) else "degraded"
    return HealthResponse(status=status, vector_db=vector_db, llm=llm_msg)
