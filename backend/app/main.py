"""FastAPI application entry point.

Heavy resources (embedding model, Qdrant client, Ollama client) are loaded once
in the ``lifespan`` startup and stored on ``app.state`` for reuse across
requests. Loading is wrapped in try/except so that a missing vector store or a
down Ollama produces a *degraded* /health report instead of crashing the server.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.services.router import Router
from app.utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s", settings.app_name)

    app.state.settings = settings
    app.state.retriever = None
    app.state.retriever_error = None
    app.state.generator = None
    app.state.generator_error = None

    # --- Vector store + embedding model ---
    try:
        app.state.retriever = Retriever(settings)
    except Exception as exc:
        app.state.retriever_error = str(exc)
        logger.error("Failed to load retriever: %s", exc)

    # --- LLM client (cheap to construct; reachability is checked in /health) ---
    try:
        app.state.generator = Generator(settings)
    except Exception as exc:
        app.state.generator_error = str(exc)
        logger.error("Failed to construct generator: %s", exc)

    # --- Router (heuristic; LLM fallback off by default for determinism) ---
    app.state.router = Router(generator=app.state.generator, use_llm_fallback=False)

    logger.info("Startup complete.")
    yield

    # --- Shutdown ---
    if app.state.retriever is not None:
        app.state.retriever.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Harry Potter RAG API",
    version="1.0.0",
    description=(
        "A Retrieval-Augmented Generation API that answers questions about the "
        "seven Harry Potter books, grounded strictly in the book text with cited "
        "sources."
    ),
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, tags=["rag"])


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "endpoints": ["POST /query", "GET /health"],
    }
