"""Application configuration.

All settings are read from environment variables (optionally via a ``.env``
file) using pydantic-settings, so nothing operational is hard-coded in the
source. See ``.env.example`` for every supported variable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Path anchors
# ---------------------------------------------------------------------------
# This file lives at:  backend/app/core/config.py
#   parents[0] = .../backend/app/core
#   parents[1] = .../backend/app
#   parents[2] = .../backend        <-- BACKEND_DIR
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VECTOR_STORE = BACKEND_DIR / "data" / "vector_store"


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "Harry Potter RAG API"
    log_level: str = "INFO"

    # --- Vector store (Qdrant) ---
    # Embedded/local mode uses a path on disk (no Docker). If QDRANT_URL is set,
    # the client connects to a running Qdrant server instead.
    qdrant_path: str = str(DEFAULT_VECTOR_STORE)
    qdrant_url: str | None = None
    collection_name: str = "harry_potter"

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- LLM (Groq hosted API) ---
    # Get a free key at https://console.groq.com/keys and put it in backend/.env
    # as GROQ_API_KEY=... (that file is git-ignored).
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout: float = 60.0

    # --- Retrieval / grounding ---
    # Chunks are small (~220 tokens, tuned to the embedding model), so we retrieve
    # a few more of them to give the LLM enough context (the answer often sits at
    # rank 5-8 for factual questions).
    top_k: int = 8
    # Minimum cosine similarity for a retrieved chunk to be considered relevant.
    # If the best hit is below this, we treat the question as "not covered by the
    # books" and refuse rather than let the model answer from its own knowledge.
    min_relevance_score: float = 0.30
    # Hard cap on how much retrieved text we stuff into the prompt.
    max_context_chars: int = 8000

    # --- CORS ---
    # Comma-separated list of allowed origins for the Streamlit frontend.
    cors_origins_raw: str = "http://localhost:8501,http://127.0.0.1:8501"

    @property
    def cors_origins(self) -> list[str]:
        """Parse the comma-separated origins string into a clean list."""
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def vector_store_path(self) -> Path:
        # Resolve a relative QDRANT_PATH against the backend dir (not the current
        # working directory) so it works whether the app is launched from
        # backend/, the repo root, or the notebook in notebooks/.
        p = Path(self.qdrant_path)
        if not p.is_absolute():
            p = (BACKEND_DIR / p).resolve()
        return p

    @property
    def config_json_path(self) -> Path:
        """Path to the ingestion metadata written by the notebook / ingest step."""
        return self.vector_store_path / "config.json"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once per process)."""
    return Settings()
