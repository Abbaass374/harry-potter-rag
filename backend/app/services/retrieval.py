"""Retrieval service: load the persisted Qdrant store + embedding model and
turn a natural-language query into the top-k most relevant book chunks.

Heavy dependencies (torch, sentence-transformers, qdrant-client) are imported
lazily inside methods so the module can be imported cheaply (e.g. by tests).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    book: str
    chapter: str
    score: float
    chunk_index: int
    position: float


class Retriever:
    """Loads the embedding model + Qdrant collection produced by ingestion.

    Construct once at app startup (expensive: loads the model and opens the
    on-disk collection) and reuse for every request.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._client = None
        self.collection_name = self.settings.collection_name
        self.embedding_model_name = self.settings.embedding_model
        self._load()

    # -- setup ---------------------------------------------------------------
    def _load(self) -> None:
        # Reconcile with the metadata the notebook/ingest wrote, so retrieval
        # always uses the same collection name + embedding model that built it.
        cfg_path: Path = self.settings.config_json_path
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.collection_name = cfg.get("collection_name", self.collection_name)
                self.embedding_model_name = cfg.get(
                    "embedding_model", self.embedding_model_name
                )
                logger.info(
                    "Loaded ingestion config: collection=%s, model=%s, chunks=%s",
                    self.collection_name,
                    self.embedding_model_name,
                    cfg.get("num_chunks", "?"),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not read config.json (%s); using defaults.", exc)
        else:
            logger.warning(
                "No config.json at %s. Has the notebook / ingest step been run?",
                cfg_path,
            )

        from qdrant_client import QdrantClient

        if self.settings.qdrant_url:
            self._client = QdrantClient(url=self.settings.qdrant_url)
        else:
            store = self.settings.vector_store_path
            if not store.exists():
                raise FileNotFoundError(
                    f"Vector store not found at {store}. Run the notebook "
                    "(notebooks/rag_pipeline.ipynb) or "
                    "`python -m app.services.ingest --force` to build it."
                )
            self._client = QdrantClient(path=str(store))

        if not self._client.collection_exists(self.collection_name):
            raise RuntimeError(
                f"Qdrant collection '{self.collection_name}' does not exist. "
                "Build the vector store first."
            )

        from app.services.ingest import load_embedder

        self._model = load_embedder(self.embedding_model_name)
        self.num_points = self._client.count(self.collection_name).count
        logger.info(
            "Retriever ready: %d points in '%s'.", self.num_points, self.collection_name
        )

    # -- query ---------------------------------------------------------------
    def embed_query(self, query: str):
        return self._model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks for ``query`` (highest score first)."""
        k = k or self.settings.top_k
        vector = self.embed_query(query)
        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=k,
            with_payload=True,
        ).points

        results: list[RetrievedChunk] = []
        for h in hits:
            payload = h.payload or {}
            results.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    book=payload.get("book", "Unknown"),
                    chapter=payload.get("chapter", "Unknown"),
                    score=float(h.score),
                    chunk_index=int(payload.get("chunk_index", h.id)),
                    position=float(payload.get("position", 0.0)),
                )
            )
        return results

    def ping(self) -> bool:
        """Cheap reachability check for /health."""
        try:
            self._client.count(self.collection_name)
            return True
        except Exception:  # pragma: no cover
            return False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
