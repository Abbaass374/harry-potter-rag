"""Generation service: build a grounded prompt from retrieved chunks and call
the local Ollama LLM, refusing to answer when the context does not cover the
question.
"""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.services.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

# The exact string the model is told to emit when the context is insufficient.
REFUSAL = "I don't know based on the Harry Potter books I have access to."

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions about the seven "
    "Harry Potter books. You must follow these rules strictly:\n"
    "1. Answer ONLY using the numbered context excerpts provided by the user. "
    "Each excerpt is labelled with its book and chapter.\n"
    "2. Do NOT use any outside knowledge, even if you know the answer. If the "
    "context does not contain the answer, reply with EXACTLY this sentence and "
    f"nothing else: \"{REFUSAL}\"\n"
    "3. Keep answers concise and factual. When you use information from an "
    "excerpt, mention the book (and chapter if helpful) in your answer.\n"
    "4. Never invent citations, characters, or events that are not in the context."
)


def build_context_block(chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Render retrieved chunks into a numbered, citation-labelled context block."""
    lines: list[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        header = f"[Excerpt {i}] (Book: {c.book} | Chapter: {c.chapter})"
        snippet = c.text.strip()
        entry = f"{header}\n{snippet}"
        if used + len(entry) > max_chars and lines:
            break  # respect the context budget
        lines.append(entry)
        used += len(entry)
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: list[RetrievedChunk], max_chars: int) -> str:
    """Build the user-message content: context excerpts + the question."""
    context = build_context_block(chunks, max_chars)
    return (
        "Context excerpts from the Harry Potter books:\n"
        "----------------------------------------\n"
        f"{context}\n"
        "----------------------------------------\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above. If they do not contain the "
        f"answer, reply exactly: \"{REFUSAL}\""
    )


class Generator:
    """Grounded generation via Groq's hosted chat API.

    Needs ``GROQ_API_KEY`` (set it in ``backend/.env``). If the key is missing the
    generator still constructs, but :meth:`generate` raises a clear error and
    :meth:`ping` reports the problem — so the server boots and ``/health`` explains
    exactly what to do instead of crashing.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = self.settings.groq_model
        self._client = None

        if not self.settings.groq_api_key:
            logger.warning("GROQ_API_KEY is not set; generation will be unavailable.")
            return

        from groq import Groq

        self._client = Groq(
            api_key=self.settings.groq_api_key, timeout=self.settings.groq_timeout
        )
        logger.info("Generator using Groq (model '%s').", self.model)

    @property
    def available(self) -> bool:
        """True if a Groq client was constructed (i.e. GROQ_API_KEY was set)."""
        return self._client is not None

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a grounded answer. If there are no chunks, refuse."""
        if not chunks:
            return REFUSAL
        if self._client is None:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env to enable answers."
            )

        prompt = build_prompt(question, chunks, self.settings.max_context_chars)
        logger.debug("Prompting Groq %s with %d chunks.", self.model, len(chunks))
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,  # deterministic, grounded answers
        )
        return resp.choices[0].message.content.strip()

    def ping(self) -> tuple[bool, str]:
        """Check that Groq is reachable and the key is valid."""
        if self._client is None:
            return False, "GROQ_API_KEY not set (add it to backend/.env)"
        try:
            self._client.models.list()
            return True, f"groq ok (model '{self.model}')"
        except Exception as exc:  # pragma: no cover - depends on runtime
            return False, f"groq unreachable or invalid key: {exc}"
