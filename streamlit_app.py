"""Self-contained Streamlit app for Streamlit Community Cloud.

Unlike frontend/app.py (which is a thin UI that calls the FastAPI backend over
HTTP), this file runs the **entire RAG pipeline in one process** — retrieval +
grounding gate + Groq generation — because Streamlit Cloud hosts a single Python
app, not a separate backend. It reuses the exact same services in
``backend/app/services`` so behaviour matches the API.

Deploy:
  * Main file path: streamlit_app.py
  * Secrets (App settings -> Secrets):
        GROQ_API_KEY = "gsk_..."
        GROQ_MODEL   = "openai/gpt-oss-20b"
  * The vector store (backend/data/vector_store) must be committed to the repo
    (repo should be PRIVATE — the chunk payloads contain the book text).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the backend package importable.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

# Bridge Streamlit secrets -> environment BEFORE settings are loaded, so
# pydantic-settings picks them up. Locally (no secrets file) this is a no-op and
# config falls back to backend/.env.
try:
    for _key in ("GROQ_API_KEY", "GROQ_MODEL", "TOP_K", "MIN_RELEVANCE_SCORE"):
        if _key in st.secrets:
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from app.core.config import get_settings  # noqa: E402
from app.services.generation import REFUSAL, Generator  # noqa: E402
from app.services.retrieval import Retriever  # noqa: E402
from app.services.router import (  # noqa: E402
    GREETING,
    GREETING_RESPONSE,
    OUT_OF_SCOPE,
    OUT_OF_SCOPE_RESPONSE,
    Router,
)

st.set_page_config(page_title="Harry Potter RAG Chatbot", page_icon="🧙", layout="centered")

ROUTE_BADGES = {
    "hp_question": "📖 book answer",
    "greeting": "👋 greeting",
    "out_of_scope": "🚫 out of scope",
}


@st.cache_resource(show_spinner="Loading the books and embedding model… (first load can take ~30–60s)")
def load_pipeline():
    """Load the retriever (embedding model + Qdrant store), generator, router once."""
    settings = get_settings()
    retriever = Retriever(settings)
    generator = Generator(settings)
    router = Router()
    return settings, retriever, generator, router


def answer_question(settings, retriever, generator, router, question: str):
    """Run the full pipeline; returns (answer, sources, route)."""
    route = router.classify(question)
    if route == GREETING:
        return GREETING_RESPONSE, [], route
    if route == OUT_OF_SCOPE:
        return OUT_OF_SCOPE_RESPONSE, [], route

    hits = retriever.retrieve(question, k=settings.top_k)
    relevant = [h for h in hits if h.score >= settings.min_relevance_score]
    if not relevant:
        return REFUSAL, [], route

    ans = generator.generate(question, relevant)
    sources = [] if REFUSAL.rstrip(".").lower() in ans.lower() else relevant
    return ans, sources, route


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**{i}. {s.book}** — *{s.chapter}* · similarity {s.score:.2f}")
            snippet = " ".join(s.text.split())
            st.caption(snippet[:240] + ("…" if len(snippet) > 240 else ""))


# --- Guard: key must be configured ---
_settings = get_settings()
if not _settings.groq_api_key:
    st.title("🧙 Harry Potter RAG Chatbot")
    st.error(
        "GROQ_API_KEY is not set. On Streamlit Cloud, add it under "
        "**Manage app → Settings → Secrets**:\n\n"
        '```toml\nGROQ_API_KEY = "gsk_your_key_here"\nGROQ_MODEL = "openai/gpt-oss-20b"\n```'
    )
    st.stop()

settings, retriever, generator, router = load_pipeline()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Status")
    st.markdown(f"🟢 **Vector DB:** {retriever.num_points} chunks")
    st.markdown(f"🟢 **LLM:** Groq · `{generator.model}`")
    st.markdown(f"**Embeddings:** `{retriever.embedding_model_name.split('/')[-1]}`")
    st.markdown(f"**top_k:** {settings.top_k} · **min score:** {settings.min_relevance_score}")
    st.divider()
    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption(
        "Ask about the 7 Harry Potter books. Answers are grounded in the book "
        "text with cited sources; unrelated questions are politely declined."
    )

# --- Chat ---
st.title("🧙 Harry Potter RAG Chatbot")
st.caption("Grounded answers from the seven books, with cited sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            badge = ROUTE_BADGES.get(msg.get("route", ""), "")
            if badge:
                st.caption(badge)
            render_sources(msg.get("sources", []))

prompt = st.chat_input("Ask a question about Harry Potter…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the books…"):
            try:
                ans, sources, route = answer_question(
                    settings, retriever, generator, router, prompt
                )
                st.markdown(ans)
                badge = ROUTE_BADGES.get(route, "")
                if badge:
                    st.caption(badge)
                render_sources(sources)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ans, "sources": sources, "route": route}
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ {exc}", "sources": [], "route": "error"}
                )
