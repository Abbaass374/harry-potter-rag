"""Streamlit chat frontend for the Harry Potter RAG chatbot.

Run with:
    streamlit run app.py

All HTTP calls are delegated to api_client.py.
"""

from __future__ import annotations

import streamlit as st

from api_client import API_BASE_URL, APIError, ask_question, check_health

st.set_page_config(page_title="Harry Potter RAG Chatbot", page_icon="🧙", layout="centered")

# Route label -> small badge shown next to answers.
ROUTE_BADGES = {
    "hp_question": "📖 book answer",
    "greeting": "👋 greeting",
    "out_of_scope": "🚫 out of scope",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    # Each message: {"role": "user"|"assistant", "content": str,
    #                "sources": list[dict], "route": str}
    st.session_state.messages = []
if "health" not in st.session_state:
    st.session_state.health = None


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources, start=1):
            score = src.get("score")
            score_str = f" · similarity {score:.2f}" if isinstance(score, (int, float)) else ""
            st.markdown(
                f"**{i}. {src.get('book', 'Unknown')}** — "
                f"*{src.get('chapter', 'Unknown')}*{score_str}"
            )
            st.caption(src.get("chunk_text_snippet", ""))


def _refresh_health() -> None:
    try:
        st.session_state.health = {"ok": True, "data": check_health()}
    except APIError as exc:
        st.session_state.health = {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# Sidebar: health indicator + controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Status")
    st.caption(f"Backend: `{API_BASE_URL}`")

    if st.button("🔄 Check health"):
        _refresh_health()

    # Auto-check once on first load.
    if st.session_state.health is None:
        _refresh_health()

    health = st.session_state.health
    if health and health.get("ok"):
        data = health["data"]
        overall = data.get("status", "unknown")
        dot = "🟢" if overall == "ok" else "🟡"
        st.markdown(f"{dot} **Backend:** {overall}")
        st.markdown(f"**Vector DB:** {data.get('vector_db', '?')}")
        st.markdown(f"**LLM:** {data.get('llm', '?')}")
    else:
        err = health.get("error") if health else "not checked"
        st.markdown(f"🔴 **Backend unreachable**")
        st.caption(err)

    st.divider()
    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Ask about the 7 Harry Potter books. Answers are grounded in the book "
        "text and cite their sources; unrelated questions are politely declined."
    )


# --------------------------------------------------------------------------- #
# Main chat area
# --------------------------------------------------------------------------- #
st.title("🧙 Harry Potter RAG Chatbot")
st.caption("Grounded answers from the seven books, with cited sources.")

# Replay conversation history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            badge = ROUTE_BADGES.get(msg.get("route", ""), "")
            if badge:
                st.caption(badge)
            _render_sources(msg.get("sources", []))

# Chat input.
prompt = st.chat_input("Ask a question about Harry Potter...")
if prompt:
    # Show + store the user's message.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant's answer.
    with st.chat_message("assistant"):
        with st.spinner("Consulting the books..."):
            try:
                result = ask_question(prompt)
                st.markdown(result.answer)
                badge = ROUTE_BADGES.get(result.route, "")
                if badge:
                    st.caption(badge)
                _render_sources(result.sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": result.sources,
                        "route": result.route,
                    }
                )
            except APIError as exc:
                st.error(str(exc))
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"⚠️ {exc}",
                        "sources": [],
                        "route": "error",
                    }
                )
