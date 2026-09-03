"""HTTP client for the FastAPI backend.

All network calls live here (never inline in app.py). Functions raise
``APIError`` with a friendly, user-facing message on any failure so the UI can
show ``st.error(...)`` instead of a raw traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Generation can be slow on CPU; give the LLM room to answer.
QUERY_TIMEOUT = float(os.getenv("QUERY_TIMEOUT", "120"))
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "5"))


class APIError(Exception):
    """Raised for any backend/network problem, with a friendly message."""


@dataclass
class QueryResult:
    answer: str
    sources: list[dict]
    route: str


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return "The backend timed out while connecting. Is it running?"
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return (
            "The backend took too long to respond. The model may still be "
            "loading - try again in a moment."
        )
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach the backend at {API_BASE_URL}. "
            "Start it with: uvicorn app.main:app --reload"
        )
    return f"Unexpected error talking to the backend: {exc}"


def check_health() -> dict:
    """Return the backend's /health payload. Raises APIError on failure."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as exc:
        raise APIError(f"Health check failed (HTTP {resp.status_code}).") from exc
    except Exception as exc:
        raise APIError(_friendly_error(exc)) from exc


def ask_question(question: str) -> QueryResult:
    """POST a question to /query and return a structured result.

    Raises APIError (with a friendly message) on any network/HTTP problem.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=QUERY_TIMEOUT,
        )
    except Exception as exc:
        raise APIError(_friendly_error(exc)) from exc

    if resp.status_code == 422:
        raise APIError("Please enter a non-empty question.")
    if resp.status_code == 503:
        detail = _safe_detail(resp)
        raise APIError(f"The backend is not ready: {detail}")
    if resp.status_code >= 500:
        raise APIError("The backend hit an internal error (HTTP 500). Check its logs.")
    if resp.status_code != 200:
        raise APIError(f"Unexpected response (HTTP {resp.status_code}).")

    data = resp.json()
    return QueryResult(
        answer=data.get("answer", ""),
        sources=data.get("sources", []),
        route=data.get("route", "unknown"),
    )


def _safe_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", "no detail")
    except Exception:
        return "no detail"
