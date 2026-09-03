"""Data-prep pipeline: PDF -> clean text -> chunks -> embeddings -> Qdrant.

This is the single source of truth for ingestion. The Jupyter notebook
(``notebooks/rag_pipeline.ipynb``) imports these functions so the report and the
production index are built from identical logic, and the backend simply *reads*
the collection this module produces.

It can also be run standalone to (re)build the vector store::

    cd backend
    python -m app.services.ingest --force

Design notes
------------
* Chunk size is tuned to the embedding model, not to a generic rule of thumb.
  ``all-MiniLM-L6-v2`` truncates inputs at 256 word-piece tokens, so chunks are
  sized to ~220 tokens to stay safely under that cap (larger chunks would be
  silently truncated at embedding time and their tail text would never be
  represented in the vector). See ``notebooks/rag_pipeline.ipynb`` 5.4 for the
  full justification.
* Chunking is chapter-aware: chunks never cross a chapter boundary, so every
  chunk carries a clean ``book`` + ``chapter`` citation.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.core.config import BACKEND_DIR, get_settings

logger = logging.getLogger(__name__)

# --- Repo-relative default paths ---------------------------------------------
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "harry_potter_complete.pdf"
DEFAULT_MARKDOWN = PROJECT_ROOT / "data" / "processed" / "harry_potter.md"

# --- Chunking parameters (tuned for all-MiniLM-L6-v2's 256-token limit) ------
CHARS_PER_TOKEN = 4          # rough English heuristic (chars / 4 ~= tokens)
TARGET_TOKENS = 220          # comfortably under the model's 256-token cap
OVERLAP_TOKENS = 32          # ~15% overlap to preserve context across cuts
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN     # 880
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN   # 128

# --- Canonical Harry Potter book titles, in publication order -----------------
# The single supplied PDF concatenates all seven books, but they are NOT
# separated by repeated title pages: the titles appear only once, together, in
# the table of contents. So we cannot segment by title text. Instead we split on
# where the chapter numbering RESETS to "Chapter One" (each book restarts at 1),
# and assign these titles to the segments in order. (This PDF is the US edition,
# hence "Sorcerer's Stone".)
BOOK_TITLES: list[str] = [
    "Harry Potter and the Sorcerer's Stone",
    "Harry Potter and the Chamber of Secrets",
    "Harry Potter and the Prisoner of Azkaban",
    "Harry Potter and the Goblet of Fire",
    "Harry Potter and the Order of the Phoenix",
    "Harry Potter and the Half-Blood Prince",
    "Harry Potter and the Deathly Hallows",
]

# Chapter / epilogue headings, tolerant of the markdown noise pymupdf4llm emits,
# e.g. "#### **CHAPTER  ONE**", "**CHAPTER ONE**", "### EPILOGUE".
_HEADING_RE = re.compile(
    r"^[#>\s]*\*{0,3}_{0,2}\s*(chapter\b[^\n]*|epilogue\b[^\n]*)",
    re.IGNORECASE | re.MULTILINE,
)

# English number words for parsing chapter ordinals ("Chapter Twenty-One" -> 21).
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


def _strip_md(s: str) -> str:
    """Remove markdown noise (#, *, _, backticks, <u>) and collapse whitespace."""
    s = re.sub(r"</?u>", "", s, flags=re.IGNORECASE)
    s = s.replace("#", "").replace("*", "").replace("_", "").replace("`", "")
    return " ".join(s.split()).strip()


def _title_case(s: str) -> str:
    # Title-case, then fix contractions ("Won'T" -> "Won't", "Dobby'S" -> "Dobby's").
    return re.sub(r"'(\w)", lambda m: "'" + m.group(1).lower(), s.title())


def _ordinal_to_int(text: str) -> int | None:
    """Parse 'ONE', 'twenty-one', '21' -> int; None if it isn't a number."""
    t = _strip_md(text).lower().replace("-", " ").split(":")[0].strip()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    total, matched = 0, False
    for word in t.split():
        if word in _NUM_WORDS:
            total += _NUM_WORDS[word]
            matched = True
        else:
            break
    return total if matched else None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    book: str
    chapter: str
    chunk_index: int
    position: float                       # 0..1, approx location within its book
    char_start: int
    token_estimate: int
    extra: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "text": self.text,
            "book": self.book,
            "chapter": self.chapter,
            "chunk_index": self.chunk_index,
            "position": round(self.position, 4),
            "char_start": self.char_start,
            "token_estimate": self.token_estimate,
        }


# ---------------------------------------------------------------------------
# 1. PDF -> Markdown/text
# ---------------------------------------------------------------------------
def pdf_to_markdown(pdf_path: Path, md_out: Path | None = None) -> str:
    """Convert the PDF to Markdown text.

    Prefers ``pymupdf4llm`` (layout- and heading-aware); falls back to ``pypdf``
    page-text extraction if it is unavailable or fails.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Place the source PDF there first."
        )

    text = ""
    try:
        import pymupdf4llm  # type: ignore

        logger.info("Parsing PDF with pymupdf4llm: %s", pdf_path)
        text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("pymupdf4llm unavailable/failed (%s); falling back to pypdf", exc)
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)

    if md_out is not None:
        md_out = Path(md_out)
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(text, encoding="utf-8")
        logger.info("Wrote raw markdown (%d chars) -> %s", len(text), md_out)

    return text


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
_PAGE_NUM_RE = re.compile(r"^\s*(?:page\s+)?\d{1,4}\s*$", re.IGNORECASE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip page numbers / OCR artifacts and normalise whitespace & quotes."""
    # Normalise unicode (curly quotes, ligatures, dashes) to plain forms.
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", " ": " ",
        "\x0c": "\n",  # form feed -> newline
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop standalone page numbers / "Page N" running footers.
        if _PAGE_NUM_RE.match(stripped):
            continue
        cleaned_lines.append(line.rstrip())

    text = "\n".join(cleaned_lines)
    text = _MULTI_BLANK_RE.sub("\n\n", text)  # collapse >2 blank lines
    return text.strip()


# ---------------------------------------------------------------------------
# 3. Structure detection (books & chapters)
# ---------------------------------------------------------------------------
def detect_books(text: str) -> list[tuple[str, int, int]]:
    """Return ``[(book_title, start_idx, end_idx), ...]`` spanning the text.

    Books are split where the chapter numbering resets to "Chapter One". The
    first book absorbs any front matter before its Chapter One. If no chapter-one
    markers are found (unexpected formatting), the whole document is returned as a
    single book so ingestion still succeeds.
    """
    starts: list[int] = []  # char offsets where a new book begins (chapter == 1)
    for m in _HEADING_RE.finditer(text):
        head = _strip_md(m.group(1))
        if head.lower().startswith("chapter"):
            if _ordinal_to_int(head[len("chapter"):]) == 1:
                starts.append(m.start())

    if not starts:
        logger.warning("No 'Chapter One' markers found; treating PDF as one book.")
        return [("Harry Potter (collected)", 0, len(text))]

    starts[0] = 0  # first book includes the front matter before its Chapter One
    if len(starts) != len(BOOK_TITLES):
        logger.warning(
            "Detected %d book(s) by chapter-one resets (expected %d).",
            len(starts), len(BOOK_TITLES),
        )

    spans: list[tuple[str, int, int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        title = BOOK_TITLES[i] if i < len(BOOK_TITLES) else f"Harry Potter (Book {i + 1})"
        spans.append((title, start, end))
    return spans


def _first_title_line(tail: str) -> str | None:
    """Grab a chapter's title from the heading line following 'Chapter N', if any."""
    for line in tail.splitlines():
        raw = line.strip()
        if not raw:
            continue
        stripped = _strip_md(raw)
        if not stripped:
            continue
        # If the next heading is itself a chapter/epilogue, there's no title.
        if stripped.lower().startswith(("chapter", "epilogue")):
            return None
        # Treat it as a title only if it was a heading / emphasised line.
        if raw.startswith("#") or raw.startswith("*") or raw.startswith("_") or raw.isupper():
            # Cut at sentence punctuation in case the PDF merged body text in.
            title = re.split(r'[.?!"]', stripped)[0].strip()
            return _title_case(title)[:60] or None
        return None
    return None


def _chapter_label(book_text: str, m: "re.Match") -> str:
    """Build a readable label like 'Chapter One: The Boy Who Lived'."""
    head = _strip_md(m.group(1))
    if head.lower().startswith("epilogue"):
        return "Epilogue"
    ordinal_txt = _strip_md(head[len("chapter"):]).strip(" :-")
    ordinal = " ".join(w.capitalize() for w in ordinal_txt.split()) or ordinal_txt
    base = f"Chapter {ordinal}".strip()
    title = _first_title_line(book_text[m.end(): m.end() + 300])
    return f"{base}: {title}" if title else base


def detect_chapters(book_text: str) -> list[tuple[str, int, int]]:
    """Split a book's text into ``[(chapter_label, start, end), ...]`` spans."""
    matches = list(_HEADING_RE.finditer(book_text))
    if not matches:
        return [("(full text)", 0, len(book_text))]

    spans: list[tuple[str, int, int]] = []
    # Any preamble before the first chapter heading (title page, dedication).
    if matches[0].start() > 0:
        spans.append(("Front matter", 0, matches[0].start()))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(book_text)
        spans.append((_chapter_label(book_text, m), start, end))
    return spans


# ---------------------------------------------------------------------------
# 4. Chunking (paragraph-greedy with overlap, scoped to a chapter)
# ---------------------------------------------------------------------------
def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Hard-split an over-long paragraph on sentence, then word boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    buf = ""
    for sent in sentences:
        if len(sent) > max_chars:  # a single monster sentence -> split on words
            words = sent.split(" ")
            for w in words:
                if len(buf) + len(w) + 1 > max_chars:
                    pieces.append(buf.strip())
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            continue
        if len(buf) + len(sent) + 1 > max_chars:
            pieces.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf.strip():
        pieces.append(buf.strip())
    return [p for p in pieces if p]


def chunk_chapter_text(
    text: str,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    """Greedily merge paragraphs into ~target-sized chunks with tail overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Ensure no paragraph on its own exceeds the target size.
    normalised: list[str] = []
    for p in paragraphs:
        normalised.extend(_split_long_paragraph(p, target_chars))

    chunks: list[str] = []
    buf = ""
    for para in normalised:
        if buf and len(buf) + len(para) + 2 > target_chars:
            chunks.append(buf.strip())
            # Start next chunk with an overlapping tail of the previous one.
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail} {para}".strip() if tail else para
        else:
            buf = f"{buf}\n\n{para}".strip() if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def build_chunks(cleaned_text: str) -> list[Chunk]:
    """Turn a cleaned full-corpus string into a list of metadata-rich chunks."""
    chunks: list[Chunk] = []
    idx = 0
    for book_title, b_start, b_end in detect_books(cleaned_text):
        book_text = cleaned_text[b_start:b_end]
        book_len = max(1, len(book_text))
        for chap_label, c_start, c_end in detect_chapters(book_text):
            chapter_text = book_text[c_start:c_end]
            for piece in chunk_chapter_text(chapter_text):
                position = c_start / book_len  # approx position within the book
                chunks.append(
                    Chunk(
                        text=piece,
                        book=book_title,
                        chapter=chap_label,
                        chunk_index=idx,
                        position=position,
                        char_start=b_start + c_start,
                        token_estimate=_estimate_tokens(piece),
                    )
                )
                idx += 1
    logger.info("Built %d chunks across %d book span(s).", len(chunks),
                len(detect_books(cleaned_text)))
    return chunks


# ---------------------------------------------------------------------------
# 5. Embedding model helper
# ---------------------------------------------------------------------------
def pick_device() -> str:
    """Use Apple Silicon GPU (MPS) if available, else CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def load_embedder(model_name: str):
    """Load a SentenceTransformer, capped at the model's 256-token window."""
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    logger.info("Loading embedding model '%s' on device '%s'", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 256  # explicit: all-MiniLM-L6-v2's hard limit
    return model


# ---------------------------------------------------------------------------
# 6. Build + persist the Qdrant collection
# ---------------------------------------------------------------------------
def prepare_chunks(
    pdf_path: Path = DEFAULT_PDF, md_out: Path = DEFAULT_MARKDOWN
) -> tuple[str, list[Chunk]]:
    """Parse -> clean -> chunk the PDF. Returns ``(cleaned_text, chunks)``."""
    raw = pdf_to_markdown(pdf_path, md_out)
    cleaned = clean_text(raw)
    chunks = build_chunks(cleaned)
    if not chunks:
        raise RuntimeError("No chunks were produced from the PDF.")
    return cleaned, chunks


def embed_and_store(
    chunks: list[Chunk],
    source_pdf: Path = DEFAULT_PDF,
    vector_store_path: Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    batch_size: int = 128,
) -> dict:
    """Embed the given chunks and (re)create the Qdrant collection + config.json.

    Always rebuilds the collection from the supplied chunks. Returns the metadata
    dict that is also written to ``config.json``.
    """
    settings = get_settings()
    vector_store_path = Path(vector_store_path or settings.vector_store_path)
    collection_name = collection_name or settings.collection_name
    embedding_model = embedding_model or settings.embedding_model
    vector_store_path.mkdir(parents=True, exist_ok=True)

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(path=str(vector_store_path))
    try:
        # --- Embed ---
        model = load_embedder(embedding_model)
        vectors = model.encode(
            [c.text for c in chunks],
            batch_size=batch_size,
            normalize_embeddings=True,   # unit vectors -> cosine == dot product
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        dim = int(vectors.shape[1])

        # --- (Re)create collection & upsert ---
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=c.chunk_index, vector=vec.tolist(), payload=c.payload())
            for c, vec in zip(chunks, vectors)
        ]
        for start in range(0, len(points), 256):
            client.upsert(collection_name, points=points[start:start + 256])
        logger.info("Upserted %d points into '%s'.", len(points), collection_name)

        # --- Persist ingestion metadata ---
        config = {
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "vector_size": dim,
            "distance": "COSINE",
            "chunk_size_tokens": TARGET_TOKENS,
            "chunk_overlap_tokens": OVERLAP_TOKENS,
            "chars_per_token": CHARS_PER_TOKEN,
            "num_chunks": len(chunks),
            "books_detected": sorted({c.book for c in chunks}),
            "source_pdf": str(source_pdf),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        settings.config_json_path.write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
        logger.info("Wrote %s", settings.config_json_path)
        return config
    finally:
        client.close()


def build_vector_store(
    pdf_path: Path = DEFAULT_PDF,
    md_out: Path = DEFAULT_MARKDOWN,
    force: bool = False,
) -> dict:
    """Run the full pipeline and persist the collection + ``config.json``.

    If the collection already exists and ``force`` is False, it is left as-is.
    Returns the metadata dict (also written to ``config.json``).
    """
    settings = get_settings()

    if not force:
        from qdrant_client import QdrantClient

        client = QdrantClient(path=str(settings.vector_store_path))
        try:
            if client.collection_exists(settings.collection_name):
                count = client.count(settings.collection_name).count
                logger.info(
                    "Collection '%s' already has %d points; skipping rebuild "
                    "(use force=True to rebuild).", settings.collection_name, count,
                )
                return _read_config(settings.config_json_path)
        finally:
            client.close()

    _, chunks = prepare_chunks(pdf_path, md_out)
    return embed_and_store(chunks, source_pdf=pdf_path)


def _read_config(path: Path) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv: Iterable[str] | None = None) -> None:
    from app.utils.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Build the Harry Potter vector store.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--force", action="store_true", help="Rebuild even if it exists.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    configure_logging(get_settings().log_level)
    config = build_vector_store(pdf_path=args.pdf, md_out=args.md_out, force=args.force)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    _cli()
