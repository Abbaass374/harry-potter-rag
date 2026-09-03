# Backend container image.
#
# NOTE: The vector store (backend/data/vector_store) must be built first (via the
# notebook or `python -m app.services.ingest --force`) and is copied in at build
# time. Ollama is expected to run on the host; point the container at it with
# OLLAMA_HOST (e.g. http://host.docker.internal:11434).

FROM python:3.12-slim

# Avoid interactive prompts, keep Python output unbuffered.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps occasionally needed by scientific wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code and the prebuilt vector store.
COPY app ./app
COPY data ./data

EXPOSE 8000

# Default: point at Ollama on the host machine.
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
