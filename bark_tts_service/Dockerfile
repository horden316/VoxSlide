FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
EXPOSE 7862

# Each worker process loads its own model copy (~5GB VRAM in fp16 for the full
# checkpoint), so keep workers at 1 unless the GPU has room to spare.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 7862 --workers ${BARK_TTS_WORKERS:-1}"]
