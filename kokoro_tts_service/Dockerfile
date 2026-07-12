FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache

# espeak-ng backs misaki's G2P fallback for out-of-dictionary words.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 espeak-ng curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
EXPOSE 7861

# Each worker process loads its own model copy; at 82M that is cheap, and
# parallel workers are what let a whole deck synthesize concurrently.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 7861 --workers ${KOKORO_TTS_WORKERS:-1}"]
