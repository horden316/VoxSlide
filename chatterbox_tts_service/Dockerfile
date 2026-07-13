FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    XDG_CACHE_HOME=/models/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
EXPOSE 7863

# The 350M turbo checkpoint loads at ~3GB VRAM in fp16 per worker, so keep
# workers at 1 unless the GPU has room to spare.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port 7863 --workers ${CHATTERBOX_TTS_WORKERS:-1}"]
