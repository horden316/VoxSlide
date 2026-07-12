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
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
EXPOSE 7861

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7861"]
