FROM python:3.10-slim

# -------------------------
# System deps
# -------------------------
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl build-essential libgl1 libglib2.0-0 libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------
# Hugging Face authentication for gated models (build-time only)
# -------------------------
# We accept an HF_TOKEN at build time so the image can download
# gated models like google/medgemma-4b-it during the preload step.
ARG HF_TOKEN
ARG PRELOAD_MODEL=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Runpod pulls in aiohttp, which will attempt to import aiodns if present.
# Some environments hit an aiodns/pycares incompatibility at import time.
# We don't need aiodns for the worker, so remove it to avoid startup crashes.
RUN pip uninstall -y aiodns || true

# -------------------------
# Model preload layer (stable)
# -------------------------
# Copy only the minimal files needed to load the model so that changing
# other application code (e.g. serverless_handler, inference, etc.)
# does NOT invalidate the cached model download layer.
COPY src/__init__.py src/model.py ./src/

# Preload model artifacts into /models to minimize cold-start time.
# Note: By default we DO NOT bake weights into the image (PRELOAD_MODEL=0).
# To preload (bake) weights, build with:
#   --build-arg PRELOAD_MODEL=1 --build-arg HF_TOKEN=hf_...
RUN mkdir -p /models && \
    if [ "$PRELOAD_MODEL" = "1" ] && [ -n "$HF_TOKEN" ]; then \
      echo "Preloading model into /models (PRELOAD_MODEL=1)..."; \
      HF_TOKEN="$HF_TOKEN" HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
      python -c "from src.model import load_model; load_model(preload=True)"; \
    else \
      echo "Skipping model preload (PRELOAD_MODEL=0 or missing HF_TOKEN)."; \
    fi

# -------------------------
# Application code (changes often)
# -------------------------
# This COPY happens *after* the preload layer, so edits to code here
# do not force a re-download of the model weights.
COPY src ./src
COPY handler.py .

CMD ["python", "handler.py"]
