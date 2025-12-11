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
# Hugging Face authentication for gated models
# -------------------------
# We accept an HF_TOKEN at build time so the image can download
# gated models like google/medgemma-4b-it during the preload step.
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Model preload layer (stable)
# -------------------------
# Copy only the minimal files needed to load the model so that changing
# other application code (e.g. serverless_handler, inference, etc.)
# does NOT invalidate the cached model download layer.
COPY src/__init__.py src/model.py ./src/

# Preload model artifacts into /models to minimize cold-start time.
# Note: the loader will download the model during build time.
RUN python - <<'PY'
from src.model import load_model
print("Preloading model into /models...")
load_model(preload=True)
PY

# -------------------------
# Application code (changes often)
# -------------------------
# This COPY happens *after* the preload layer, so edits to code here
# do not force a re-download of the model weights.
COPY src ./src
COPY handler.py .

CMD ["python", "handler.py"]
