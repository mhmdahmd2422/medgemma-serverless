        FROM python:3.10-slim

        # -------------------------
        # System deps
        # -------------------------
        ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

        RUN apt-get update && apt-get install -y --no-install-recommends \
            git wget curl build-essential libgl1 libglib2.0-0 libjpeg-dev \
            && rm -rf /var/lib/apt/lists/*

        WORKDIR /app

        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt

        # Copy source code
        COPY src ./src
        COPY handler.py .

        # Preload model artifacts into /models to minimize cold-start time.
        # Note: the loader will download the model during build time.
        RUN python - <<'PY'
from src.model import load_model
print('Preloading model into /models...')
load_model(preload=True)
PY

        CMD ["python", "handler.py"]
