# MedGemma Serverless Endpoint (Runpod)

This repository contains a production-ready serverless endpoint for **MedGemma-4B-IT**
that you can deploy to Runpod Serverless (GPU). It includes:

- Fast cold-start optimized `Dockerfile` that preloads model weights.
- Streaming output support (token-by-token) via Runpod serverless streaming.
- Modular Python package (`src/`) with `model.py`, `inference.py` and `serverless_handler.py`.
- Example client script to call the endpoint using the Runpod SDK.

**Important**: adjust `MODEL_ID` or cache behavior if you prefer a different MedGemma variant.

## Repo layout
```
medgemma-serverless-repo/
├── Dockerfile
├── handler.py
├── requirements.txt
├── README.md
├── .dockerignore
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── inference.py
│   └── serverless_handler.py
└── examples/
    └── client_stream.py
```

## Quick deploy (build + push)
1. Build:
   ```bash
   docker build -t <your-registry>/medgemma-serverless:latest .
   ```
2. Push:
   ```bash
   docker push <your-registry>/medgemma-serverless:latest
   ```
3. Create Runpod Serverless endpoint:
   - Use "Docker image" runtime and point to the pushed image.
   - Entrypoint: `python handler.py`
   - Enable streaming if desired (the handler is already configured).

## Runpod "Build from GitHub" vs "Docker image" (important for gated models)

`google/medgemma-4b-it` is gated on Hugging Face and requires an HF token to download.

- **If you build the image yourself (local/EC2)**:
  - By default, the image is built **without baking model weights**.
  - If you want to preload (bake) weights into `/models` to minimize cold starts, pass both `HF_TOKEN` and `PRELOAD_MODEL=1`:
    ```bash
    docker build --platform linux/amd64 --build-arg PRELOAD_MODEL=1 --build-arg HF_TOKEN=hf_... -t <your-registry>/medgemma-serverless:latest .
    ```
  - This minimizes cold starts.

- **If you use Runpod "Build from GitHub"**:
  - The build environment may not provide `HF_TOKEN`, so the Dockerfile will **skip preloading**.
  - In this mode you must set a Secret on the endpoint:
    - `HF_TOKEN` (or `HUGGINGFACE_HUB_TOKEN`)
  - The model will download on the first cold start and cache under `/models`.

## Notes
- The Dockerfile builds a **small image by default** (no model weights baked in). It can optionally preload the model into `/models` during build when `PRELOAD_MODEL=1` and `HF_TOKEN` are provided.
- For production, prefer hosting model artifacts in a private model store or bake them into the image.
- If you want OpenAI-compatible streaming JSON (delta messages), see `src/serverless_handler.py` for where to adapt the generator format.
