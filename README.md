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

## Notes
- The Dockerfile preloads the model into `/models` during image build to minimize cold starts.
- For production, prefer hosting model artifacts in a private model store or bake them into the image.
- If you want OpenAI-compatible streaming JSON (delta messages), see `src/serverless_handler.py` for where to adapt the generator format.
