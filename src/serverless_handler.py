import base64
from io import BytesIO
from PIL import Image
from typing import Generator

from src.inference import stream_generate

# Top-level function that Runpod starts
def runpod_start():
    # Import runpod lazily so local dev/tests don't require Runpod's serverless deps.
    import runpod
    # configure serverless to return streaming generator
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": False})

def load_image(input_data):
    if not input_data:
        return None
    if isinstance(input_data, str) and (input_data.startswith("http://") or input_data.startswith("https://")):
        import requests
        return Image.open(BytesIO(requests.get(input_data).content)).convert("RGB")
    # assume base64
    try:
        img_bytes = base64.b64decode(input_data)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image input. Provide URL or base64 string.")

def handler(event):
    # event expected to be { "prompt": "...", "image": "<url|base64|null>", "max_new_tokens": 256 }
    prompt = None
    if isinstance(event, dict):
        prompt = event.get("prompt") or (event.get("input") and event["input"].get("prompt"))
    if not prompt:
        return {"error": "Missing 'prompt' field in event."}

    image_data = None
    if isinstance(event, dict):
        image_data = event.get("image")
    img = load_image(image_data) if image_data else None

    # Return a generator that yields stream chunks (strings)
    return stream_generate(prompt, img, max_new_tokens=event.get("max_new_tokens", 256))
