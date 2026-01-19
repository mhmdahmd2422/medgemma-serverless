import base64
import json
import uuid
from io import BytesIO
from PIL import Image
from typing import Generator, List, Dict, Any, Optional, Tuple

from src.inference import stream_generate

def _as_bool(v) -> bool:
    """Parse common truthy/falsey representations safely."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return False

# Top-level function that Runpod starts
def runpod_start():
    # Import runpod lazily so local dev/tests don't require Runpod's serverless deps.
    import runpod
    # configure serverless to aggregate streaming output into a list
    # This allows both streaming (via /stream endpoint) and sync (via /runsync) to work
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})

def load_image(input_data):
    """Load image from URL, base64 string, or data URL."""
    if not input_data:
        return None

    # Handle data URLs (e.g., "data:image/png;base64,...")
    if isinstance(input_data, str) and input_data.startswith("data:"):
        # Extract base64 part after the comma
        try:
            _, base64_data = input_data.split(",", 1)
            img_bytes = base64.b64decode(base64_data)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Invalid data URL format for image.")

    if isinstance(input_data, str) and (input_data.startswith("http://") or input_data.startswith("https://")):
        import requests
        return Image.open(BytesIO(requests.get(input_data).content)).convert("RGB")

    # assume base64
    try:
        img_bytes = base64.b64decode(input_data)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image input. Provide URL, data URL, or base64 string.")

def parse_openai_messages(messages: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    """
    Parse OpenAI-style messages array into a prompt string and optional image.

    Returns:
        Tuple of (combined_prompt, first_image_url_or_base64)
    """
    prompt_parts = []
    image_data = None

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        # Handle string content
        if isinstance(content, str):
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")

        # Handle array content (multimodal messages)
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")

                    if item_type == "text":
                        text_parts.append(item.get("text", ""))

                    elif item_type == "image_url":
                        # Extract image URL or data URL
                        image_url_obj = item.get("image_url", {})
                        if isinstance(image_url_obj, dict):
                            url = image_url_obj.get("url", "")
                        else:
                            url = str(image_url_obj)

                        if url and image_data is None:
                            image_data = url

                    elif item_type == "image_file":
                        # Handle base64 image file
                        file_data = item.get("image_file", {})
                        if isinstance(file_data, dict):
                            b64 = file_data.get("base64", "")
                            if b64 and image_data is None:
                                image_data = b64

            if text_parts:
                combined_text = " ".join(text_parts)
                if role == "system":
                    prompt_parts.append(f"System: {combined_text}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {combined_text}")
                else:
                    prompt_parts.append(f"User: {combined_text}")

    combined_prompt = "\n\n".join(prompt_parts)
    return combined_prompt, image_data

def stream_generate_sse(prompt: str, image=None, max_new_tokens: int = 256, model_id: str = "google/medgemma-4b-it") -> Generator[str, None, None]:
    """
    Wrap stream_generate to yield SSE-formatted responses compatible with OpenAI API.

    Yields SSE strings like:
        data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"token"}}]}
    """
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(__import__("time").time())

    prompt_tokens = len(prompt.split())  # Rough estimate
    completion_tokens = 0

    for token in stream_generate(prompt, image, max_new_tokens=max_new_tokens):
        completion_tokens += 1
        chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Send final chunk with finish_reason and usage
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

def handler(event):
    """
    Handle both legacy format and OpenAI-compatible format.

    For streaming: Uses yield from to make this handler a generator function.
    RunPod serverless with return_aggregate_stream=False will stream each yielded value.

    Legacy format:
        { "prompt": "...", "image": "<url|base64|null>", "max_new_tokens": 256, "stream": false }
        or { "input": { "prompt": "...", ... } }

    OpenAI-compatible format (from ConnectX API):
        {
            "input": {
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": "google/medgemma-4b-it",
                    "messages": [...],
                    "stream": true,
                    "max_tokens": 4096
                }
            }
        }
    """
    payload = event if isinstance(event, dict) else {}
    input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else {}

    # Check for OpenAI-compatible format
    openai_input = input_payload.get("openai_input") or payload.get("openai_input")

    if openai_input and isinstance(openai_input, dict):
        # OpenAI-compatible format
        messages = openai_input.get("messages", [])
        if not messages:
            yield {"error": "Missing 'messages' field in openai_input."}
            return

        prompt, image_data = parse_openai_messages(messages)
        if not prompt:
            yield {"error": "Could not extract prompt from messages."}
            return

        img = load_image(image_data) if image_data else None
        max_new_tokens = openai_input.get("max_tokens", 4096)
        stream = _as_bool(openai_input.get("stream", True))
        model_id = openai_input.get("model", "google/medgemma-4b-it")

        if stream:
            # Streaming: yield each SSE chunk directly
            yield from stream_generate_sse(prompt, img, max_new_tokens=max_new_tokens, model_id=model_id)
            return

        # Non-streaming: yield full OpenAI-compatible response
        text = "".join(stream_generate(prompt, img, max_new_tokens=max_new_tokens))
        prompt_tokens = len(prompt.split())
        completion_tokens = len(text.split())

        yield {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        return

    # Legacy format handling
    prompt = payload.get("prompt") or input_payload.get("prompt")
    if not prompt:
        yield {"error": "Missing 'prompt' field in event."}
        return

    image_data = payload.get("image") or input_payload.get("image")
    img = load_image(image_data) if image_data else None

    max_new_tokens = payload.get("max_new_tokens", input_payload.get("max_new_tokens", 256))

    # Support both sync and streaming
    stream = _as_bool(payload.get("stream", input_payload.get("stream", False)))
    if stream:
        # Streaming: yield each token directly
        yield from stream_generate(prompt, img, max_new_tokens=max_new_tokens)
        return

    # Non-streaming: yield full response
    text = "".join(stream_generate(prompt, img, max_new_tokens=max_new_tokens))
    yield {"output": text}
