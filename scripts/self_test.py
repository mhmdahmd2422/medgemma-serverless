"""
Fast local contract tests to build confidence before pushing a large MedGemma image.

Runs without downloading MedGemma. Uses your chosen MODEL_ID (e.g. Gemma 2B IT)
for a quick end-to-end generation smoke test, and uses mocks to exercise:
- serverless event parsing + return type (generator of strings)
- multimodal vs text-only preprocessing paths
- chat-template behavior (tokenize=False)

Usage (WSL):
  source .venv/bin/activate
  export MODEL_ID=google/gemma-2-2b-it
  export MODEL_CACHE_DIR=$HOME/.cache/huggingface
  python scripts/self_test.py
"""

from __future__ import annotations

import os
import sys
import base64
import types
from io import BytesIO
import time

# Ensure repo root is on sys.path so `import src...` works when running as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def test_serverless_handler_contract():
    import src.serverless_handler as sh
    import json

    # Monkeypatch stream_generate so we don't need a model for this test.
    def fake_stream_generate(prompt: str, image=None, max_new_tokens: int = 256):
        yield "hello"
        yield " world"

    sh.stream_generate = fake_stream_generate  # type: ignore[attr-defined]

    # Missing prompt -> error dict
    out = sh.handler({})
    _assert(isinstance(out, dict) and "error" in out, "Expected error dict for missing prompt")

    # prompt at top-level (legacy format - non-streaming by default)
    out = sh.handler({"prompt": "x", "max_new_tokens": 5})
    _assert(isinstance(out, dict) and "output" in out, "Expected dict with output for non-streaming legacy format")
    _assert(out["output"] == "hello world", "Expected output to be 'hello world'")

    # prompt at top-level with stream=true (legacy format)
    out = sh.handler({"prompt": "x", "max_new_tokens": 5, "stream": True})
    _assert(isinstance(out, types.GeneratorType), "Expected generator return for streaming legacy format")
    _assert("".join(list(out)) == "hello world", "Expected streaming strings to join")

    # prompt under input (legacy format - non-streaming by default)
    out = sh.handler({"input": {"prompt": "x"}})
    _assert(isinstance(out, dict) and "output" in out, "Expected dict with output for input.prompt")

    # prompt under input with stream=true
    out = sh.handler({"input": {"prompt": "x", "stream": True}})
    _assert(isinstance(out, types.GeneratorType), "Expected generator return for input.prompt with stream")

    # base64 image should parse (we don't validate image content deeply here)
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    out = sh.handler({"prompt": "x", "image": b64, "stream": True})
    _assert(isinstance(out, types.GeneratorType), "Expected generator with base64 image")


def test_openai_compatible_format():
    """Test OpenAI-compatible format from ConnectX API."""
    import src.serverless_handler as sh
    import json

    # Monkeypatch stream_generate so we don't need a model for this test.
    def fake_stream_generate(prompt: str, image=None, max_new_tokens: int = 256):
        yield "hello"
        yield " world"

    sh.stream_generate = fake_stream_generate  # type: ignore[attr-defined]

    # OpenAI-compatible format with streaming
    openai_event = {
        "input": {
            "openai_route": "/v1/chat/completions",
            "openai_input": {
                "model": "google/medgemma-4b-it",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "You are a medical assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": "What is diabetes?"}]}
                ],
                "stream": True,
                "max_tokens": 4096
            }
        }
    }

    out = sh.handler(openai_event)
    _assert(isinstance(out, types.GeneratorType), "Expected generator for OpenAI streaming format")

    # Collect SSE chunks and verify format
    chunks = list(out)
    _assert(len(chunks) > 0, "Expected at least one SSE chunk")

    # First chunks should be data: {...}\n\n format
    for chunk in chunks[:-1]:  # Exclude [DONE]
        if chunk.strip() == "data: [DONE]":
            continue
        _assert(chunk.startswith("data: "), f"Expected SSE format, got: {chunk[:50]}")
        json_str = chunk[6:].strip()  # Remove "data: " prefix
        if json_str and json_str != "[DONE]":
            data = json.loads(json_str)
            _assert("choices" in data, "Expected 'choices' in SSE chunk")
            _assert("id" in data, "Expected 'id' in SSE chunk")

    # Last chunk should be [DONE]
    _assert(chunks[-1].strip() == "data: [DONE]", "Expected final [DONE] marker")

    # OpenAI-compatible format with images (data URL)
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    openai_with_image = {
        "input": {
            "openai_input": {
                "model": "google/medgemma-4b-it",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image."},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                "stream": True,
                "max_tokens": 256
            }
        }
    }

    out = sh.handler(openai_with_image)
    _assert(isinstance(out, types.GeneratorType), "Expected generator for OpenAI format with image")
    chunks = list(out)
    _assert(len(chunks) > 0, "Expected output with image")

    # Non-streaming OpenAI format
    openai_non_stream = {
        "input": {
            "openai_input": {
                "model": "google/medgemma-4b-it",
                "messages": [
                    {"role": "user", "content": "What is the flu?"}
                ],
                "stream": False,
                "max_tokens": 256
            }
        }
    }

    out = sh.handler(openai_non_stream)
    _assert(isinstance(out, dict), "Expected dict for non-streaming OpenAI format")
    _assert("choices" in out, "Expected 'choices' in response")
    _assert(out["choices"][0]["message"]["content"] == "hello world", "Expected content in message")
    _assert("usage" in out, "Expected 'usage' in response")

    # Missing messages -> error
    out = sh.handler({"input": {"openai_input": {"model": "test"}}})
    _assert(isinstance(out, dict) and "error" in out, "Expected error for missing messages")


def test_inference_preprocess_contracts():
    """
    Tests preprocessing logic without loading a real model by monkeypatching:
    - load_model() to return fakes
    - TextIteratorStreamer to a fake streamer that yields fixed chunks
    """
    import src.inference as inf

    class FakeTensor:
        def to(self, device):
            return self

    class FakeStreamer:
        def __init__(self, tokenizer, skip_prompt: bool, skip_special_tokens: bool):
            self._chunks = ["ok"]

        def __iter__(self):
            return iter(self._chunks)

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            # no-op; FakeStreamer yields independently
            return None

    class FakeProcessorMM:
        # Multimodal-like processor with apply_chat_template
        tokenizer = object()

        def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool = False):
            _assert(tokenize is False, "Expected tokenize=False in chat templating")
            return "CHAT"

        def __call__(self, text=None, images=None, return_tensors=None):
            _assert(isinstance(text, str), "Expected text to be a string")
            return {"input_ids": FakeTensor()}

    class FakeTokenizerText:
        def apply_chat_template(self, messages, add_generation_prompt: bool, tokenize: bool = False):
            _assert(tokenize is False, "Expected tokenize=False in chat templating")
            return "CHAT"

        def __call__(self, text, return_tensors=None):
            _assert(isinstance(text, str), "Expected text to be a string")
            return {"input_ids": FakeTensor()}

    # Patch streamer + threading to avoid background execution complexity.
    original_streamer = inf.TextIteratorStreamer
    inf.TextIteratorStreamer = FakeStreamer  # type: ignore[assignment]

    original_thread = inf.threading.Thread

    class FakeThread:
        def __init__(self, target=None, kwargs=None):
            self._alive = False
            self._target = target
            self._kwargs = kwargs or {}

        def start(self):
            self._alive = False
            if self._target:
                self._target(**self._kwargs)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            return None

    inf.threading.Thread = FakeThread  # type: ignore[assignment]

    try:
        # Multimodal processor path (supports images kw)
        inf.load_model = lambda: (FakeModel(), FakeProcessorMM())  # type: ignore[assignment]
        out = "".join(list(inf.stream_generate("hi", image=None, max_new_tokens=4)))
        _assert(out == "ok", "Expected fake streamer output")

        # Text-only tokenizer path (no images kw)
        inf.load_model = lambda: (FakeModel(), FakeTokenizerText())  # type: ignore[assignment]
        out = "".join(list(inf.stream_generate("hi", image=None, max_new_tokens=4)))
        _assert(out == "ok", "Expected fake streamer output (text-only)")
    finally:
        inf.TextIteratorStreamer = original_streamer
        inf.threading.Thread = original_thread


def test_real_model_smoke():
    """
    Minimal real-model smoke using current MODEL_ID.
    This is meant to run with a small dev model like google/gemma-2-2b-it.
    """
    import os
    from src.inference import stream_generate

    if os.environ.get("SKIP_REAL_MODEL_SMOKE") == "1":
        print("[SKIP] real-model smoke test (SKIP_REAL_MODEL_SMOKE=1)")
        return

    model_id = os.environ.get("MODEL_ID", "<unset>")
    print(f"[INFO] real-model smoke: MODEL_ID={model_id!r} (this may download/load on first run)")

    start = time.time()
    max_seconds = float(os.environ.get("REAL_MODEL_SMOKE_MAX_SECONDS", "120"))
    max_chars = int(os.environ.get("REAL_MODEL_SMOKE_MAX_CHARS", "400"))

    chunks = []
    char_count = 0
    for chunk in stream_generate("What are common symptoms of the flu?", None, 32):
        chunks.append(chunk)
        char_count += len(chunk)
        if char_count >= max_chars:
            break
        if time.time() - start > max_seconds:
            break

    text = "".join(chunks)
    elapsed = time.time() - start
    print(f"[INFO] real-model smoke: got {len(text)} chars in {elapsed:.1f}s")

    _assert(isinstance(text, str), "Expected output string")
    _assert(len(text.strip()) > 0, "Expected non-empty output from real-model smoke test")


def main():
    test_serverless_handler_contract()
    print("[OK] serverless handler contract (legacy format)")

    test_openai_compatible_format()
    print("[OK] OpenAI-compatible format (SSE streaming)")

    test_inference_preprocess_contracts()
    print("[OK] inference preprocess contracts (mocked)")

    test_real_model_smoke()
    print("[OK] real-model smoke test (current MODEL_ID)")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()


