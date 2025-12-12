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

# Ensure repo root is on sys.path so `import src...` works when running as a script.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def test_serverless_handler_contract():
    import src.serverless_handler as sh

    # Monkeypatch stream_generate so we don't need a model for this test.
    def fake_stream_generate(prompt: str, image=None, max_new_tokens: int = 256):
        yield "hello"
        yield " world"

    sh.stream_generate = fake_stream_generate  # type: ignore[attr-defined]

    # Missing prompt -> error dict
    out = sh.handler({})
    _assert(isinstance(out, dict) and "error" in out, "Expected error dict for missing prompt")

    # prompt at top-level
    out = sh.handler({"prompt": "x", "max_new_tokens": 5})
    _assert(isinstance(out, types.GeneratorType), "Expected generator return for prompt")
    _assert("".join(list(out)) == "hello world", "Expected streaming strings to join")

    # prompt under input
    out = sh.handler({"input": {"prompt": "x"}})
    _assert(isinstance(out, types.GeneratorType), "Expected generator return for input.prompt")

    # base64 image should parse (we don't validate image content deeply here)
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    out = sh.handler({"prompt": "x", "image": b64})
    _assert(isinstance(out, types.GeneratorType), "Expected generator with base64 image")


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
    from src.inference import stream_generate

    text = "".join(stream_generate("What are common symptoms of the flu?", None, 64))
    _assert(isinstance(text, str), "Expected output string")
    _assert(len(text.strip()) > 0, "Expected non-empty output")


def main():
    test_serverless_handler_contract()
    print("[OK] serverless handler contract")

    test_inference_preprocess_contracts()
    print("[OK] inference preprocess contracts (mocked)")

    test_real_model_smoke()
    print("[OK] real-model smoke test (current MODEL_ID)")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()


