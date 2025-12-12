import os
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
)

# Default model ID (MedGemma 4B Italian variant)
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-4b-it")
_model = None
_processor = None

def load_model(preload=False):
    """Load and cache the model and processor.

    Args:
        preload (bool): If True, run a tiny warmup generation so the weights are loaded to GPU.

    Returns:
        (model, processor)
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    # Cache directory
    cache_dir = "/models"
    os.environ.setdefault("HF_HOME", cache_dir)

    print(f"Loading model {MODEL_ID} (this happens once at cold start / build time)...")

    # Some models (e.g. Gemma2 text-only) don't have an AutoProcessor. Fall back to AutoTokenizer.
    try:
        _processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    except Exception:
        _processor = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir)

    # Choose dtype / device map depending on whether a GPU is available.
    # fp16 on CPU can easily produce NaNs, so we stick to fp32 when no CUDA.
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"

    # Try multimodal first (MedGemma/Gemma3). If the config is text-only (Gemma2),
    # fall back to a causal LM.
    try:
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=dtype,  # `torch_dtype` is deprecated in newer transformers
            device_map=device_map,
            cache_dir=cache_dir,
        )
    except ValueError:
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )

    if preload:
        try:
            print("Warming up the model with a short token generation...")
            # Tokenizers don't accept `images=...`; processors do. Try images, fall back to text-only.
            try:
                dummy_inputs = _processor(text="hello", images=None, return_tensors="pt")
            except TypeError:
                dummy_inputs = _processor("hello", return_tensors="pt")
            dummy_inputs = {k: v.to(_model.device) for k, v in dummy_inputs.items()}
            _model.generate(**dummy_inputs, max_new_tokens=2)
        except Exception as e:
            print("Warmup failed:", e)

    return _model, _processor
