import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

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
    _processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    # MedGemma now uses an image-text-to-text architecture
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,  # `torch_dtype` is deprecated in newer transformers
        device_map="auto",
        cache_dir=cache_dir,
    )

    if preload:
        try:
            print("Warming up the model with a short token generation...")
            dummy_inputs = _processor(text="hello", images=None, return_tensors="pt")
            dummy_inputs = {k: v.to(_model.device) for k, v in dummy_inputs.items()}
            _model.generate(**dummy_inputs, max_new_tokens=2)
        except Exception as e:
            print("Warmup failed:", e)

    return _model, _processor
