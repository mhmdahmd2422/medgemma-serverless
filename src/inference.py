import threading
from transformers import TextIteratorStreamer
from src.model import load_model
from typing import Generator

def stream_generate(prompt: str, image=None, max_new_tokens: int = 256) -> Generator[str, None, None]:
    """Yield token-by-token strings from the model.

    This generator yields text chunks (already decoded) as they arrive.
    """
    model, processor = load_model()

    # Wrap prompt into a chat template when available.
    # - For MedGemma/Gemma3 processors: use multimodal message format with image placeholder.
    # - For text-only tokenizers: use simple chat template if supported, else raw prompt.
    chat = prompt
    if hasattr(processor, "apply_chat_template"):
        try:
            # Multimodal processor chat format
            # When an image is provided, include {"type": "image"} placeholder in content
            if image is not None:
                content = [
                    {"type": "image"},  # Image placeholder for the processor
                    {"type": "text", "text": prompt}
                ]
            else:
                content = [{"type": "text", "text": prompt}]

            messages = [{"role": "user", "content": content}]
            try:
                chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            except TypeError:
                # Some processors don't expose tokenize=...; they already return a string.
                chat = processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            try:
                # Some tokenizers accept list-of-dicts with string content
                messages = [{"role": "user", "content": prompt}]
                try:
                    chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                except TypeError:
                    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
            except Exception:
                chat = prompt

    # Prepare inputs (text + optional image) for the model
    try:
        inputs = processor(text=chat, images=image, return_tensors="pt")
    except (TypeError, ValueError):
        # Text-only models/tokenizers do not accept `images=...`
        inputs = processor(chat if isinstance(chat, str) else prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Create streamer attached to the tokenizer
    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        # For this chatty vision-language model, sampling gives more useful answers.
        do_sample=True,
        temperature=0.7,
        # Add top_p and top_k for numerical stability to prevent
        # "probability tensor contains inf, nan or element < 0" CUDA errors
        top_p=0.9,
        top_k=50,
    )

    # Run generation in background thread so we can yield from streamer
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()

    try:
        for chunk in streamer:
            # chunk is a string piece of newly generated text
            yield chunk
    finally:
        # If thread still alive, we join (non-blocking small timeout)
        if thread.is_alive():
            try:
                thread.join(timeout=0.1)
            except Exception:
                pass
