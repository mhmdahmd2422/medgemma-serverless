import threading
from transformers import TextIteratorStreamer
from src.model import load_model
from typing import Generator

def stream_generate(prompt: str, image=None, max_new_tokens: int = 256) -> Generator[str, None, None]:
    """Yield token-by-token strings from the model.

    This generator yields text chunks (already decoded) as they arrive.
    """
    model, processor = load_model()

    # Wrap the plain prompt into the chat template expected by MedGemma/Gemma3.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Prepare inputs (text + optional image) for the model
    inputs = processor(text=chat, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Create streamer attached to the tokenizer
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        # For this chatty vision-language model, sampling gives more useful answers.
        do_sample=True,
        temperature=0.7,
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
