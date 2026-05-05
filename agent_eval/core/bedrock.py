"""Shared Bedrock Lambda proxy client — handles SSE streaming response."""
import json
import requests
from ..config import BEDROCK_URL, EVAL_MODEL


def bedrock_call(prompt: str, model_id: str = None, max_gen_len: int = 512) -> str:
    """Call Bedrock proxy, parse SSE stream, return full generated text."""
    payload = {
        "model_id": model_id or EVAL_MODEL,
        "prompt": prompt,
        "max_gen_len": max_gen_len,
    }
    resp = requests.post(BEDROCK_URL, json=payload, timeout=60, stream=True)
    resp.raise_for_status()

    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8") if isinstance(line, bytes) else line
        if decoded.startswith("data: "):
            try:
                chunk = json.loads(decoded[6:])
                text += chunk.get("generation", "")
            except json.JSONDecodeError:
                pass
    return text.strip()
