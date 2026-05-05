"""Shared Bedrock judge client for all LLM-as-judge evals."""
import json
import requests
from ..config import BEDROCK_URL, EVAL_MODEL


def judge(prompt: str) -> str:
    """Call Bedrock proxy with llama3-3-70b, return text response."""
    payload = {
        "model_id": EVAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    resp = requests.post(BEDROCK_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Handle both streaming text and message formats
    if isinstance(data, list):
        return "".join(c.get("text", "") for item in data for c in (item.get("content") or []) if isinstance(c, dict))
    if "content" in data:
        content = data["content"]
        if isinstance(content, list):
            return "".join(c.get("text", "") for c in content)
        return str(content)
    return str(data)


def judge_score(prompt: str, low: float = 0.0, high: float = 1.0) -> float:
    """Ask judge for a numeric score, parse first float found in response."""
    import re
    text = judge(prompt)
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for m in matches:
        val = float(m)
        if low <= val <= high:
            return val
    # fallback: if score asked 1-5, normalise
    for m in matches:
        val = float(m)
        if 1 <= val <= 5:
            return (val - 1) / 4
    return 0.5
