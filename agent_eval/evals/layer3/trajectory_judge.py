"""
Agent-as-Judge: iterates over trace spans turn-by-turn and scores the full trajectory.
Implements Agent-as-a-Judge (arXiv 2410.10934) using Bedrock via Lambda proxy.
No LangGraph, no Groq — pure Bedrock ReAct loop.
"""
import json
import re
import requests
from ...core.db import query_by_trace_id
from ...config import BEDROCK_URL, EVAL_MODEL


def _llm(messages: list[dict]) -> str:
    payload = {"model_id": EVAL_MODEL, "messages": messages, "max_tokens": 1024}
    resp = requests.post(BEDROCK_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return "".join(
            c.get("text", "")
            for item in data
            for c in (item.get("content") or [])
            if isinstance(c, dict)
        )
    content = data.get("content", "")
    return "".join(c.get("text", "") for c in content) if isinstance(content, list) else str(content)


def score(spans: list[dict], label: dict) -> float:
    """
    Evaluator agent reads each span turn-by-turn, then scores 0.0–1.0.
    Uses a ReAct loop: INSPECT_TURN: <n> to request a span, SCORE: <float> to finish.
    """
    if not spans:
        return 0.0

    # Build a lookup by turn number
    turn_map: dict[int, dict] = {}
    for s in spans:
        t = (s.get("operational") or {}).get("turn_number")
        if t is not None:
            turn_map[t] = s

    total_turns = max(turn_map.keys(), default=0)
    task_input = label.get("input", "")

    system = f"""You are a trajectory evaluation judge.
Task: {task_input}
Total turns: {total_turns}

To inspect a turn, respond with: INSPECT_TURN: <number>
When done, respond with: SCORE: <0.0-1.0> REASON: <brief justification>

Evaluate: goal completion, reasoning coherence, tool use efficiency."""

    messages = [{"role": "user", "content": system}]
    observations: list[str] = []

    for _ in range(total_turns + 2):  # max iterations = turns + buffer
        response = _llm(messages)
        messages.append({"role": "assistant", "content": response})

        # Check for final score
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response)
        if score_match:
            return min(1.0, max(0.0, float(score_match.group(1))))

        # Check for turn inspection request
        turn_match = re.search(r"INSPECT_TURN:\s*(\d+)", response)
        if turn_match:
            turn_n = int(turn_match.group(1))
            span = turn_map.get(turn_n)
            if span:
                content = json.dumps({
                    "cognitive":   span.get("cognitive", {}),
                    "operational": span.get("operational", {}),
                })
            else:
                content = f"No span found for turn {turn_n}"
            messages.append({"role": "user", "content": f"Turn {turn_n} data:\n{content}"})
        else:
            # No instruction — prompt to conclude
            messages.append({"role": "user", "content": "Please provide your final SCORE: <0.0-1.0>"})

    return 0.5  # fallback if loop exhausted
