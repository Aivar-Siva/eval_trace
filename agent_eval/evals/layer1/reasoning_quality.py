from ..judge import judge_score


def score(span: dict, label: dict) -> float:
    """LLM-as-judge: rate reasoning quality 0.0–1.0."""
    cog = span.get("cognitive") or {}
    task  = cog.get("task_query", "")
    reasoning = cog.get("reasoning_text", "")
    if not reasoning:
        return 0.0
    prompt = f"""You are an evaluation judge. Rate the quality of the agent's reasoning on a scale of 1 to 5.

Task: {task}
Agent reasoning: {reasoning}

Criteria:
- Did the reasoning correctly understand the task?
- Was the approach logical and sound?
- Did it avoid unnecessary steps?

Reply with only a single integer score from 1 (very poor) to 5 (excellent)."""
    return judge_score(prompt, low=1, high=5)
