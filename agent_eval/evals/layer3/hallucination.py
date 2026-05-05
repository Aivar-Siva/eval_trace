from ..judge import judge_score


def score(spans: list[dict], label: dict) -> float:
    """
    LLM-as-judge: does the final output contain claims not supported by any tool output?
    Returns faithfulness score 0.0–1.0 (1.0 = fully grounded, no hallucination).
    """
    final_output = ""
    tool_outputs = []
    for span in spans:
        if span.get("final", {}).get("output"):
            final_output = span["final"]["output"]
        op = span.get("operational") or {}
        if op.get("tool_output"):
            tool_outputs.append(op["tool_output"])

    if not final_output:
        return 1.0
    context = "\n---\n".join(tool_outputs[:5]) or "(no tool outputs)"
    prompt = f"""You are a faithfulness judge.

Tool outputs (ground truth context):
{context[:1500]}

Agent's final answer:
{final_output[:800]}

Does the final answer contain any claims NOT supported by the tool outputs above?
Reply with only a single integer from 1 (heavily hallucinated) to 5 (fully grounded)."""
    return judge_score(prompt, low=1, high=5)
