from ..judge import judge_score


def score(span: dict, label: dict) -> float:
    """LLM-as-judge: was the next action appropriate given the tool output?"""
    op = span.get("operational") or {}
    tool_output = op.get("tool_output", "")
    next_action = op.get("next_action", "")
    if not tool_output:
        return 0.0
    prompt = f"""You are an evaluation judge.

Tool output: {tool_output[:800]}
Agent's next action: {next_action or '(none recorded)'}

Was the next action a reasonable response to this tool output?
Reply with only a single integer from 1 (completely wrong) to 5 (perfectly appropriate)."""
    return judge_score(prompt, low=1, high=5)
