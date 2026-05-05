"""
Orchestrates all scorers on a trace. Returns a flat scores dict.

Usage:
    from agent_eval.evals.runner import run_evals
    scores = run_evals(trace_id="...", label={...})
"""
from ..core.db import query_by_trace_id
from .layer1 import intent_correctness, tool_selection, reasoning_quality
from .layer2 import param_schema_check, param_value_check, result_interpretation
from .layer3 import task_success, efficiency, hallucination, trajectory_judge


def run_evals(trace_id: str, label: dict) -> dict[str, float]:
    spans = query_by_trace_id(trace_id)
    if not spans:
        return {}

    # Pick the first span with operational data for per-turn evals
    op_span  = next((s for s in spans if s.get("operational", {}).get("tool_name")), spans[0])
    # Pick the span with final data
    fin_span = next((s for s in spans if s.get("final", {}).get("output")), spans[-1])

    scores: dict[str, float] = {}

    # Layer 1
    scores["intent_correctness"]    = intent_correctness.score(op_span, label)
    scores["tool_selection"]        = tool_selection.score(op_span, label)
    scores["reasoning_quality"]     = reasoning_quality.score(op_span, label)

    # Layer 2
    scores["param_schema_check"]    = param_schema_check.score(op_span, label)
    scores["param_value_check"]     = param_value_check.score(op_span, label)
    scores["result_interpretation"] = result_interpretation.score(op_span, label)

    # Layer 3
    scores["task_success"]          = task_success.score(fin_span, label)
    scores["efficiency"]            = efficiency.score(fin_span, label)
    scores["hallucination"]         = hallucination.score(spans, label)
    scores["trajectory_quality"]    = trajectory_judge.score(spans, label)

    return scores
