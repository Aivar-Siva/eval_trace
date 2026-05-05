from typing import TypedDict, Optional


class CognitiveSpan(TypedDict, total=False):
    task_query: str
    reasoning_text: str
    tool_selected: str
    intent_label: str


class OperationalSpan(TypedDict, total=False):
    tool_name: str
    tool_input: dict
    tool_output: str
    tool_output_tokens: int
    next_action: str
    turn_number: int


class ContextualSpan(TypedDict, total=False):
    context_tokens_before: int
    context_tokens_after: int
    node_name: str
    run_id: str


class FinalSpan(TypedDict, total=False):
    output: str
    success: bool
    total_turns: int
    total_tokens: int


class TraceSpan(TypedDict, total=False):
    trace_id: str
    agent_version: str
    task_id: str
    timestamp_start: str
    timestamp_end: str
    cognitive: CognitiveSpan
    operational: OperationalSpan
    contextual: ContextualSpan
    final: FinalSpan
