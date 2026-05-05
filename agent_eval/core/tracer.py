import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from .schema import TraceSpan
from .db import insert_span

JSONL_PATH = Path("data/traces.jsonl")


class BaseTracer:
    def __init__(self, version: str, task_id: str = ""):
        self.version = version
        self.task_id = task_id
        self.trace_id = str(uuid.uuid4())
        self._pending: dict = {}

    def log_llm_start(self, prompt: str):
        self._pending["task_query"] = prompt
        self._pending["timestamp_start"] = _now()

    def log_llm_end(self, response: str, tool_selected: str = "", intent_label: str = ""):
        self._pending["reasoning_text"] = response
        self._pending["tool_selected"] = tool_selected
        self._pending["intent_label"] = intent_label

    def log_tool_start(self, tool_name: str, tool_input: dict, turn: int):
        self._pending["tool_name"] = tool_name
        self._pending["tool_input"] = tool_input
        self._pending["turn_number"] = turn

    def log_tool_end(self, tool_output: str, next_action: str = ""):
        self._pending["tool_output"] = tool_output
        self._pending["next_action"] = next_action
        self._pending["timestamp_end"] = _now()
        self._flush()

    def log_final(self, output: str, total_turns: int, success: bool, total_tokens: int = 0):
        span: TraceSpan = {
            "trace_id": self.trace_id,
            "agent_version": self.version,
            "task_id": self.task_id,
            "timestamp_start": _now(),
            "timestamp_end": _now(),
            "final": {
                "output": output,
                "success": success,
                "total_turns": total_turns,
                "total_tokens": total_tokens,
            },
        }
        _write(span)

    def _flush(self):
        p = self._pending
        span: TraceSpan = {
            "trace_id": self.trace_id,
            "agent_version": self.version,
            "task_id": self.task_id,
            "timestamp_start": p.pop("timestamp_start", _now()),
            "timestamp_end": p.pop("timestamp_end", _now()),
            "cognitive": {
                "task_query": p.pop("task_query", ""),
                "reasoning_text": p.pop("reasoning_text", ""),
                "tool_selected": p.pop("tool_selected", ""),
                "intent_label": p.pop("intent_label", ""),
            },
            "operational": {
                "tool_name": p.pop("tool_name", ""),
                "tool_input": p.pop("tool_input", {}),
                "tool_output": p.pop("tool_output", ""),
                "next_action": p.pop("next_action", ""),
                "turn_number": p.pop("turn_number", 0),
            },
        }
        self._pending = {}
        _write(span)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(span: TraceSpan):
    insert_span(span)
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(span) + "\n")
