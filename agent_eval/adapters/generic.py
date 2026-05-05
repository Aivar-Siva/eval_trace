from contextlib import contextmanager
from ..core.tracer import BaseTracer


@contextmanager
def trace_session(version: str, task_id: str = ""):
    """
    Context manager for any framework.

    Usage:
        with trace_session(version="v1", task_id="task_001") as t:
            t.log_llm_start(prompt)
            t.log_llm_end(response, tool_selected="web_search")
            t.log_tool_start("web_search", {"query": "..."}, turn=1)
            result = my_tool(...)
            t.log_tool_end(result)
            t.log_final(answer, total_turns=1, success=True)
    """
    tracer = BaseTracer(version=version, task_id=task_id)
    yield tracer
