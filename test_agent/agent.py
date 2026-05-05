"""
System Diagnosis Agent — Bedrock (via Lambda proxy) + system_diagnosis_mcp tools.

The agent is a simple ReAct loop:
  1. Send user query + tool descriptions to Bedrock LLM
  2. Parse tool call from response
  3. Execute the MCP tool
  4. Feed result back, repeat until final answer

All LLM calls and tool calls are traced via agent_eval.adapters.generic.
"""
import json
import re
import requests
from .system_diagnosis_mcp import MCP_TOOL_DEFINITIONS

BEDROCK_URL = None   # set at runtime from config


# ── Tool registry ──────────────────────────────────────────────────────────────

def _build_registry() -> dict:
    """Map tool name → callable from MCP_TOOL_DEFINITIONS."""
    return {t["name"]: t["function"] for t in MCP_TOOL_DEFINITIONS}


def _tool_descriptions() -> str:
    lines = []
    for t in MCP_TOOL_DEFINITIONS:
        params = t.get("parameters", {}).get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type','any')}" for k, v in params.items()
        ) if params else "none"
        lines.append(f"- {t['name']}({param_str}): {t['description']}")
    return "\n".join(lines)


_SYSTEM = f"""You are a Windows system diagnosis assistant.
You have access to these tools:

{_tool_descriptions()}

To call a tool, respond with EXACTLY this format (nothing else on that line):
TOOL_CALL: {{"tool": "<tool-name>", "args": {{...}}}}

When you have enough information to answer, respond with:
FINAL: <your answer>

Think step by step. Use the minimum tools needed."""


# ── Bedrock call ───────────────────────────────────────────────────────────────

def _llm(messages: list[dict], url: str) -> str:
    payload = {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "messages": messages,
        "max_tokens": 1024,
    }
    resp = requests.post(url, json=payload, timeout=60)
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
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content)
    return str(content)


# ── ReAct loop ─────────────────────────────────────────────────────────────────

def run(query: str, bedrock_url: str, tracer=None, max_turns: int = 8) -> str:
    """
    Run the agent. If tracer (BaseTracer) is provided, all LLM + tool calls are logged.
    Returns the final answer string.
    """
    registry = _build_registry()
    messages = [
        {"role": "user", "content": _SYSTEM + f"\n\nUser query: {query}"}
    ]
    turn = 0

    if tracer:
        tracer.log_llm_start(query)

    while turn < max_turns:
        response = _llm(messages, bedrock_url)
        messages.append({"role": "assistant", "content": response})

        # Check for FINAL answer
        final_match = re.search(r"FINAL:\s*(.+)", response, re.DOTALL)
        if final_match:
            answer = final_match.group(1).strip()
            if tracer:
                tracer.log_llm_end(response, tool_selected="")
                tracer.log_final(answer, total_turns=turn, success=True)
            return answer

        # Check for tool call
        tool_match = re.search(r"TOOL_CALL:\s*(\{.+?\})", response, re.DOTALL)
        if tool_match:
            try:
                call = json.loads(tool_match.group(1))
                tool_name = call.get("tool", "")
                args = call.get("args", {})
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "Invalid JSON in TOOL_CALL. Try again."})
                continue

            turn += 1
            tool_fn = registry.get(tool_name)

            if tracer:
                tracer.log_llm_end(response, tool_selected=tool_name)
                tracer.log_tool_start(tool_name, args, turn)

            if tool_fn is None:
                result = json.dumps({"error": f"Unknown tool: {tool_name}"})
            else:
                try:
                    result = tool_fn(**args) if args else tool_fn()
                except Exception as e:
                    result = json.dumps({"error": str(e)})

            if tracer:
                tracer.log_tool_end(result)

            messages.append({"role": "user", "content": f"Tool result:\n{result}"})
        else:
            # No tool call, no FINAL — treat as final answer
            if tracer:
                tracer.log_llm_end(response, tool_selected="")
                tracer.log_final(response, total_turns=turn, success=True)
            return response

    # Max turns reached
    if tracer:
        tracer.log_final("Max turns reached", total_turns=turn, success=False)
    return "Max turns reached without a final answer."
