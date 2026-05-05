# Agent Evaluation Framework

A framework-agnostic evaluation SDK for AI agents that measures quality at every stage of execution — not just the final output.

Applied to a real Windows system diagnosis agent built on 21 MCP tools.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env          # BEDROCK_URL is pre-filled, nothing else required

python main.py                # run agent v1 + v2, eval all traces, print report + gate
uvicorn agent_eval.server.app:app --reload --port 8000   # dashboard → http://localhost:8000
```

---

## Architecture

```
test_agent/                        # Agent under test (outside the SDK)
├── agent.py                       # Bedrock ReAct loop, 21 MCP tools
└── system_diagnosis_mcp/          # Pulled from github.com/Precision-Recall/Echo
    └── granular_diagnostic_tools.py

agent_eval/                        # Reusable evaluation SDK
├── core/
│   ├── schema.py                  # TraceSpan TypedDict (cognitive / operational / final)
│   ├── db.py                      # SQLite helpers — structured, queryable traces
│   └── tracer.py                  # BaseTracer — framework-agnostic span collector
├── adapters/
│   ├── langchain.py               # LangChain/LangGraph — BaseCallbackHandler
│   ├── generic.py                 # Any custom agent — context manager
│   └── bedrock.py                 # Bedrock Agent Core — boto3 wrapper
├── evals/
│   ├── judge.py                   # Shared Bedrock LLM-as-judge client
│   ├── layer1/                    # Pre-tool-call evals
│   ├── layer2/                    # At/after tool call evals
│   ├── layer3/                    # End-to-end evals
│   └── runner.py                  # Orchestrates all 10 scorers on one trace
├── dataset/                       # Versioned eval dataset (SQLite)
├── scoring/                       # Score history + deployment gate
└── server/                        # FastAPI dashboard + Chart.js UI
```

---

## Evals (10 total)

| Layer | Eval | Type | What it measures |
|---|---|---|---|
| 1 | `intent_correctness` | Deterministic | Predicted intent == ground truth intent |
| 1 | `tool_selection` | Deterministic | Selected tool == expected tool |
| 1 | `reasoning_quality` | LLM-as-Judge | Was pre-tool reasoning sound? (Bedrock Llama 3.3 70B) |
| 2 | `param_schema_check` | Deterministic | Tool input validates against JSON schema |
| 2 | `param_value_check` | Deterministic | Tool input values match expected values |
| 2 | `result_interpretation` | LLM-as-Judge | Was next action appropriate given tool output? |
| 3 | `task_success` | Deterministic | Final answer contains expected output |
| 3 | `efficiency` | Deterministic | `optimal_turns / actual_turns` (TRACE paper) |
| 3 | `hallucination` | LLM-as-Judge | Final answer grounded in tool outputs? |
| 3 | `trajectory_quality` | Agent-as-Judge | Second Bedrock agent scores full trajectory 0–1 |

All LLM calls use `us.meta.llama3-3-70b-instruct-v1:0` via Bedrock Lambda proxy.

---

## Test Tasks (10)

| Task | Input | Tool | Layer 2 params |
|---|---|---|---|
| task_001 | CPU usage + top processes | `get-cpu-usage` | — |
| task_002 | Slow PC, check RAM | `get-memory-usage` | — |
| task_003 | Test internet connectivity | `test-internet` | — |
| task_004 | Disk space on all drives | `get-disk-usage` | — |
| task_005 | What is chrome.exe doing? | `get-process-info` | `process_name: chrome` |
| task_006 | Check Defender + firewall | `check-windows-defender` | — |
| task_007 | PC specs (OS, CPU, RAM) | `get-system-info` | — |
| task_008 | Recent Windows errors | `get-recent-errors` | — |
| task_009 | Find files >500MB in Downloads | `find-large-files` | `directory, min_size_mb: 500` |
| task_010 | Slow PC + disk light on | `get-disk-io` | — |

---

## Agent Versions

| Version | Behaviour | Expected scores |
|---|---|---|
| v1 | Vague system prompt — no tool routing guidance | Lower `tool_selection`, `reasoning_quality` |
| v2 | Explicit tool routing rules in system prompt | Higher across all layers |

The deployment gate compares v2 vs v1. If any metric regresses by more than 0.05, the gate fails and blocks the release.

---

## Reusable SDK

Attach tracing to any agent with minimal code change:

```python
# LangChain/LangGraph — one line
from agent_eval.adapters.langchain import LangChainTracer
agent.invoke(input, config={"callbacks": [LangChainTracer(version="v1", task_id="task_001")]})

# Custom agent — context manager
from agent_eval.adapters.generic import trace_session
with trace_session(version="v1", task_id="task_001") as t:
    t.log_tool_start("my_tool", params, turn=1)
    result = my_tool(**params)
    t.log_tool_end(result)
    t.log_final(answer, total_turns=1, success=True)

# Bedrock Agent Core — wrapper
from agent_eval.adapters.bedrock import BedrockTracer
agent = BedrockTracer(boto3_client, agent_id="...", alias_id="...", version="v1")
result = agent.invoke("query", session_id="sess-1")
```

Run evals on any trace:

```python
from agent_eval.evals.runner import run_evals
scores = run_evals(trace_id="...", label=task_dict)
# → {"intent_correctness": 1.0, "tool_selection": 1.0, "reasoning_quality": 0.82, ...}
```

---

## Dashboard

```bash
uvicorn agent_eval.server.app:app --reload --port 8000
```

- **Score History** — bar chart comparing all metrics across v1 vs v2. Hover any metric for description.
- **Trace Browser** — click any trace to see the full turn-by-turn timeline: LLM reasoning → tool call → tool output → next action → final answer.

---

## Deployment Gate

```bash
# First run — checks absolute minimum floors
python -m agent_eval.scoring.gate --version v1

# Regression check — fails if any metric drops > 0.05 vs prior
python -m agent_eval.scoring.gate --version v2 --prior v1
```

Integrated in `.github/workflows/eval_gate.yml` — gate runs on every push to `main` and exits 1 on failure.

---

## Dataset

```bash
# Label a trace manually
python -m agent_eval.dataset.labeller --trace-id <id> --task-file tasks/task_001.json

# Bump dataset version after adding new labels
python -c "from agent_eval.dataset.versioning import bump_version; print(bump_version())"
```

All data stored in `data/` (SQLite + JSONL):
- `data/traces.db` — structured queryable traces
- `data/dataset.db` — versioned eval dataset
- `data/scores.db` — score history across versions
- `data/traces.jsonl` — JSONL mirror of all spans

---

## Inspiration

| Tool | What we took |
|---|---|
| **LangSmith** | Traces-become-datasets workflow; callback-based tracing pattern |
| **RAGAS** | `ToolCallAccuracy`, `AgentGoalAccuracyWithReference` scorer design; `MultiTurnSample` schema |
| **Langfuse** | `config={"callbacks": [...]}` injection pattern |

---

## Papers

| Paper | arXiv | Used for |
|---|---|---|
| TRACE | 2602.21230 | Efficiency metric: `optimal_turns / actual_turns` |
| AgentTrace | 2602.10133 | Three-surface schema: cognitive / operational / contextual |
| Agent-as-a-Judge | 2410.10934 | `trajectory_judge.py` — evaluator agent inspects spans turn-by-turn |
