# Plan: Agent Evaluation Framework
> **For Cursor.** Problem 5 — Agent Evaluation Framework assignment.
> Build a framework that measures agent quality at every execution stage, not just the final output.

---

## 1. Problem Statement

AI agents executing tool calls (search, code execution, DB queries, API calls) fail in ways invisible from the final output alone. An agent can reach the correct answer via wrong reasoning, or fail at step 3 of 7 with no signal in the final result.

Static LLM benchmarks do not transfer to agents because agents are multi-step and correctness depends on reasoning quality AND action quality at every step.

**What needs to be built:**

- A framework that captures structured traces from any LangGraph agent at runtime — zero code modification to the agent, via LangChain's `BaseCallbackHandler` passed in `config={"callbacks": [...]}` at invocation time
- Evals defined at three layers: pre-tool-call, at/after tool call, end-to-end
- A dataset store where production traces can be sampled, labelled, and versioned
- A score history that tracks agent performance across releases
- A deployment gate that blocks a release if eval scores regress

**Hard constraints from the assignment:**

- Framework-agnostic (the callback-based tracer works with LangChain, LangGraph, custom agents — anything that accepts `config={"callbacks": [...]}`)
- Supports deterministic evals (schema validation) AND model-based evals (LLM-as-judge via Groq)
- Traces must be structured and queryable — not just flat logs
- Datasets must be versioned
- At least one eval at each of the three layers
- Must integrate with or take inspiration from at least one existing tool (RAGAS)

---

## 2. Foundational Research

### RAGAS

RAGAS is primarily a RAG eval library, but its agent support works by converting a trace (input + reasoning + tool calls + tool responses + final output) into a structured object and running metric scorers on it. The scorers are LLM-as-judge prompts.

RAGAS does **not** handle trace storage, dataset versioning, or score history — it only provides the scorers. The correct import paths as of RAGAS current version:

```
# Correct — modern API
from ragas.metrics.collections import ToolCallAccuracy
from ragas.metrics import AgentGoalAccuracyWithReference

# Deprecated — do NOT use
from ragas.metrics import ToolCallAccuracy  # removed in v1.0
```

Use RAGAS for scorer implementations only, not for infrastructure.

### LangChain Callbacks (the tracing mechanism)

LangChain's `BaseCallbackHandler` provides hooks that fire at every stage of agent execution with zero modification to the agent under test:

| Hook | Fires when | Used for |
|---|---|---|
| `on_chat_model_start` | LLM receives input | Capture reasoning prompt → Layer 1 |
| `on_llm_end` | LLM produces output | Capture tool selection + reasoning text → Layer 1 |
| `on_tool_start` | Tool is invoked | Capture tool name + parameters → Layer 2 |
| `on_tool_end` | Tool returns result | Capture tool output + next decision → Layer 2 |
| `on_chain_start` | LangGraph node begins | Capture node entry (for LangGraph graphs) |
| `on_chain_end` | LangGraph node ends | Capture full agent run output → Layer 3 |
| `on_agent_finish` | Agent produces final answer | Capture final output + turn count → Layer 3 |

The handler is passed at invocation time only:
```
graph.invoke({"messages": [...]}, config={"callbacks": [AgentTracingCallback()]})
```

This means the agent code itself is never touched. The framework is attached at run time, not build time.

**Known limitation:** In LangGraph, `on_agent_action` and `on_agent_finish` may not fire depending on the graph structure — use `on_chain_start`/`on_chain_end` for node-level tracing and `on_tool_start`/`on_tool_end` for tool tracing instead. These fire reliably for all LangGraph patterns.

### Langfuse / LangSmith Architecture (taken as inspiration)

Key architectural idea from LangSmith: **traces become datasets**. You see a production failure → sample it → label it → it joins the regression suite. We replicate this workflow locally with SQLite.

Langfuse passes a `CallbackHandler` into `config={"callbacks": [...]}` — exactly the same pattern we use. This validates the approach.

### Key Papers

| Paper | arXiv | What it contributes to this framework |
|---|---|---|
| **TRACE** | 2602.21230 | Hierarchical Trajectory Utility Function — scores full trajectory, not just final answer. Source of the efficiency metric: `actual_turns / optimal_turns`. |
| **AgentTrace** | 2602.10133 | Three-surface taxonomy: cognitive / operational / contextual. Each callback hook maps to one surface. |
| **Agent-as-a-Judge** | 2410.10934 | Agentic evaluator with tool use evaluates another agent's trajectory. Implemented here as a second LangGraph graph (the evaluator graph) that receives a serialized trace and scores it. |
| **Claw-Eval** | 2604.06132 | Three evidence channels (execution traces, audit logs, environment snapshots) to ground scores in observable evidence. Informs our three-surface schema design. |

---

## 3. Architecture

Four components. They communicate via SQLite and JSONL — no tight coupling, no shared state at runtime.

```
+----------------------------------+
|  LangGraph Demo Agent            |
|  (agent/research_agent.py)       |
|  Tools: web_search, calculator,  |
|         summarize                |
|  LLM: ChatGroq (llama-3.3-70b)  |
+----------+-----------------------+
           |
           | graph.invoke(input, config={"callbacks": [AgentTracingCallback()]})
           | Zero modification to agent code
           v
+----------+-----------------------+
|  AgentTracingCallback            |
|  (sdk/callback.py)               |
|  Subclass of BaseCallbackHandler |
|                                  |
|  on_chat_model_start → cognitive |
|  on_llm_end          → cognitive |
|  on_tool_start       → operational|
|  on_tool_end         → operational|
|  on_chain_start/end  → contextual |
+----------+-----------------------+
           |
           | emits structured spans
           v
+----------+-----------------------+
|  Trace Collector                 |
|  (tracer/collector.py)           |
|  Writes: JSONL (one span/line)   |
|  Inserts: SQLite (queryable)     |
|  Schema: three-surface TypedDict |
+----------+-----------------------+
           |
           +-------- sample + label --------+
           |                                v
           |                   +------------+----------+
           |                   |  Dataset Store         |
           |                   |  (dataset/store.py)    |
           |                   |  SQLite: dataset_items |
           |                   |  Versioned by string   |
           |                   |  (v1, v2, v3...)       |
           |                   +------------+----------+
           |                                |
           v                                v
+----------+-----------------------+  +-----+------------------+
|  Eval Runner                     |  |  Score History DB       |
|  (evals/runner.py)               |  |  (scoring/history.py)   |
|                                  |  |  SQLite: score_history  |
|  Layer 1 — pre-tool-call         |  |  Tagged by agent_version|
|  Layer 2 — at/after tool call    |  +-----+------------------+
|  Layer 3 — end-to-end            |        |
|                                  |        v
|  Scorers:                        |  +-----+------------------+
|  - deterministic (jsonschema)    |  |  Deployment Gate        |
|  - LLM-as-judge (ChatGroq)       |  |  (scoring/gate.py)      |
|  - Agent-as-judge (LangGraph     |  |  CLI + GitHub Actions   |
|    evaluator graph, ChatGroq)    |  +------------------------+
+----------------------------------+
```

---

## 4. Key Architecture Decisions

### Decision 1 — Callback-based tracing, not decorator wrapping

The original plan used `wrap_llm_call()` / `wrap_tool_call()` decorators, which require touching the agent's call sites. This is replaced entirely by `BaseCallbackHandler` subclassing. The handler is injected at invocation time via `config={"callbacks": [...]}`. The agent source is never modified. This is how Langfuse and LangSmith themselves instrument LangGraph agents.

### Decision 2 — LLM provider: Groq with llama-3.3-70b-versatile

All LLM calls (agent + LLM-as-judge + Agent-as-judge evaluator) use `ChatGroq`. Two separate `ChatGroq` instances: one for the agent under test, one for the evaluator. They can use different models if needed (e.g., agent on `llama-3.3-70b-versatile`, evaluator on `llama-3.1-8b-instant` to reduce eval cost).

```
GROQ_API_KEY=...
AGENT_MODEL=llama-3.3-70b-versatile
EVAL_MODEL=llama-3.3-70b-versatile
TAVILY_API_KEY=...   # for web_search tool
```

### Decision 3 — Agent-as-Judge is a real LangGraph evaluator graph

Not a stub, not an LLM-as-judge with a long prompt relabeled. A second `StateGraph` in `evals/layer3/trajectory_judge.py` that:
1. Receives the serialized trace as input state
2. Has access to one tool: `lookup_span(trace_id, turn_number)` — reads from the trace DB
3. Reasons over the trajectory turn-by-turn
4. Produces a structured score + justification

This directly implements the Agent-as-a-Judge paper (2410.10934).

### Decision 4 — Demo agent: research agent with three real tools

`agent/research_agent.py` uses:
- `web_search` via Tavily (real external API call)
- `calculator` via a `@tool`-decorated Python function (deterministic, no API key)
- `summarize` via a second `ChatGroq` call wrapped as a `@tool`

Three tools is the minimum to exercise all three layers meaningfully. Tavily has a free tier — no cost concern.

### Decision 5 — Deployment gate: local CLI + minimal GitHub Actions

`scoring/gate.py` runs as a CLI script. A `.github/workflows/eval_gate.yml` file calls it with `|| exit 1`. The YAML is minimal — its only job is to run the gate and fail the workflow if the gate returns non-zero. No infra, no Docker, no deployment pipeline beyond this.

### Decision 6 — First-run gate behavior

The original plan had a silent bug: `prior_scores.get(metric, score)` meant the gate always passed on the first run (prior defaulted to current, diff = 0). Fix: on first run (no prior version in DB), the gate checks against absolute minimum thresholds instead of regression deltas. Both threshold sets live in `config.py`.

---

## 5. Trace Schema (AgentTrace Three-Surface Taxonomy)

Every span emitted by `AgentTracingCallback` follows this schema. Stored as one JSON object per line in JSONL, and inserted into SQLite for queryability.

```json
{
  "trace_id": "uuid4",
  "agent_version": "v1.0.0",
  "task_id": "task_001",
  "timestamp_start": "ISO8601",
  "timestamp_end": "ISO8601",

  "cognitive": {
    "task_query": "What is the GDP of India in 2024?",
    "reasoning_text": "I need to search for current GDP figures...",
    "tool_selected": "web_search",
    "intent_label": "information_retrieval"
  },

  "operational": {
    "tool_name": "web_search",
    "tool_input": {"query": "India GDP 2024"},
    "tool_output": "India's GDP in 2024 was approximately $3.9 trillion...",
    "tool_output_tokens": 312,
    "next_action": "summarize",
    "turn_number": 2
  },

  "contextual": {
    "context_tokens_before": 980,
    "context_tokens_after": 1292,
    "node_name": "agent",
    "run_id": "uuid4"
  },

  "final": {
    "output": "India's GDP in 2024 was approximately $3.9 trillion.",
    "success": true,
    "total_turns": 3,
    "total_tokens": 2140
  }
}
```

**SQLite table: `traces`**

```sql
CREATE TABLE traces (
    trace_id        TEXT PRIMARY KEY,
    agent_version   TEXT,
    task_id         TEXT,
    timestamp_start TEXT,
    timestamp_end   TEXT,
    cognitive_json  TEXT,   -- JSON blob
    operational_json TEXT,  -- JSON blob
    contextual_json TEXT,   -- JSON blob
    final_json      TEXT    -- JSON blob
);
```

---

## 6. Project Structure

```
agent-eval/
├── main.py                        # Entry point: run agent v1 + v2 → collect traces → run evals → gate → report
├── config.py                      # Model names, thresholds, tool schemas, eval weights
├── .env.example                   # GROQ_API_KEY, TAVILY_API_KEY, AGENT_MODEL, EVAL_MODEL
│
├── sdk/
│   └── callback.py                # AgentTracingCallback(BaseCallbackHandler)
│                                  # Implements: on_chat_model_start, on_llm_end,
│                                  #             on_tool_start, on_tool_end,
│                                  #             on_chain_start, on_chain_end
│
├── tracer/
│   ├── schema.py                  # TraceSpan TypedDict — three-surface schema
│   ├── collector.py               # Receives spans from callback, writes JSONL + SQLite
│   └── db.py                      # SQLite helpers: insert_span, query_by_trace_id,
│                                  #                 query_by_version, query_by_task
│
├── evals/
│   ├── layer1/
│   │   ├── intent_correctness.py  # Deterministic: predicted intent == ground_truth intent
│   │   ├── tool_selection.py      # Deterministic: tool_selected == expected_tool
│   │   └── reasoning_quality.py   # LLM-as-judge (ChatGroq): "Was this reasoning sound? 1–5"
│   ├── layer2/
│   │   ├── param_schema_check.py  # Deterministic: jsonschema validate tool_input
│   │   ├── param_value_check.py   # Deterministic: tool_input values == expected values from label
│   │   └── result_interpretation.py # LLM-as-judge (ChatGroq): "Given tool output, was next_action appropriate?"
│   ├── layer3/
│   │   ├── task_success.py        # Deterministic: pass/fail vs expected_output
│   │   ├── efficiency.py          # Deterministic: actual_turns / optimal_turns (TRACE paper)
│   │   ├── hallucination.py       # LLM-as-judge (ChatGroq): unsupported claims in final output?
│   │   └── trajectory_judge.py    # Agent-as-judge: LangGraph evaluator graph with lookup_span tool
│   └── runner.py                  # Orchestrates all scorers on one trace, returns scores dict
│
├── dataset/
│   ├── store.py                   # CRUD: add_item, get_item, list_by_version
│   ├── versioning.py              # bump_version(), current_version(), list_versions()
│   └── labeller.py                # CLI: python -m dataset.labeller --trace-id <id>
│                                  #      Prompts for label fields, saves to dataset_items
│
├── scoring/
│   ├── history.py                 # insert_score(), scores_for_version(), compare_versions()
│   └── gate.py                    # CLI: python -m scoring.gate --version v2 --prior v1
│                                  # First-run: checks absolute minimums from config.py
│                                  # Subsequent runs: checks regression delta vs prior
│
├── dashboard/
│   └── report.py                  # Generates PLAN_REPORT.md: score tables per layer per version,
│                                  # regression highlights, trace count, dataset version
│
├── agent/
│   └── research_agent.py          # LangGraph ReAct agent, ChatGroq (llama-3.3-70b-versatile)
│                                  # Tools: web_search (Tavily), calculator (@tool), summarize (@tool+ChatGroq)
│                                  # v1: deliberate flaw (wrong tool preference)
│                                  # v2: fixed
│
├── tasks/
│   ├── task_001.json              # Factual question → web_search → summarize
│   ├── task_002.json              # Math from description → calculator
│   └── task_003.json              # Document summarization → web_search → summarize → extract
│
├── traces/                        # Runtime JSONL output (gitignored in prod)
├── datasets/                      # SQLite dataset store
├── scores/                        # SQLite score history
│
├── .github/
│   └── workflows/
│       └── eval_gate.yml          # Minimal CI: install deps → run evals → run gate → exit 1 on fail
│
├── requirements.txt
└── README.md
```

---

## 7. SDK Callback Design (Zero-Modification Tracing)

`sdk/callback.py` subclasses `BaseCallbackHandler` from `langchain_core.callbacks`. It is instantiated once per agent run and passed in the invocation config. The collector reference is held on the instance.

Callback hook → trace surface mapping:

| Hook | Surface | What to capture |
|---|---|---|
| `on_chat_model_start` | cognitive | `task_query` from messages, `timestamp_start` |
| `on_llm_end` | cognitive | `reasoning_text` from response content, `tool_selected` from tool_calls if present |
| `on_tool_start` | operational | `tool_name`, `tool_input`, `turn_number`, `timestamp_start` |
| `on_tool_end` | operational | `tool_output`, `tool_output_tokens` (estimated), `timestamp_end` |
| `on_chain_start` | contextual | `node_name` from serialized, `run_id`, `context_tokens_before` |
| `on_chain_end` | contextual | `context_tokens_after`, `next_action` inferred from output |
| `on_chain_end` (root) | final | `output`, `total_turns`, `total_tokens`, `success` flag |

The callback accumulates span fields across hooks (since one span's data arrives across multiple hooks) and flushes a complete span to the collector when enough fields are populated. Turn number is tracked as an instance counter incremented on each `on_tool_start`.

---

## 8. Three-Layer Eval Design

### Layer 1 — Pre-Tool-Call (Cognitive Surface)

What to measure: Was the agent's reasoning sound before selecting a tool? Did it identify the right intent? Did it select the right tool?

| Eval | Type | Scorer | Trace fields needed |
|---|---|---|---|
| Intent classification correctness | Deterministic | String match: `cognitive.intent_label == label.ground_truth_intent` | `intent_label`, `ground_truth_intent` |
| Tool selection accuracy | Deterministic | String match: `cognitive.tool_selected == label.ground_truth_tool` | `tool_selected`, `ground_truth_tool` |
| Reasoning quality | Model-based | LLM-as-judge via ChatGroq: prompt asks "Given the task, rate the reasoning 1–5 and explain" | `task_query`, `reasoning_text` |

### Layer 2 — At/After Tool Call (Operational Surface)

What to measure: Were tool parameters correct? Did the agent interpret the result and decide the right next step?

| Eval | Type | Scorer | Trace fields needed |
|---|---|---|---|
| Tool parameter schema validation | Deterministic | `jsonschema.validate(tool_input, TOOL_SCHEMAS[tool_name])` — schemas defined in `config.py` | `tool_name`, `tool_input` |
| Parameter value correctness | Deterministic | Field-by-field compare `tool_input` vs `label.expected_tool_input` | `tool_input`, `expected_tool_input` |
| Result interpretation quality | Model-based | LLM-as-judge via ChatGroq: "Given this tool output, was the next action appropriate?" | `tool_output`, `next_action` |

### Layer 3 — End-to-End (Contextual Surface)

What to measure: Did the agent accomplish the goal? How efficiently? Are there hallucinations?

| Eval | Type | Scorer | Trace fields needed |
|---|---|---|---|
| Task success | Deterministic | Exact match or substring match: `final.output` vs `label.expected_output` | `final.output`, `expected_output` |
| Efficiency | Deterministic | `final.total_turns / label.optimal_turns` — from TRACE paper | `total_turns`, `optimal_turns` |
| Hallucination check | Model-based | LLM-as-judge via ChatGroq: "Does the final response contain claims not supported by any tool output in this trace?" | `final.output`, all `tool_output` fields |
| Trajectory quality | Model-based | Agent-as-judge: LangGraph evaluator graph with `lookup_span` tool. Scores goal completion + reasoning coherence over the full trajectory | full trace |

---

## 9. Agent-as-Judge: LangGraph Evaluator Graph

`evals/layer3/trajectory_judge.py` defines a second LangGraph `StateGraph`. It is entirely separate from the agent under test and uses its own `ChatGroq` instance.

**State:** `{"trace_id": str, "turn_index": int, "observations": list[str], "final_score": float | None}`

**Nodes:**
- `load_trace` — reads all spans for `trace_id` from SQLite, sets `turn_index = 0`
- `evaluate_turn` — LLM call: "Given this span, was the agent's reasoning and action appropriate? Note any issues."
- `should_continue` — edge: if `turn_index < total_turns`, loop back; else go to `score`
- `score` — LLM call: "Given all observations, score the trajectory 0.0–1.0 for goal completion and reasoning coherence. Return JSON."

**Tool available to the evaluator:** `lookup_span(trace_id: str, turn: int) -> dict` — reads from `tracer/db.py`. This is what makes it an agent rather than a plain LLM call.

The evaluator graph is invoked from `evals/runner.py` like any other scorer. Its output is a float score inserted into `score_history`.

---

## 10. Dataset Versioning

SQLite table `dataset_items`. Version is a string bumped manually or via the labeller CLI.

```sql
CREATE TABLE dataset_items (
    item_id              TEXT PRIMARY KEY,
    dataset_version      TEXT,
    trace_id             TEXT,
    task_input           TEXT,
    expected_output      TEXT,
    ground_truth_tool    TEXT,
    ground_truth_intent  TEXT,
    expected_tool_input  TEXT,   -- JSON blob
    optimal_turns        INTEGER,
    label                TEXT,
    labeller             TEXT,
    created_at           TEXT
);
```

Version bump policy: when new items are added to the dataset, `versioning.py` increments the version string (`v1` → `v2`). No migrations. The version string is stored on each item at insert time, so old items retain their version label and historical score queries remain accurate.

---

## 11. Score History Schema

```sql
CREATE TABLE score_history (
    run_id          TEXT,
    agent_version   TEXT,
    task_id         TEXT,
    layer           INTEGER,   -- 1, 2, or 3
    metric          TEXT,
    score           REAL,
    eval_type       TEXT,      -- "deterministic" or "model_based"
    timestamp       TEXT
);
```

Dashboard query (used by `dashboard/report.py`):

```sql
SELECT agent_version, layer, metric, AVG(score) as avg_score, COUNT(*) as n
FROM score_history
GROUP BY agent_version, layer, metric
ORDER BY agent_version, layer, metric;
```

---

## 12. Deployment Gate

`scoring/gate.py` handles two cases:

**Case 1 — No prior version exists (first run):**
Check each metric against the absolute minimum floor defined in `config.py`. If any metric is below its floor, gate fails. This fixes the original silent-pass bug.

**Case 2 — Prior version exists:**
For each metric, compute `prior_score - current_score`. If the delta exceeds the regression threshold (default 0.05, configurable per metric in `config.py`), gate fails.

Gate exits with code 0 (pass) or 1 (fail). The GitHub Actions step uses `|| exit 1` to fail the workflow.

**`.github/workflows/eval_gate.yml`** — minimal, ~15 lines:
- Trigger: `push` to `main`
- Steps: checkout → `pip install -r requirements.txt` → `python main.py --version $VERSION` → `python -m scoring.gate --version $VERSION --prior $PRIOR_VERSION`

---

## 13. Demo Agent: research_agent.py

A LangGraph ReAct agent (`create_react_agent` pattern) using `ChatGroq(model="llama-3.3-70b-versatile")`.

**Three tools:**
- `web_search(query: str) -> str` — Tavily `TavilySearchResults`, max_results=3
- `calculator(expression: str) -> str` — evaluates a Python math expression safely
- `summarize(text: str, focus: str) -> str` — secondary `ChatGroq` call, returns a focused summary

**Two versions:**

`v1` — deliberately flawed: the agent's system prompt nudges it to prefer `web_search` even for pure math tasks, causing tool selection failures on task_002. This makes Layer 1 (tool_selection) and Layer 2 (param_schema_check) scores regress visibly.

`v2` — fixed: system prompt corrected, tool routing logic sound.

The version is passed as a constructor argument to `research_agent.py` and written into every trace span as `agent_version`. No other code changes between v1 and v2 — the flaw is entirely in the system prompt.

---

## 14. Tasks

Three task JSON files in `tasks/`. Each exercises different tool paths.

**task_001.json** — Factual question
```json
{
  "task_id": "task_001",
  "input": "What was India's GDP in 2024 in USD?",
  "expected_output": "approximately $3.9 trillion",
  "ground_truth_tool": "web_search",
  "ground_truth_intent": "information_retrieval",
  "expected_tool_input": {"query": "India GDP 2024 USD"},
  "optimal_turns": 2
}
```

**task_002.json** — Math from description
```json
{
  "task_id": "task_002",
  "input": "A rectangle is 14.5m wide and 9.3m long. What is its area?",
  "expected_output": "134.85",
  "ground_truth_tool": "calculator",
  "ground_truth_intent": "calculation",
  "expected_tool_input": {"expression": "14.5 * 9.3"},
  "optimal_turns": 1
}
```

**task_003.json** — Research and summarize
```json
{
  "task_id": "task_003",
  "input": "Find recent developments in LLM agent evaluation and summarize the key trends.",
  "expected_output": "trajectory-aware evaluation",
  "ground_truth_tool": "web_search",
  "ground_truth_intent": "research_summarization",
  "expected_tool_input": {"query": "LLM agent evaluation 2025 trends"},
  "optimal_turns": 3
}
```

---

## 15. Scorers: What Comes From RAGAS vs What Is Custom

| Scorer | Source | Import / location |
|---|---|---|
| Tool call accuracy | RAGAS | `from ragas.metrics.collections import ToolCallAccuracy` |
| Goal completion | RAGAS | `from ragas.metrics import AgentGoalAccuracyWithReference` |
| Schema validation | `jsonschema` stdlib | `evals/layer2/param_schema_check.py` |
| Intent correctness | Custom deterministic | `evals/layer1/intent_correctness.py` |
| Tool selection | Custom deterministic | `evals/layer1/tool_selection.py` |
| Task success | Custom deterministic | `evals/layer3/task_success.py` |
| Efficiency | Custom deterministic (TRACE paper) | `evals/layer3/efficiency.py` |
| Reasoning quality | LLM-as-judge, ChatGroq | `evals/layer1/reasoning_quality.py` |
| Result interpretation | LLM-as-judge, ChatGroq | `evals/layer2/result_interpretation.py` |
| Hallucination | LLM-as-judge, ChatGroq | `evals/layer3/hallucination.py` |
| Trajectory quality | Agent-as-judge, LangGraph + ChatGroq | `evals/layer3/trajectory_judge.py` |

Do not rebuild what RAGAS provides. RAGAS scorers require traces to be converted to `MultiTurnSample` format — this conversion happens inside `evals/runner.py`.

---

## 16. Implementation Order

Build in this exact sequence. Each step is independently testable before the next begins.

1. `tracer/schema.py` — TypedDict for TraceSpan (no dependencies)
2. `tracer/db.py` — SQLite helpers (no dependencies beyond stdlib)
3. `tracer/collector.py` — receives spans, writes JSONL + calls db.py
4. `sdk/callback.py` — BaseCallbackHandler subclass, calls collector
5. `agent/research_agent.py` (v1) — LangGraph agent with Groq + 3 tools, no evals yet
6. Wire `sdk/callback.py` onto agent invocation, run one task, verify JSONL + SQLite populated
7. `dataset/store.py` + `dataset/versioning.py` + `dataset/labeller.py`
8. Manually label the traces from step 6 using the labeller CLI — this creates the first dataset items
9. `evals/layer1/intent_correctness.py` + `tool_selection.py` — deterministic first
10. `evals/layer2/param_schema_check.py` + `param_value_check.py` — deterministic
11. `evals/layer3/task_success.py` + `efficiency.py` — deterministic
12. `evals/runner.py` — runs all deterministic evals, verify scores appear
13. `evals/layer1/reasoning_quality.py` — first LLM-as-judge eval
14. `evals/layer2/result_interpretation.py` — LLM-as-judge
15. `evals/layer3/hallucination.py` — LLM-as-judge
16. `evals/layer3/trajectory_judge.py` — Agent-as-judge LangGraph evaluator graph
17. `scoring/history.py` + `scoring/gate.py`
18. `agent/research_agent.py` (v2) — fix the deliberate flaw, run full suite again
19. `dashboard/report.py` — generate Markdown report comparing v1 vs v2
20. `main.py` — ties everything end to end
21. `.github/workflows/eval_gate.yml` — minimal CI

---

## 17. Requirements

```
# LLM + Agent
langchain-groq
langchain-core
langgraph
langchain-community      # for TavilySearchResults

# Evals
ragas

# Tooling
jsonschema
python-dotenv
rich                     # CLI output formatting
tavily-python            # web_search tool

# All storage is SQLite (stdlib) + JSONL (stdlib)
# No other infrastructure required
```

---

## 18. Alignment Check vs. Assignment Requirements

| Requirement | Covered? | Notes |
|---|---|---|
| Framework-agnostic | ✅ | BaseCallbackHandler via `config={"callbacks": [...]}` — zero agent modification |
| Deterministic evals | ✅ | Schema validation, intent match, tool name match, task success, efficiency |
| Model-based evals | ✅ | LLM-as-judge ×3 (reasoning, interpretation, hallucination) + Agent-as-judge ×1 (trajectory) |
| Structured queryable traces | ✅ | SQLite with three-surface schema + JSONL |
| Versioned datasets | ✅ | `dataset_version` column, bump via `versioning.py` |
| At least one eval per layer | ✅ | Layer 1: 3 evals, Layer 2: 3 evals, Layer 3: 4 evals |
| Integration with existing tool | ✅ | RAGAS `ToolCallAccuracy` + `AgentGoalAccuracyWithReference` used directly |
| Working framework on real agent | ✅ | `research_agent.py` with 3 real tools, 3 tasks |
| Two agent versions with score history | ✅ | v1 (flawed system prompt) → v2 (fixed), full score history |
| Dashboard / report | ✅ | Markdown report with score tables per layer per version, regression highlights |
| Deployment gate | ✅ | `scoring/gate.py` CLI + GitHub Actions, first-run floor check + regression delta |

---

## 19. Sources

- RAGAS Agent Evaluation (correct import paths): https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/
- LangChain BaseCallbackHandler API reference: https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html
- Langfuse LangGraph integration (callback pattern validation): https://langfuse.com/guides/cookbook/integration_langgraph
- LangSmith LangGraph tracing: https://docs.langchain.com/langsmith/trace-with-langgraph
- ChatGroq + LangGraph usage: https://console.groq.com/docs/langchain
- Tavily + LangChain tool: https://python.langchain.com/docs/integrations/tools/tavily_search/
- TRACE paper (arXiv 2602.21230): Trajectory-Aware Comprehensive Evaluation for Deep Research Agents
- AgentTrace paper (arXiv 2602.10133): A Structured Logging Framework for Agent System Observability
- Agent-as-a-Judge (arXiv 2410.10934): Evaluate Agents with Agents
- Claw-Eval (arXiv 2604.06132): Toward Trustworthy Evaluation of Autonomous Agents
