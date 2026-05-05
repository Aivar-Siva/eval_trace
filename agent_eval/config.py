import os
from dotenv import load_dotenv

load_dotenv()

# Bedrock proxy — agent LLM + LLM-as-judge
BEDROCK_URL = os.getenv("BEDROCK_URL", "https://ug36pewdpyfaepokw55klfit7y0ltgbn.lambda-url.us-west-2.on.aws/")
EVAL_MODEL  = os.getenv("EVAL_MODEL", "us.meta.llama3-3-70b-instruct-v1:0")

# JSON schemas for Layer 2 schema validation (MCP tools with parameters)
TOOL_SCHEMAS: dict = {
    "get-process-info": {
        "type": "object",
        "properties": {"process_name": {"type": "string"}},
        "required": ["process_name"],
    },
    "kill-process": {
        "type": "object",
        "properties": {"pid": {"type": "integer"}},
        "required": ["pid"],
    },
    "find-large-files": {
        "type": "object",
        "properties": {
            "directory":    {"type": "string"},
            "min_size_mb":  {"type": "integer"},
        },
        "required": ["directory"],
    },
}

# Deployment gate thresholds
ABSOLUTE_MINIMUMS: dict = {
    "intent_correctness":      0.5,
    "tool_selection":          0.5,
    "reasoning_quality":       0.4,
    "param_schema_check":      0.8,
    "param_value_check":       0.5,
    "result_interpretation":   0.4,
    "task_success":            0.5,
    "efficiency":              0.4,
    "hallucination":           0.5,
    "trajectory_quality":      0.4,
}

REGRESSION_DELTA: float = 0.05
