import jsonschema
from ...config import TOOL_SCHEMAS


def score(span: dict, label: dict) -> float:
    """1.0 if tool_input validates against the tool's JSON schema, else 0.0."""
    op = span.get("operational") or {}
    tool_name  = op.get("tool_name", "")
    tool_input = op.get("tool_input") or {}
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return 1.0  # unknown tool — no schema to validate against
    try:
        jsonschema.validate(tool_input, schema)
        return 1.0
    except jsonschema.ValidationError:
        return 0.0
