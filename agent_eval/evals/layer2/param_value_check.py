def score(span: dict, label: dict) -> float:
    """
    Partial match: fraction of expected_tool_input keys whose values
    appear (as substrings) in the actual tool_input values.
    """
    expected: dict = label.get("expected_tool_input") or {}
    actual: dict   = (span.get("operational") or {}).get("tool_input") or {}
    if not expected:
        return 1.0
    hits = 0
    for key, exp_val in expected.items():
        act_val = str(actual.get(key, "")).lower()
        if str(exp_val).lower() in act_val:
            hits += 1
    return hits / len(expected)
