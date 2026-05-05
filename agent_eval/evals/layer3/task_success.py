def score(span: dict, label: dict) -> float:
    """1.0 if expected_output appears in final output (case-insensitive substring)."""
    final = span.get("final") or {}
    output   = final.get("output", "").lower()
    expected = label.get("expected_output", "").lower()
    if not expected:
        return 1.0 if final.get("success") else 0.0
    return 1.0 if expected in output else 0.0
