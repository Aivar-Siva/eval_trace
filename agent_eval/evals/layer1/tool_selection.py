def score(span: dict, label: dict) -> float:
    """1.0 if agent selected the correct tool, else 0.0."""
    selected = (span.get("cognitive") or {}).get("tool_selected", "").strip().lower()
    expected = label.get("ground_truth_tool", "").strip().lower()
    if not expected:
        return 1.0
    return 1.0 if selected == expected else 0.0
