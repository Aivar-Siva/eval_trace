def score(span: dict, label: dict) -> float:
    """1.0 if predicted intent matches ground truth, else 0.0."""
    predicted = (span.get("cognitive") or {}).get("intent_label", "").strip().lower()
    expected  = label.get("ground_truth_intent", "").strip().lower()
    if not expected:
        return 1.0
    return 1.0 if predicted == expected else 0.0
