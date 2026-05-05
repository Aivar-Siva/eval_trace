def score(span: dict, label: dict) -> float:
    """
    Efficiency = optimal_turns / actual_turns  (TRACE paper formula).
    Capped at 1.0. Lower actual turns than optimal → 1.0 (bonus).
    """
    final = span.get("final") or {}
    actual  = final.get("total_turns") or 1
    optimal = label.get("optimal_turns") or 1
    return min(1.0, optimal / actual)
