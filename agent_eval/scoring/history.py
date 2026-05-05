import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/scores.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS score_history (
            run_id        TEXT,
            agent_version TEXT,
            task_id       TEXT,
            layer         INTEGER,
            metric        TEXT,
            score         REAL,
            eval_type     TEXT,
            timestamp     TEXT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_version ON score_history(agent_version)")


_LAYER_MAP = {
    "intent_correctness": (1, "deterministic"),
    "tool_selection":     (1, "deterministic"),
    "reasoning_quality":  (1, "model_based"),
    "param_schema_check": (2, "deterministic"),
    "param_value_check":  (2, "deterministic"),
    "result_interpretation": (2, "model_based"),
    "task_success":       (3, "deterministic"),
    "efficiency":         (3, "deterministic"),
    "hallucination":      (3, "model_based"),
    "trajectory_quality": (3, "model_based"),
}


def insert_scores(agent_version: str, task_id: str, scores: dict[str, float]):
    _init()
    run_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        for metric, score in scores.items():
            layer, eval_type = _LAYER_MAP.get(metric, (0, "unknown"))
            conn.execute(
                "INSERT INTO score_history VALUES (?,?,?,?,?,?,?,?)",
                (run_id, agent_version, task_id, layer, metric, score, eval_type, ts),
            )


def scores_for_version(version: str) -> dict[str, float]:
    """Returns avg score per metric for a given agent version."""
    _init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT metric, AVG(score) as avg FROM score_history WHERE agent_version=? GROUP BY metric",
            (version,),
        ).fetchall()
    return {r["metric"]: round(r["avg"], 4) for r in rows}


def all_versions_summary() -> list[dict]:
    """Returns per-version per-layer per-metric averages for the dashboard."""
    _init()
    with _conn() as conn:
        rows = conn.execute("""
            SELECT agent_version, layer, metric, AVG(score) as avg_score, COUNT(*) as n
            FROM score_history
            GROUP BY agent_version, layer, metric
            ORDER BY agent_version, layer, metric
        """).fetchall()
    return [dict(r) for r in rows]
