import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/dataset.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_items (
            item_id             TEXT PRIMARY KEY,
            dataset_version     TEXT,
            trace_id            TEXT,
            task_id             TEXT,
            task_input          TEXT,
            expected_output     TEXT,
            ground_truth_tool   TEXT,
            ground_truth_intent TEXT,
            expected_tool_input TEXT,
            optimal_turns       INTEGER,
            label               TEXT,
            labeller            TEXT,
            created_at          TEXT
        )""")


def add_item(trace_id: str, task: dict, label: str = "unlabelled", labeller: str = "auto") -> str:
    _init()
    from .versioning import current_version
    item_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute(
            "INSERT INTO dataset_items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                item_id, current_version(), trace_id,
                task.get("task_id", ""),
                task.get("input", ""),
                task.get("expected_output", ""),
                task.get("ground_truth_tool", ""),
                task.get("ground_truth_intent", ""),
                json.dumps(task.get("expected_tool_input") or {}),
                task.get("optimal_turns", 1),
                label, labeller,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return item_id


def get_item(item_id: str) -> dict | None:
    _init()
    with _conn() as conn:
        row = conn.execute("SELECT * FROM dataset_items WHERE item_id=?", (item_id,)).fetchone()
    return _row(row) if row else None


def list_by_version(version: str) -> list[dict]:
    _init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM dataset_items WHERE dataset_version=?", (version,)
        ).fetchall()
    return [_row(r) for r in rows]


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["expected_tool_input"] = json.loads(d.get("expected_tool_input") or "{}")
    return d
