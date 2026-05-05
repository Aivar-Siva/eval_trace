import sqlite3
import json
from pathlib import Path
from .schema import TraceSpan

DB_PATH = Path("data/traces.db")


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS traces (
            trace_id         TEXT,
            agent_version    TEXT,
            task_id          TEXT,
            timestamp_start  TEXT,
            timestamp_end    TEXT,
            cognitive_json   TEXT,
            operational_json TEXT,
            contextual_json  TEXT,
            final_json       TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_trace_id ON traces(trace_id);
        CREATE INDEX IF NOT EXISTS idx_version  ON traces(agent_version);
        CREATE INDEX IF NOT EXISTS idx_task     ON traces(task_id);
        """)


def insert_span(span: TraceSpan):
    init_db()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO traces VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                span.get("trace_id"),
                span.get("agent_version"),
                span.get("task_id"),
                span.get("timestamp_start"),
                span.get("timestamp_end"),
                json.dumps(span.get("cognitive") or {}),
                json.dumps(span.get("operational") or {}),
                json.dumps(span.get("contextual") or {}),
                json.dumps(span.get("final") or {}),
            ),
        )


def query_by_trace_id(trace_id: str) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE trace_id=?", (trace_id,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def query_by_version(version: str) -> list[dict]:
    init_db()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE agent_version=?", (version,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def list_traces(limit: int = 100, version: str = None, task_id: str = None) -> list[dict]:
    init_db()
    sql = "SELECT * FROM traces WHERE 1=1"
    params: list = []
    if version:
        sql += " AND agent_version=?"
        params.append(version)
    if task_id:
        sql += " AND task_id=?"
        params.append(task_id)
    sql += " ORDER BY rowid DESC LIMIT ?"
    params.append(limit)
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("cognitive_json", "operational_json", "contextual_json", "final_json"):
        surface = key.replace("_json", "")
        d[surface] = json.loads(d.pop(key) or "{}")
    return d
