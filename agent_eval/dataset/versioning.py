import sqlite3
from pathlib import Path

DB_PATH = Path("data/dataset.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _init():
    with _conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_version (
            id      INTEGER PRIMARY KEY CHECK (id=1),
            version TEXT NOT NULL DEFAULT 'v1'
        )""")
        conn.execute("INSERT OR IGNORE INTO dataset_version VALUES (1, 'v1')")


def current_version() -> str:
    _init()
    with _conn() as conn:
        row = conn.execute("SELECT version FROM dataset_version WHERE id=1").fetchone()
    return row[0] if row else "v1"


def bump_version() -> str:
    _init()
    cur = current_version()
    num = int(cur.lstrip("v")) + 1
    new = f"v{num}"
    with _conn() as conn:
        conn.execute("UPDATE dataset_version SET version=? WHERE id=1", (new,))
    return new


def list_versions() -> list[str]:
    _init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT dataset_version FROM dataset_items ORDER BY dataset_version"
        ).fetchall()
    return [r[0] for r in rows]
