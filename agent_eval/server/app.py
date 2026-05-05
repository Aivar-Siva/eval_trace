"""
FastAPI visualization server.

Run: uvicorn agent_eval.server.app:app --reload --port 8000
"""
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from ..core.db import list_traces, query_by_trace_id
from ..scoring.history import all_versions_summary

app = FastAPI(title="Agent Eval Dashboard")

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/scores")
def scores():
    """Per-version per-layer per-metric averages."""
    return all_versions_summary()


@app.get("/api/traces")
def traces(
    version: str = Query(None),
    task_id: str = Query(None),
    limit:   int = Query(50),
):
    """Trace list with optional filters."""
    return list_traces(limit=limit, version=version, task_id=task_id)


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str):
    """All spans for a single trace."""
    return query_by_trace_id(trace_id)


@app.get("/api/versions")
def versions():
    """Distinct agent versions in the score history."""
    from ..scoring.history import _init, _conn
    _init()
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT agent_version FROM score_history ORDER BY agent_version"
        ).fetchall()
    return [r[0] for r in rows]
