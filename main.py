"""
End-to-end runner.

Usage:
    python main.py                    # runs v1 + v2, evals both, prints report
    python main.py --version v1       # run only v1
    python main.py --skip-agent       # skip agent runs, only re-run evals on existing traces
"""
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parent))

from test_agent import run as agent_run
from agent_eval.core.tracer import BaseTracer
from agent_eval.evals.runner import run_evals
from agent_eval.dataset.store import add_item
from agent_eval.scoring.history import insert_scores, scores_for_version
from agent_eval.scoring.gate import run_gate
from agent_eval.config import BEDROCK_URL

console = Console()
TASKS_DIR = Path("tasks")

# v1: agent uses all tools freely (no guidance — may pick wrong tool)
# v2: system prompt guides tool selection more precisely (patched in agent.py via version flag)
VERSIONS = {
    "v1": "v1",
    "v2": "v2",
}


def load_tasks() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(TASKS_DIR.glob("task_*.json"))]


def run_version(version: str, tasks: list[dict]):
    console.print(f"\n[bold cyan]Running agent {version}[/bold cyan]")

    for task in tasks:
        tracer = BaseTracer(version=version, task_id=task["task_id"])
        console.print(f"  Task: {task['task_id']} — {task['input'][:60]}…")
        try:
            agent_run(
                query=task["input"],
                bedrock_url=BEDROCK_URL,
                tracer=tracer,
                version=version,
            )
        except Exception as e:
            console.print(f"  [red]Agent error: {e}[/red]")
            tracer.log_final(f"Error: {e}", total_turns=0, success=False)

        add_item(tracer.trace_id, task, label="auto", labeller="main.py")
        scores = run_evals(tracer.trace_id, task)
        insert_scores(version, task["task_id"], scores)
        console.print(f"  Scores: { {k: round(v, 2) for k, v in scores.items()} }")


def print_report(versions: list[str]):
    console.print("\n[bold]Score Report[/bold]")
    table = Table()
    table.add_column("Metric")
    for v in versions:
        table.add_column(v)

    all_scores = {v: scores_for_version(v) for v in versions}
    metrics = sorted({m for s in all_scores.values() for m in s})
    for metric in metrics:
        row = [metric]
        for v in versions:
            s = all_scores[v].get(metric)
            row.append(f"{s:.3f}" if s is not None else "-")
        table.add_row(*row)
    console.print(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Run only this version (v1 or v2)")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks()
    versions = [args.version] if args.version else ["v1", "v2"]

    if not args.skip_agent:
        for v in versions:
            run_version(v, tasks)

    print_report(versions)

    if "v2" in versions and "v1" in versions:
        console.print("\n[bold]Deployment Gate (v2 vs v1)[/bold]")
        run_gate("v2", prior="v1")
    elif versions:
        console.print(f"\n[bold]Deployment Gate ({versions[-1]} first-run)[/bold]")
        run_gate(versions[-1], prior=None)


if __name__ == "__main__":
    main()
