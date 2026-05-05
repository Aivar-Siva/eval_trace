"""
Deployment gate CLI.

Usage:
    python -m agent_eval.scoring.gate --version v2 --prior v1
    python -m agent_eval.scoring.gate --version v1          # first-run: checks absolute minimums
"""
import argparse
import sys
from rich.console import Console
from rich.table import Table
from .history import scores_for_version
from ..config import ABSOLUTE_MINIMUMS, REGRESSION_DELTA

console = Console()


def run_gate(version: str, prior: str | None) -> bool:
    current = scores_for_version(version)
    if not current:
        console.print(f"[red]No scores found for version {version}[/red]")
        return False

    table = Table(title=f"Gate: {version}" + (f" vs {prior}" if prior else " (first run)"))
    table.add_column("Metric")
    table.add_column("Score")
    table.add_column("Threshold")
    table.add_column("Status")

    passed = True

    if prior:
        prior_scores = scores_for_version(prior)
        for metric, score in current.items():
            baseline = prior_scores.get(metric, score)
            drop = baseline - score
            threshold = f"Δ ≤ {REGRESSION_DELTA}"
            ok = drop <= REGRESSION_DELTA
            if not ok:
                passed = False
            table.add_row(metric, f"{score:.3f}", threshold, "✅" if ok else "❌")
    else:
        for metric, score in current.items():
            floor = ABSOLUTE_MINIMUMS.get(metric, 0.0)
            ok = score >= floor
            if not ok:
                passed = False
            table.add_row(metric, f"{score:.3f}", f"≥ {floor}", "✅" if ok else "❌")

    console.print(table)
    if passed:
        console.print(f"[green]Gate PASSED — {version} is deployable.[/green]")
    else:
        console.print(f"[red]Gate FAILED — {version} blocked.[/red]")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--prior", default=None)
    args = parser.parse_args()
    ok = run_gate(args.version, args.prior)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
