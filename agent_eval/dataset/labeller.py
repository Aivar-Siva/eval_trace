"""
CLI labeller: python -m agent_eval.dataset.labeller --trace-id <id> --task-file tasks/task_001.json

Prompts for label, saves to dataset_items.
"""
import argparse
import json
from pathlib import Path
from .store import add_item
from .versioning import current_version


def main():
    parser = argparse.ArgumentParser(description="Label a trace and add to dataset")
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--labeller", default="human")
    args = parser.parse_args()

    task = json.loads(Path(args.task_file).read_text())
    print(f"\nDataset version: {current_version()}")
    print(f"Task: {task.get('input', '')}")
    print(f"Expected output: {task.get('expected_output', '')}")
    label = input("\nLabel (pass/fail/needs_review) [pass]: ").strip() or "pass"

    item_id = add_item(args.trace_id, task, label=label, labeller=args.labeller)
    print(f"Saved item_id={item_id}")


if __name__ == "__main__":
    main()
