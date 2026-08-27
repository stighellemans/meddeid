"""Run one internal workflow adapter without mutating workflow state.

The parent workflow process owns state transitions and output validation.  This
small entry point lets long-running internal adapters use the same detached
runner as component commands.
"""

from __future__ import annotations

import argparse

from .workflow import WorkflowError, _run_internal, load_workflow


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m meddeid.workflow_internal")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    root, manifest = load_workflow(args.workspace)
    try:
        stage = next(item for item in manifest["stages"] if item["id"] == args.stage)
    except StopIteration:
        parser.error(f"unknown stage: {args.stage}")
    try:
        _run_internal(root, manifest, stage)
    except WorkflowError as exc:
        parser.exit(exc.code, f"workflow adapter: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
