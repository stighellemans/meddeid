"""Argument parsing and human-readable rendering for guided workflows."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from .workflow import (
    EXIT_NEEDS_INPUT,
    MANIFEST_FILENAME,
    WorkflowError,
    configure_workflow,
    doctor,
    explain_workflow,
    guide_text,
    initialize_workflow,
    parse_set_arguments,
    prompt_unresolved,
    render_stage_action,
    run_next,
    run_stage,
    workflow_status,
)
from .workflow_templates import list_guide_groups, list_templates


def add_parsers(sub: argparse._SubParsersAction) -> None:
    guide = sub.add_parser("guide", help="choose from six common MedDeID goals")
    guide.add_argument("--json", action="store_true", help="print workflow choices as JSON")

    start = sub.add_parser("start", help="create a workflow through the simplified guide")
    start.add_argument("workspace", nargs="?", type=Path)
    start.add_argument("--workflow", choices=[item["id"] for item in list_templates()])
    start.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    start.add_argument("--non-interactive", action="store_true")
    start.add_argument("--details", action="store_true", help="show the technical stage table")

    simple_status = sub.add_parser("status", help="show concise workflow progress")
    simple_status.add_argument("workspace", nargs="?", type=Path)
    simple_status.add_argument("--details", action="store_true", help="show every stage and reason")
    simple_status.add_argument("--json", action="store_true")

    simple_next = sub.add_parser("next", help="answer the next decision or run one stage")
    simple_next.add_argument("workspace", nargs="?", type=Path)
    simple_next.add_argument("--yes", action="store_true", help="confirm an expensive or external stage")
    simple_next.add_argument("--detach", action="store_true", help="detach a supported automated stage")
    simple_next.add_argument("--details", action="store_true", help="show every stage and reason")

    doctor_parser = sub.add_parser("doctor", help="check prerequisites for one workflow")
    doctor_parser.add_argument("--workflow", choices=[item["id"] for item in list_templates()])
    doctor_parser.add_argument("--json", action="store_true")

    workflow = sub.add_parser("workflow", help="create, explain, resume, and run guided workflows")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)

    workflow_sub.add_parser("list", help="list available workflow templates")

    init = workflow_sub.add_parser("init", help="initialize an explicit, resumable workflow")
    init.add_argument("type", choices=[item["id"] for item in list_templates()])
    init.add_argument("workspace", type=Path)
    init.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    init.add_argument("--non-interactive", action="store_true")

    configure = workflow_sub.add_parser("configure", help="answer or change workflow decisions")
    configure.add_argument("workspace", type=Path)
    configure.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    configure.add_argument("--reason", help="auditable reason for a protocol change")
    configure.add_argument("--yes", action="store_true", help="archive invalidated completed outputs")

    status = workflow_sub.add_parser("status", help="show validated stage status and the next action")
    status.add_argument("workspace", type=Path)
    status.add_argument("--json", action="store_true")

    next_parser = workflow_sub.add_parser("next", help="run exactly one eligible stage")
    next_parser.add_argument("workspace", type=Path)
    next_parser.add_argument("--yes", action="store_true", help="confirm an expensive or external stage")
    next_parser.add_argument("--detach", action="store_true", help="detach a supported automated stage")

    run = workflow_sub.add_parser("run", help="inspect or run one named stage")
    run.add_argument("workspace", type=Path)
    run.add_argument("stage")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--yes", action="store_true")
    run.add_argument("--detach", action="store_true")

    explain = workflow_sub.add_parser("explain", help="explain why each stage is included or excluded")
    explain.add_argument("workspace", type=Path)
    explain.add_argument("--json", action="store_true")


def _print_templates() -> None:
    for item in list_templates():
        print(f"{item['id']:<20} {item['summary']}")


def _print_status(payload: dict[str, Any]) -> None:
    print(f"{payload['template']['title']} — {payload['workspace']}")
    for stage in payload["stages"]:
        print(
            f"  {stage['state']:<14} {stage['requirement']:<11} "
            f"{stage['id']:<24} {stage['message']}"
        )
    if payload["complete"]:
        print("\nWorkflow complete. All applicable outputs validate.")
    elif payload.get("next"):
        item = payload["next"]
        print(f"\nNext: {item['id']} ({item['state']})")


def _print_simple_status(payload: dict[str, Any], workspace: Path) -> None:
    active = [
        stage for stage in payload["stages"]
        if stage["state"] not in {"skipped", "not_applicable"}
    ]
    completed = sum(stage["state"] == "completed" for stage in active)
    print(payload["template"]["title"])
    print(f"{completed} of {len(active)} stages complete")
    if payload["complete"]:
        print("\nComplete. All included outputs validate.")
    elif payload.get("next"):
        item = payload["next"]
        labels = {
            "needs_input": "Decision needed before",
            "blocked": "Blocked at",
            "failed": "Failed at",
            "running": "Currently running",
            "ready": "Next",
        }
        print(f"\n{labels.get(item['state'], 'Next')}: {item['title']}")
        if item["state"] in {"blocked", "failed"}:
            print(item["message"])
        else:
            print(f"Run: meddeid next {shlex.quote(str(workspace))}")
    excluded = [
        stage for stage in payload["stages"]
        if stage["state"] in {"skipped", "not_applicable"}
        and stage.get("simple_exclusion")
    ]
    if excluded:
        print("\nNot included:")
        for stage in excluded:
            message = stage["message"]
            reason = message.split("; ", 1)[1] if "; " in message else "not selected"
            print(f"  - {stage['simple_label']} — {reason}")
    print("\nMore detail: meddeid status --details " + shlex.quote(str(workspace)))


def _print_rendered(payload: dict[str, Any]) -> None:
    if payload.get("kind") == "internal":
        print(f"internal adapter: {payload['name']}")
        if payload.get("description"):
            print(f"  {payload['description']}")
    for command in payload.get("commands", []):
        for key, value in sorted(command.get("env", {}).items()):
            print(f"{key}={shlex.quote(value)} \\")
        print(shlex.join(command["argv"]))
    if payload.get("url"):
        print(f"opens: {payload['url']}")


def _choose_workflow_interactively() -> str:
    groups = list_guide_groups()
    templates = {item["id"]: item for item in list_templates()}
    while True:
        print(guide_text(include_instructions=False))
        raw = input("Choose a goal number, or enter a workflow ID: ").strip()
        if raw == "?":
            print("Choose the outcome you want; the next menu will ask only relevant details.\n")
            continue
        if raw in templates:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(groups):
            group = groups[int(raw) - 1]
        else:
            group = next((item for item in groups if item["id"] == raw), None)
        if group is None:
            print("Please choose one of the six goal numbers.", file=sys.stderr)
            continue
        options = group["workflows"]
        if len(options) == 1:
            return options[0]["id"]
        print(f"\n{group['title']} — what best matches your situation?\n")
        for index, option in enumerate(options, 1):
            print(f"  {index}. {option['title']}")
            print(f"     {option['summary']}")
        while True:
            selected = input(f"Choose 1-{len(options)}, or ? for help: ").strip()
            if selected == "?":
                print("This selects the workflow structure only. Optional stages are decided later.\n")
                continue
            if selected in templates and any(item["id"] == selected for item in options):
                return selected
            if selected.isdigit() and 1 <= int(selected) <= len(options):
                return options[int(selected) - 1]["id"]
            print("Please choose one of the listed options.", file=sys.stderr)


def _interactive_guide() -> None:
    selected = _choose_workflow_interactively()
    print("\nCreate it with:")
    print(f"  meddeid start WORKSPACE --workflow {selected}")


def _prompt_workspace(template_id: str) -> Path:
    default = Path.cwd() / f"meddeid-{template_id}-workflow"
    raw = input(f"Where should this workflow be saved? [{default}]: ").strip()
    return Path(raw).expanduser() if raw else default


def _resolve_workspace(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_FILENAME).is_file():
            return candidate
    raise WorkflowError(
        "could not find workflow.json here or in a parent directory; "
        "run this inside a workflow or pass its path"
    )


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "guide":
        if args.json:
            print(json.dumps({"goals": list_guide_groups(), "workflows": list_templates()}, indent=2))
        elif sys.stdin.isatty():
            _interactive_guide()
        else:
            print(guide_text())
        return 0

    if args.command == "start":
        interactive = not args.non_interactive and sys.stdin.isatty()
        template_id = args.workflow
        if template_id is None:
            if not interactive:
                raise WorkflowError(
                    "non-interactive start requires --workflow TYPE; "
                    "use `meddeid workflow list` to see advanced workflow IDs"
                )
            template_id = _choose_workflow_interactively()
        workspace = args.workspace
        if workspace is None:
            if not interactive:
                raise WorkflowError("non-interactive start requires a workspace path")
            workspace = _prompt_workspace(template_id)
        values = parse_set_arguments(args.set)
        initialize_workflow(
            workspace,
            template_id,
            values=values,
            interactive=interactive,
            simple=not args.details,
        )
        payload = workflow_status(workspace)
        resolved = Path(workspace).expanduser().resolve()
        if args.details:
            _print_status(payload)
        else:
            _print_simple_status(payload, resolved)
        if (
            args.non_interactive
            and payload.get("next")
            and payload["next"].get("state") == "needs_input"
        ):
            missing = payload["next"].get("missing_decisions", [])
            settings = " ".join(f"--set {key}=VALUE" for key in missing)
            print(
                "\nResolve the unanswered decision(s) with:\n"
                f"  meddeid workflow configure {shlex.quote(str(resolved))} {settings}",
                file=sys.stderr,
            )
            return EXIT_NEEDS_INPUT
        return 0

    if args.command == "status":
        workspace = _resolve_workspace(args.workspace)
        payload = workflow_status(workspace)
        if args.json:
            print(json.dumps(payload, indent=2))
        elif args.details:
            _print_status(payload)
        else:
            _print_simple_status(payload, workspace)
        return 0

    if args.command == "next":
        workspace = _resolve_workspace(args.workspace)
        payload = run_next(
            workspace,
            interactive=sys.stdin.isatty(),
            yes=args.yes,
            detach=args.detach,
        )
        if args.details:
            _print_status(payload)
        else:
            _print_simple_status(payload, workspace)
        return 0

    if args.command == "doctor":
        payload = doctor(args.workflow)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Python {payload['python']['version']}: {'ok' if payload['python']['supported'] else 'unsupported'}")
            for item in payload["tools"]:
                print(f"  {'ok' if item['available'] else 'missing':<7} {item['tool']}")
            missing_packages = [name for name, version in payload["packages"].items() if version is None]
            if missing_packages:
                print("\nMissing optional packages: " + ", ".join(missing_packages))
                print("Install the research path with: python -m pip install 'meddeid[research]'")
        return 0 if payload["ready"] else 1

    if args.workflow_command == "list":
        _print_templates()
        return 0

    if args.workflow_command == "init":
        values = parse_set_arguments(args.set)
        initialize_workflow(
            args.workspace,
            args.type,
            values=values,
            interactive=not args.non_interactive and sys.stdin.isatty(),
        )
        payload = workflow_status(args.workspace)
        _print_status(payload)
        if (
            args.non_interactive
            and payload.get("next")
            and payload["next"].get("state") == "needs_input"
        ):
            missing = payload["next"].get("missing_decisions", [])
            settings = " ".join(f"--set {key}=VALUE" for key in missing)
            print(
                "\nResolve the unanswered decision(s) with:\n"
                f"  meddeid workflow configure {shlex.quote(str(args.workspace.resolve()))} {settings}",
                file=sys.stderr,
            )
            return EXIT_NEEDS_INPUT
        print(f"\nResume with: meddeid workflow next {shlex.quote(str(args.workspace.resolve()))}")
        return 0

    if args.workflow_command == "configure":
        values = parse_set_arguments(args.set)
        if not values:
            if not sys.stdin.isatty():
                raise WorkflowError("configure needs --set KEY=VALUE in non-interactive use")
            changed = prompt_unresolved(args.workspace)
            if not changed:
                raise WorkflowError("no unresolved decisions; provide --set KEY=VALUE to change one")
            print("Answered: " + ", ".join(changed))
        else:
            result = configure_workflow(
                args.workspace,
                values=values,
                reason=args.reason,
                yes=args.yes,
            )
            print("Changed: " + ", ".join(result["changed"]))
            if result["impacted"]:
                print("Recomputed stages: " + ", ".join(result["impacted"]))
            if result["archived"]:
                print(f"Archived invalidated outputs: {result['archived']}")
        _print_status(workflow_status(args.workspace))
        return 0

    if args.workflow_command == "status":
        payload = workflow_status(args.workspace)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            _print_status(payload)
        return 0

    if args.workflow_command == "next":
        payload = run_next(
            args.workspace,
            interactive=sys.stdin.isatty(),
            yes=args.yes,
            detach=args.detach,
        )
        _print_status(payload)
        return 0

    if args.workflow_command == "run":
        if args.dry_run:
            _print_rendered(render_stage_action(args.workspace, args.stage))
        else:
            _print_status(run_stage(args.workspace, args.stage, yes=args.yes, detach=args.detach))
        return 0

    if args.workflow_command == "explain":
        payload = explain_workflow(args.workspace)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in payload:
                print(f"{item['id']} [{item['state']}]\n  Why: {item['why']}\n  Decision: {item['reason']}\n")
        return 0

    raise WorkflowError(f"unsupported workflow command: {args.workflow_command}")
