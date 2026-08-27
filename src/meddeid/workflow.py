"""Guided, resumable workflow engine used by :mod:`meddeid.cli`.

The engine is intentionally dependency-light.  It coordinates released
component executables and browser containers without importing component
internals, and it stores only privacy-safe state transitions in its event log.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from meddeid_core import BERT_ENTITY_LABELS, ProfileRef, validate_record
import meddeid_core

from .workflow_templates import (
    WORKFLOW_CONTRACT,
    get_template,
    list_guide_groups,
    list_templates,
)
from .workflow_actions import (
    ActionContext,
    ActionRegistry,
    ArtifactValidatorRegistry,
)

MANIFEST_FILENAME = "workflow.json"
STATE_DIRECTORY = ".workflow"

EXIT_INVALID = 2
EXIT_NEEDS_INPUT = 3
EXIT_BLOCKED = 4
EXIT_CONFIRMATION = 5
EXIT_FAILED = 6

TERMINAL_SUCCESS_STATES = {"completed", "skipped", "not_applicable"}
ALL_STATES = {
    "pending",
    "needs_input",
    "ready",
    "running",
    "completed",
    "blocked",
    "failed",
    "skipped",
    "not_applicable",
}

BROWSER_IMAGES = {
    "annotate": (
        "ghcr.io/stighellemans/meddeid-annotate:0.1.0@"
        "sha256:72f3e0fa0935da41e635e668573ec9c434cc3e8e1ef97bc793917bdfe6a7b78d"
    ),
    "curate": (
        "ghcr.io/stighellemans/meddeid-curate:0.1.0@"
        "sha256:8b3dde675cadc81f42a7fc34917d7b472c1556d14bc3acd1babf5bee8699875b"
    ),
    "subannotate": (
        "ghcr.io/stighellemans/meddeid-subannotate:0.1.0@"
        "sha256:d7da6967cb29b6cf8377458959dca84626a9c0e157320b42fe8815f49e880c87"
    ),
}


class WorkflowError(RuntimeError):
    """A user-actionable workflow failure with a stable CLI exit code."""

    def __init__(self, message: str, *, code: int = EXIT_INVALID) -> None:
        super().__init__(message)
        self.code = code


class WorkflowPause(WorkflowError):
    """A resumable human stage stopped without failing its work."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code=EXIT_NEEDS_INPUT)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _atomic_json(path: Path, payload: object, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    if mode is not None:
        os.chmod(temporary, mode)
    os.replace(temporary, path)


def _installed_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in (
        "meddeid",
        "meddeid-core",
        "meddeid-data",
        "meddeid-eval",
        "meddeid-training",
        "meddeid-language-en",
        "meddeid-language-nl",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(_sha256_file(child).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkflowError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise WorkflowError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise WorkflowError(f"{path}: JSONL is empty")
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def manifest_path(workspace: Path | str) -> Path:
    return Path(workspace).expanduser().resolve() / MANIFEST_FILENAME


def _state_dir(workspace: Path) -> Path:
    return workspace / STATE_DIRECTORY


def _event(
    kind: str,
    *,
    stage: str | None = None,
    keys: Iterable[str] = (),
    detail: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"at": _utc_now(), "kind": kind}
    if stage:
        payload["stage"] = stage
    if keys:
        payload["decision_keys"] = sorted(set(keys))
    if detail:
        payload["detail"] = detail
    return payload


def load_workflow(workspace: Path | str) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    path = root / MANIFEST_FILENAME
    if not path.is_file():
        raise WorkflowError(
            f"missing {path}; initialize it with: meddeid workflow init TYPE {shlex.quote(str(root))}"
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"failed to read {path}: {exc}") from exc
    if manifest.get("contract_version") != WORKFLOW_CONTRACT:
        raise WorkflowError(
            f"unsupported workflow contract {manifest.get('contract_version')!r}; expected {WORKFLOW_CONTRACT}"
        )
    stage_ids = [stage.get("id") for stage in manifest.get("stages", [])]
    if not stage_ids or len(stage_ids) != len(set(stage_ids)):
        raise WorkflowError(f"{path}: stage identifiers are missing or duplicated")
    return root, manifest


def save_workflow(workspace: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _atomic_json(workspace / MANIFEST_FILENAME, manifest)


def _decision_spec(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    for spec in manifest.get("decision_schema", []):
        if spec.get("key") == key:
            return spec
    raise WorkflowError(f"workflow has no decision named {key!r}")


def _condition_refs(condition: dict[str, Any] | None) -> set[str]:
    if not condition:
        return set()
    if "decision" in condition:
        return {str(condition["decision"])}
    refs: set[str] = set()
    for key in ("all", "any"):
        for child in condition.get(key, []):
            refs.update(_condition_refs(child))
    if "not" in condition:
        refs.update(_condition_refs(condition["not"]))
    return refs


def evaluate_condition(
    condition: dict[str, Any] | None, decisions: dict[str, Any]
) -> bool | None:
    """Evaluate the small declarative condition language.

    ``None`` means that one or more referenced decisions are unanswered.
    """

    if not condition:
        return True
    if "decision" in condition:
        value = decisions.get(str(condition["decision"]))
        if value is None:
            return None
        if "eq" in condition:
            return value == condition["eq"]
        if "gt" in condition:
            try:
                return int(value) > int(condition["gt"])
            except (TypeError, ValueError):
                return False
        if "truthy" in condition:
            return bool(value) is bool(condition["truthy"])
        raise WorkflowError(f"unsupported workflow condition: {condition}")
    if "not" in condition:
        result = evaluate_condition(condition["not"], decisions)
        return None if result is None else not result
    if "all" in condition:
        results = [evaluate_condition(child, decisions) for child in condition["all"]]
        if any(result is False for result in results):
            return False
        return None if any(result is None for result in results) else True
    if "any" in condition:
        results = [evaluate_condition(child, decisions) for child in condition["any"]]
        if any(result is True for result in results):
            return True
        return None if any(result is None for result in results) else False
    raise WorkflowError(f"unsupported workflow condition: {condition}")


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("expected true/false")


def parse_decision_value(spec: dict[str, Any], raw: Any) -> Any:
    if raw is None:
        return None
    kind = spec.get("kind", "string")
    if kind == "bool":
        value = raw if isinstance(raw, bool) else _parse_bool(str(raw))
    elif kind == "int":
        value = int(raw)
        minimum = int(spec.get("minimum") or 1)
        if value < minimum:
            raise ValueError(f"expected an integer of at least {minimum}")
    elif kind == "profiles":
        values = raw if isinstance(raw, list) else str(raw).split(",")
        value = [
            ProfileRef.parse(str(item).strip()).identifier
            for item in values
            if str(item).strip()
        ]
        if not value:
            raise ValueError(
                "profiles must be comma-separated unversioned locales such as en-GB,en-US"
            )
    elif kind == "choice":
        value = str(raw).strip()
        if value not in spec.get("choices", []):
            raise ValueError(f"choose one of: {', '.join(spec.get('choices', []))}")
    elif kind == "path":
        value = str(Path(str(raw)).expanduser().resolve())
    else:
        value = str(raw).strip()
        if not value:
            raise ValueError("value must not be empty")
    return value


def parse_set_arguments(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise WorkflowError(f"invalid --set {item!r}; expected KEY=VALUE")
        key = key.strip()
        if key in parsed:
            raise WorkflowError(f"decision {key!r} was provided more than once")
        parsed[key] = value
    return parsed


def _prompt(spec: dict[str, Any]) -> Any:
    choices = spec.get("choices", [])
    default = spec.get("default")
    labels = spec.get("choice_labels", {})
    while True:
        if choices:
            print(f"\n{spec['prompt']}")
            for index, choice in enumerate(choices, 1):
                print(f"  {index}. {labels.get(choice, choice.replace('_', ' '))}")
            default_text = f", default {default}" if default is not None else ""
            raw = input(
                f"Choose 1-{len(choices)}{default_text}, or ? for help: "
            ).strip()
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                raw = choices[int(raw) - 1]
        else:
            if spec.get("kind") == "bool":
                default_text = ""
                if default is not None:
                    default_text = f", default {'yes' if default else 'no'}"
                prompt = f"{spec['prompt']} (y/n{default_text}, or ? for help): "
            else:
                default_text = f" [default: {default}]" if default is not None else ""
                prompt = f"{spec['prompt']}{default_text} [? for help]: "
            raw = input(prompt).strip()
        if raw == "?":
            print(f"Why this is asked: {spec['why']}")
            continue
        if not raw and default is not None:
            return default
        if not raw and not spec.get("required", True):
            return None
        try:
            return parse_decision_value(spec, raw)
        except (TypeError, ValueError) as exc:
            print(f"Invalid value: {exc}", file=sys.stderr)


def _spec_applies(spec: dict[str, Any], decisions: dict[str, Any]) -> bool | None:
    return evaluate_condition(spec.get("ask_when"), decisions)


def _simple_exclusion_visible(stage_id: str, states: dict[str, str]) -> bool:
    if stage_id in {"preannotate", "curate"}:
        return states.get("primary_review") not in {"skipped", "not_applicable"}
    return True


def initialize_workflow(
    workspace: Path | str,
    template_id: str,
    *,
    values: dict[str, str] | None = None,
    interactive: bool = False,
    simple: bool = False,
) -> dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise WorkflowError(
            f"workflow directory is not empty: {root}; use status/configure to resume or choose a new directory"
        )
    try:
        template = get_template(template_id)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    provided = values or {}
    known = {spec["key"] for spec in template["decisions"]}
    unknown = sorted(set(provided) - known)
    if unknown:
        raise WorkflowError(f"unknown workflow decision(s): {', '.join(unknown)}")
    decisions: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for spec in template["decisions"]:
        if spec["key"] in provided:
            try:
                decisions[spec["key"]] = parse_decision_value(
                    spec, provided[spec["key"]]
                )
            except (TypeError, ValueError) as exc:
                raise WorkflowError(f"{spec['key']}: {exc}") from exc
            sources[spec["key"]] = "command_line"
        elif spec.get("default") is not None:
            decisions[spec["key"]] = spec["default"]
            sources[spec["key"]] = "default"
        else:
            decisions[spec["key"]] = None
    if interactive:
        progress = True
        while progress:
            progress = False
            for spec in template["decisions"]:
                if (
                    spec.get("scope") != "protocol"
                    or decisions.get(spec["key"]) is not None
                    or spec["key"] in sources
                ):
                    continue
                applies = _spec_applies(spec, decisions)
                if applies is True:
                    decisions[spec["key"]] = _prompt(spec)
                    sources[spec["key"]] = "interactive_init"
                    progress = True
    now = _utc_now()
    manifest: dict[str, Any] = {
        "contract_version": WORKFLOW_CONTRACT,
        "template": {
            "id": template_id,
            "version": template["version"],
            "title": template["title"],
        },
        "created_at": now,
        "updated_at": now,
        "decisions": decisions,
        "decision_sources": sources,
        "decision_reasons": {},
        "decision_schema": template["decisions"],
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "packages": _installed_versions(),
            "browser_images": BROWSER_IMAGES,
        },
        "stages": [],
        "artifacts": {},
        "events": [_event("workflow_initialized", keys=provided.keys())],
    }
    for stage in template["stages"]:
        stage["execution"] = {"state": "pending", "attempts": 0}
        manifest["stages"].append(stage)
    if interactive:
        resolved_graph: list[tuple[dict[str, Any], str]] = []
        for stage in manifest["stages"]:
            applicable = evaluate_condition(stage.get("applicable_when"), decisions)
            enabled = evaluate_condition(stage.get("enabled_when"), decisions)
            if applicable is False:
                state = "not_applicable"
            elif applicable is None or enabled is None:
                state = "needs_input"
            elif enabled is False:
                state = "skipped"
            else:
                state = "included"
            resolved_graph.append((stage, state))
        if simple:
            print("\nYour workflow (no files written yet):")
            included = [
                stage
                for stage, state in resolved_graph
                if state in {"included", "needs_input"}
            ]
            for index, stage in enumerate(included, 1):
                print(f"  {index}. {stage['title']}")
            state_by_id = {stage["id"]: state for stage, state in resolved_graph}
            excluded = [
                stage
                for stage, state in resolved_graph
                if state in {"skipped", "not_applicable"}
                and stage.get("simple_label")
                and _simple_exclusion_visible(stage["id"], state_by_id)
            ]
            if excluded:
                print("\nNot included:")
                for stage in excluded:
                    print(f"  - {stage['simple_label']}")
            print(
                "\nUse `meddeid status --details` later to inspect every technical stage and reason."
            )
        else:
            print("\nResolved stage graph (no files written yet):")
            for stage, state in resolved_graph:
                print(f"  {state:<14} {stage['id']:<24} {stage['title']}")
        answer = input("Write this workflow manifest? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            raise WorkflowError(
                "workflow initialization cancelled before writing",
                code=EXIT_CONFIRMATION,
            )
    root.mkdir(parents=True, exist_ok=True)
    for relative in (STATE_DIRECTORY, "artifacts"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _atomic_text(root / ".gitignore", ".workflow/logs/\n.workflow/runs/\n")
    save_workflow(root, manifest)
    return manifest


def _stage_decision_refs(stage: dict[str, Any]) -> set[str]:
    refs = set(stage.get("decisions", []))
    refs.update(_condition_refs(stage.get("applicable_when")))
    refs.update(_condition_refs(stage.get("enabled_when")))
    action = stage.get("action", {})
    refs.update(option["decision"] for option in action.get("options", []))
    refs.update(option["decision"] for option in action.get("env_options", []))
    return refs


def _descendants(manifest: dict[str, Any], seeds: set[str]) -> set[str]:
    impacted = set(seeds)
    progress = True
    while progress:
        progress = False
        for stage in manifest["stages"]:
            if stage["id"] in impacted:
                continue
            if any(
                requirement in impacted for requirement in stage.get("requires", [])
            ):
                impacted.add(stage["id"])
                progress = True
    return impacted


def _format_context(root: Path, manifest: dict[str, Any]) -> dict[str, str]:
    context: dict[str, str] = {"workspace": str(root)}
    for key, value in manifest.get("decisions", {}).items():
        if value is None:
            continue
        context[key] = ",".join(value) if isinstance(value, list) else str(value)
    detailed = manifest.get("decisions", {}).get("detailed_evaluation") is True
    context["benchmark_gold"] = str(
        root
        / (
            "subannotation/evaluation-bundle/benchmark.jsonl"
            if detailed
            else "artifacts/authoritative-annotations.jsonl"
        )
    )
    context["adaptation_development"] = str(
        root / "artifacts" / "authoritative-annotations.jsonl"
        if manifest.get("decisions", {}).get("review_development") is True
        else manifest.get("decisions", {}).get("development", "")
    )
    context["adaptation_test_gold"] = str(
        root / "test-subannotation" / "evaluation-bundle" / "benchmark.jsonl"
        if detailed
        else manifest.get("decisions", {}).get("test_gold", "")
    )
    return context


class _StrictFormat(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def _render(value: str, context: dict[str, str]) -> str:
    try:
        return value.format_map(_StrictFormat(context))
    except KeyError as exc:
        raise WorkflowError(
            f"stage needs decision {exc.args[0]!r}; configure it with: "
            f"meddeid workflow configure WORKSPACE --set {exc.args[0]}=VALUE",
            code=EXIT_NEEDS_INPUT,
        ) from exc


def _output_paths(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> list[Path]:
    context = _format_context(root, manifest)
    paths: list[Path] = []
    for value in stage.get("outputs", []):
        rendered = Path(_render(value, context)).expanduser()
        paths.append(rendered if rendered.is_absolute() else root / rendered)
    return paths


def _outputs_match(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> tuple[bool, str | None]:
    recorded = stage.get("execution", {}).get("outputs", {})
    for path in _output_paths(root, manifest, stage):
        if not path.exists():
            return False, f"expected output is missing: {path}"
        if path.is_file() and path.stat().st_size == 0:
            return False, f"expected output is empty: {path}"
        key = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        expected = recorded.get(key)
        if expected and _artifact_digest(path) != expected:
            return False, f"completed output changed after validation: {path}"
    if stage.get("action", {}).get("name") == "start-deployment":
        try:
            service = json.loads(
                (root / "artifacts" / "service.json").read_text(encoding="utf-8")
            )
            os.kill(int(service["pid"]), 0)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False, "the recorded deployment service is no longer running"
    if stage.get("execution", {}).get("semantic_contract"):
        try:
            _validate_stage_artifacts(root, manifest, stage)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return False, f"semantic output validation failed: {exc}"
    return True, None


def _record_outputs(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> None:
    outputs: dict[str, str] = {}
    for path in _output_paths(root, manifest, stage):
        if not path.exists():
            raise WorkflowError(
                f"stage {stage['id']} did not create expected output: {path}",
                code=EXIT_BLOCKED,
            )
        if path.is_file() and path.stat().st_size == 0:
            raise WorkflowError(
                f"stage {stage['id']} created an empty output: {path}",
                code=EXIT_BLOCKED,
            )
        key = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
        digest = _artifact_digest(path)
        outputs[key] = digest
        manifest["artifacts"][f"{stage['id']}:{key}"] = {"path": key, "sha256": digest}
    try:
        validated = _validate_stage_artifacts(root, manifest, stage)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"stage {stage['id']} produced invalid semantic output: {exc}",
            code=EXIT_FAILED,
        ) from exc
    stage["execution"]["outputs"] = outputs
    if validated:
        stage["execution"]["semantic_contract"] = "meddeid.workflow-artifacts.v1"


def _record_stage_inputs(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> None:
    """Pin explicit path decisions before a stage uses them."""

    recorded: dict[str, dict[str, str]] = {}
    for key in stage.get("decisions", []):
        spec = _decision_spec(manifest, key)
        if (
            spec.get("kind") != "path"
            or spec.get("path_role", "input") != "input"
            or manifest["decisions"].get(key) is None
        ):
            continue
        path = Path(str(manifest["decisions"][key])).resolve()
        if not path.exists():
            raise WorkflowError(f"{key} does not exist: {path}", code=EXIT_BLOCKED)
        if path.is_dir() and (root == path or root.is_relative_to(path)):
            raise WorkflowError(
                f"workflow workspace {root} is inside input directory {path}; "
                "choose a separate workspace so workflow state cannot change a pinned input",
                code=EXIT_BLOCKED,
            )
        recorded[key] = {"path": str(path), "sha256": _artifact_digest(path)}
    stage["execution"]["inputs"] = recorded


def _stage_inputs_match(stage: dict[str, Any]) -> tuple[bool, str | None]:
    for key, item in stage.get("execution", {}).get("inputs", {}).items():
        path = Path(item["path"])
        if not path.exists():
            return False, f"input {key} is missing: {path}"
        if _artifact_digest(path) != item["sha256"]:
            return False, f"input {key} changed after this stage completed: {path}"
    return True, None


def _reconcile_detached(root: Path, manifest: dict[str, Any]) -> bool:
    changed = False
    for stage in manifest["stages"]:
        execution = stage.get("execution", {})
        if execution.get("state") != "running" or not execution.get("detached"):
            continue
        result_path = _state_dir(root) / "runs" / f"{stage['id']}.result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            execution.update(
                {
                    "state": "failed",
                    "finished_at": _utc_now(),
                    "message": f"detached result is unreadable: {exc}",
                }
            )
            manifest["events"].append(_event("stage_failed", stage=stage["id"]))
            changed = True
            continue
        if result.get("returncode") == 0:
            try:
                _record_outputs(root, manifest, stage)
            except WorkflowError as exc:
                execution.update(
                    {"state": "blocked", "message": str(exc), "finished_at": _utc_now()}
                )
            else:
                execution.update(
                    {
                        "state": "completed",
                        "finished_at": result.get("finished_at", _utc_now()),
                    }
                )
                manifest["events"].append(_event("stage_completed", stage=stage["id"]))
        else:
            execution.update(
                {
                    "state": "failed",
                    "returncode": result.get("returncode"),
                    "finished_at": result.get("finished_at", _utc_now()),
                    "message": f"detached command failed; see {execution.get('log')}",
                }
            )
            manifest["events"].append(_event("stage_failed", stage=stage["id"]))
        changed = True
    return changed


def _needed_decisions(manifest: dict[str, Any], stage: dict[str, Any]) -> list[str]:
    decisions = manifest.get("decisions", {})
    missing: set[str] = set()
    for condition in (stage.get("applicable_when"), stage.get("enabled_when")):
        if evaluate_condition(condition, decisions) is None:
            missing.update(
                key for key in _condition_refs(condition) if decisions.get(key) is None
            )
    for key in stage.get("decisions", []):
        spec = _decision_spec(manifest, key)
        applies = _spec_applies(spec, decisions)
        if applies is None:
            missing.update(
                ref
                for ref in _condition_refs(spec.get("ask_when"))
                if decisions.get(ref) is None
            )
        elif (
            applies is True
            and spec.get("required", True)
            and decisions.get(key) is None
        ):
            missing.add(key)
    return sorted(missing)


def _excluded_message(
    manifest: dict[str, Any], stage: dict[str, Any], *, skipped: bool
) -> str:
    condition = stage.get("enabled_when" if skipped else "applicable_when")
    refs = sorted(_condition_refs(condition))
    reasons = [
        manifest.get("decision_reasons", {}).get(key)
        for key in refs
        if manifest.get("decision_reasons", {}).get(key)
    ]
    base = (
        "user explicitly declined this optional stage"
        if skipped
        else "stage does not belong to the selected workflow design"
    )
    if reasons:
        return f"{base}; reason: {reasons[-1]}"
    context: list[str] = []
    for key in refs:
        value = manifest.get("decisions", {}).get(key)
        if value is None:
            continue
        spec = _decision_spec(manifest, key)
        label = spec.get("prompt", key).rstrip("?")
        if isinstance(value, bool):
            rendered = "yes" if value else "no"
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = spec.get("choice_labels", {}).get(
                str(value), str(value).replace("_", " ")
            )
        context.append(f"{label}: {rendered}")
    return f"{base}; {'; '.join(context)}" if context else base


def workflow_status(workspace: Path | str, *, persist: bool = True) -> dict[str, Any]:
    root, manifest = load_workflow(workspace)
    changed = _reconcile_detached(root, manifest)
    resolved: list[dict[str, Any]] = []
    states: dict[str, str] = {}
    messages: dict[str, str] = {}
    decisions = manifest.get("decisions", {})
    for stage in manifest["stages"]:
        stage_id = stage["id"]
        applicable = evaluate_condition(stage.get("applicable_when"), decisions)
        missing = _needed_decisions(manifest, stage)
        if applicable is None:
            state, message = "needs_input", f"needs decision(s): {', '.join(missing)}"
        elif applicable is False:
            state, message = "not_applicable", _excluded_message(
                manifest, stage, skipped=False
            )
        else:
            enabled = evaluate_condition(stage.get("enabled_when"), decisions)
            if enabled is None:
                state, message = (
                    "needs_input",
                    f"needs decision(s): {', '.join(missing)}",
                )
            elif enabled is False:
                state, message = "skipped", _excluded_message(
                    manifest, stage, skipped=True
                )
            else:
                execution = stage.get("execution", {})
                execution_state = execution.get("state", "pending")
                if execution_state == "completed":
                    matches, mismatch = _stage_inputs_match(stage)
                    if matches:
                        matches, mismatch = _outputs_match(root, manifest, stage)
                    state, message = (
                        ("completed", "validated output")
                        if matches
                        else ("blocked", mismatch or "output mismatch")
                    )
                elif execution_state in {"running", "failed", "blocked"}:
                    state, message = execution_state, execution.get(
                        "message", execution_state
                    )
                else:
                    dependency_states = [
                        states.get(dependency, "pending")
                        for dependency in stage.get("requires", [])
                    ]
                    if any(
                        value in {"failed", "blocked"} for value in dependency_states
                    ):
                        state, message = (
                            "blocked",
                            "a prerequisite failed or has invalid output",
                        )
                    elif not all(
                        value in TERMINAL_SUCCESS_STATES for value in dependency_states
                    ):
                        state, message = "pending", "waiting for an earlier stage"
                    elif missing:
                        state, message = (
                            "needs_input",
                            f"needs decision(s): {', '.join(missing)}",
                        )
                    else:
                        state, message = (
                            "ready",
                            "all prerequisites and decisions are satisfied",
                        )
        states[stage_id] = state
        messages[stage_id] = message
        resolved.append(
            {
                "id": stage_id,
                "title": stage["title"],
                "requirement": (
                    "optional"
                    if stage.get("enabled_when")
                    else "conditional" if stage.get("applicable_when") else "required"
                ),
                "state": state,
                "why": stage["why"],
                "message": message,
                "simple_label": stage.get("simple_label"),
                "missing_decisions": missing if state == "needs_input" else [],
            }
        )
    for item in resolved:
        item["simple_exclusion"] = bool(
            item.get("simple_label") and _simple_exclusion_visible(item["id"], states)
        )
    if changed and persist:
        save_workflow(root, manifest)
    next_stage = next(
        (
            item
            for item in resolved
            if item["state"] in {"needs_input", "ready", "blocked", "failed", "running"}
        ),
        None,
    )
    return {
        "contract_version": WORKFLOW_CONTRACT,
        "template": manifest["template"],
        "workspace": root.name,
        "stages": resolved,
        "next": next_stage,
        "complete": all(item["state"] in TERMINAL_SUCCESS_STATES for item in resolved),
    }


def configure_workflow(
    workspace: Path | str,
    *,
    values: dict[str, str],
    reason: str | None = None,
    yes: bool = False,
) -> dict[str, Any]:
    root, manifest = load_workflow(workspace)
    parsed: dict[str, Any] = {}
    for key, raw in values.items():
        spec = _decision_spec(manifest, key)
        try:
            parsed[key] = parse_decision_value(spec, raw)
        except (TypeError, ValueError) as exc:
            raise WorkflowError(f"{key}: {exc}") from exc
    changed_keys = {
        key for key, value in parsed.items() if manifest["decisions"].get(key) != value
    }
    if not changed_keys:
        return {"changed": [], "impacted": [], "archived": None}
    directly_impacted = {
        stage["id"]
        for stage in manifest["stages"]
        if changed_keys & _stage_decision_refs(stage)
    }
    impacted = _descendants(manifest, directly_impacted)
    completed = [
        stage
        for stage in manifest["stages"]
        if stage["id"] in impacted
        and stage.get("execution", {}).get("state") == "completed"
    ]
    if completed and not yes:
        names = ", ".join(stage["id"] for stage in completed)
        raise WorkflowError(
            f"changing {', '.join(sorted(changed_keys))} invalidates completed stage(s): {names}. "
            "Review the impact, then rerun with --yes to archive those outputs.",
            code=EXIT_CONFIRMATION,
        )
    archive_root: Path | None = None
    if completed:
        archive_root = _state_dir(root) / "archive" / _timestamp()
        for stage in completed:
            outputs = _output_paths(root, manifest, stage)
            if stage.get("action", {}).get("name") == "scaffold-language-package":
                outputs = [Path(str(manifest["decisions"]["output_dir"]))]
            for output in outputs:
                if not output.exists():
                    continue
                destination = (
                    archive_root / output.relative_to(root)
                    if output.is_relative_to(root)
                    else archive_root / "external" / stage["id"] / output.name
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(output), str(destination))
    for key, value in parsed.items():
        manifest["decisions"][key] = value
        manifest["decision_sources"][key] = "configure"
        if reason:
            manifest["decision_reasons"][key] = reason
    for stage in manifest["stages"]:
        if stage["id"] in impacted:
            stage["execution"] = {"state": "pending", "attempts": 0}
            for artifact_key in [
                key
                for key in manifest["artifacts"]
                if key.startswith(f"{stage['id']}:")
            ]:
                del manifest["artifacts"][artifact_key]
    manifest["events"].append(
        _event(
            "decisions_changed",
            keys=changed_keys,
            detail=(
                f"invalidated {len(impacted)} stage(s); archived completed outputs"
                if completed
                else f"invalidated {len(impacted)} stage(s)"
            ),
        )
    )
    save_workflow(root, manifest)
    return {
        "changed": sorted(changed_keys),
        "impacted": sorted(impacted),
        "archived": str(archive_root) if archive_root else None,
    }


def unresolved_decisions(workspace: Path | str) -> list[dict[str, Any]]:
    root, manifest = load_workflow(workspace)
    status = workflow_status(root)
    next_item = status.get("next")
    if not next_item or next_item.get("state") != "needs_input":
        return []
    return [
        _decision_spec(manifest, key) for key in next_item.get("missing_decisions", [])
    ]


def prompt_unresolved(workspace: Path | str) -> list[str]:
    root, manifest = load_workflow(workspace)
    changed: list[str] = []
    while True:
        specs = unresolved_decisions(root)
        if not specs:
            break
        for spec in specs:
            value = _prompt(spec)
            manifest["decisions"][spec["key"]] = value
            manifest["decision_sources"][spec["key"]] = "interactive_next"
            manifest["events"].append(_event("decision_answered", keys=[spec["key"]]))
            changed.append(spec["key"])
            save_workflow(root, manifest)
        if (workflow_status(root).get("next") or {}).get("state") != "needs_input":
            break
    return changed


def _suite_root(manifest: dict[str, Any]) -> Path:
    configured = manifest.get("decisions", {}).get("suite_root")
    candidates = []
    if configured:
        candidates.append(Path(str(configured)))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        if (candidate / "repos" / "meddeid-annotate").is_dir():
            return candidate.resolve()
    raise WorkflowError(
        "source runtime needs the suite checkout; configure --set suite_root=/path/to/meddeid-suite",
        code=EXIT_NEEDS_INPUT,
    )


def _available_port(start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise WorkflowError(
        f"no available localhost port found from {start} to {start + 49}",
        code=EXIT_BLOCKED,
    )


def _next_assignment(root: Path) -> Path:
    manifest_path_value = root / "assignments" / "assignment-manifest.json"
    if not manifest_path_value.is_file():
        raise WorkflowError(
            "assignment manifest does not exist; run the preparation stage first",
            code=EXIT_BLOCKED,
        )
    assignment_manifest = json.loads(manifest_path_value.read_text(encoding="utf-8"))
    for relative in assignment_manifest.get("assignments", []):
        path = root / relative
        if not _primary_complete(path):
            return path
    return root / assignment_manifest["assignments"][-1]


def _primary_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        rows = _read_jsonl(path)
    except WorkflowError:
        return False
    return all(
        row.get("annotated") is True
        and all(
            span.get("confirmed") is True
            for span in row.get("spans", [])
            if isinstance(span, dict)
        )
        for row in rows
    )


def _all_assignments_complete(root: Path) -> bool:
    path = root / "assignments" / "assignment-manifest.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("assignments")) and all(
        _primary_complete(root / item) for item in payload["assignments"]
    )


def _subannotations_complete(data_dir: Path) -> bool:
    path = data_dir / "subannotations.jsonl"
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        rows = _read_jsonl(path)
    except WorkflowError:
        return False
    return all(
        isinstance(row.get("items"), list)
        and all(item.get("status") == "confirmed" for item in row.get("items", []))
        for row in rows
    )


def _prepare_assignments(
    root: Path,
    manifest: dict[str, Any],
    *,
    benchmark: bool,
    source_override: Path | None = None,
) -> None:
    decisions = manifest["decisions"]
    if source_override is not None:
        source = source_override
    elif benchmark:
        source = Path(str(decisions["source"]))
    else:
        source = root / "project" / "artifacts" / "annotations.jsonl"
    rows = _read_jsonl(source)
    reviewer_count = int(decisions.get("reviewer_count") or 1)
    reset = (
        source_override is not None
        or benchmark
        or decisions.get("review_mode") == "blinded"
    )
    prepared: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        if reset:
            next_row["annotated"] = False
            next_spans = []
            for span in row.get("spans", []):
                next_span = {
                    key: value
                    for key, value in span.items()
                    if key
                    not in {
                        "subannotations",
                        "span_id",
                        "suggestionReview",
                        "_changeStatus",
                    }
                }
                next_span["confirmed"] = False
                next_spans.append(next_span)
            next_row["spans"] = next_spans
        prepared.append(next_row)
    assignments = []
    for index in range(1, reviewer_count + 1):
        relative = Path("assignments") / f"reviewer-{index}.jsonl"
        target = root / relative
        if target.exists():
            raise WorkflowError(f"refusing to overwrite existing assignment: {target}")
        _write_jsonl(target, prepared)
        assignments.append(relative.as_posix())
    assignment_manifest = {
        "contract_version": "meddeid.workflow-assignments.v1",
        "source_sha256": _sha256_file(source),
        "review_mode": decisions.get("review_mode"),
        "assignments": assignments,
    }
    _atomic_json(root / "assignments" / "assignment-manifest.json", assignment_manifest)


def _authoritative_source(root: Path, manifest: dict[str, Any]) -> Path:
    curated = root / "curation" / "exports" / "annotations.jsonl"
    if curated.is_file():
        return curated
    decisions = manifest["decisions"]
    if decisions.get("gold_policy") == "selected_reviewer":
        selected = str(decisions.get("selected_reviewer") or "").strip()
        if not selected:
            raise WorkflowError(
                "selected_reviewer is required when gold_policy=selected_reviewer",
                code=EXIT_NEEDS_INPUT,
            )
        name = selected if selected.endswith(".jsonl") else f"{selected}.jsonl"
        candidate = root / "assignments" / name
        if not candidate.is_file() and selected.isdigit():
            candidate = root / "assignments" / f"reviewer-{selected}.jsonl"
        if not candidate.is_file():
            raise WorkflowError(
                f"selected reviewer assignment does not exist: {candidate}",
                code=EXIT_BLOCKED,
            )
        return candidate
    first = root / "assignments" / "reviewer-1.jsonl"
    if first.is_file():
        return first
    source = manifest["decisions"].get("source")
    if source:
        return Path(str(source))
    raise WorkflowError(
        "could not resolve authoritative annotations", code=EXIT_BLOCKED
    )


def _package_authoritative(root: Path, manifest: dict[str, Any]) -> None:
    source = _authoritative_source(root, manifest)
    rows = _read_jsonl(source)
    decisions = manifest["decisions"]
    accepted_completed_source = (
        decisions.get("input_role")
        in {"existing_gold", "completed_annotations", "sealed_test"}
        and decisions.get("re_review") is not True
        and source == Path(str(decisions.get("source")))
    )
    rows_marked_complete = all(
        row.get("annotated") is True or row.get("completed") is True for row in rows
    )
    if not accepted_completed_source and not rows_marked_complete:
        raise WorkflowError(
            f"authoritative annotations are incomplete: {source}", code=EXIT_BLOCKED
        )
    target = root / "artifacts" / "authoritative-annotations.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    payload = {
        "manifest_version": "meddeid.authoritative-annotations.v1",
        "source": source.name,
        "source_sha256": _sha256_file(source),
        "annotations_sha256": _sha256_file(target),
        "counts": {
            "documents": len(rows),
            "spans": sum(len(row.get("spans", [])) for row in rows),
        },
        "gold_policy": decisions.get("gold_policy") or "existing_completed_source",
        "selection_rationale": decisions.get("selection_rationale"),
        "completion_evidence": (
            f"accepted input_role={decisions.get('input_role')} after validate_source"
            if accepted_completed_source
            else "all rows carry annotated=true or completed=true"
        ),
    }
    _atomic_json(
        root / "artifacts" / "authoritative-annotations.manifest.json", payload
    )


def _run_checked(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    try:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise WorkflowError(
            f"required command is unavailable: {argv[0]}; run meddeid doctor",
            code=EXIT_BLOCKED,
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(
            f"command failed with exit code {exc.returncode}: {shlex.join(argv)}",
            code=EXIT_FAILED,
        ) from exc


def _export_subannotation_bundle(
    root: Path,
    manifest: dict[str, Any],
    *,
    data_dir: Path,
    source: Path,
) -> None:
    if not _subannotations_complete(data_dir):
        raise WorkflowError("subannotation review is not complete", code=EXIT_BLOCKED)
    runtime = manifest["decisions"].get("runtime")
    environment = os.environ.copy()
    environment.update(
        {
            "MEDDEID_DATA_DIR": str(data_dir),
            "MEDDEID_ANNOTATIONS_PATH": str(source),
        }
    )
    if runtime == "source":
        repo = _suite_root(manifest) / "repos" / "meddeid-subannotate"
        _run_checked(["npm", "--prefix", str(repo), "run", "bundle"], env=environment)
    elif runtime == "docker":
        _run_checked(
            [
                "docker",
                "run",
                "--rm",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "-e",
                "MEDDEID_DATA_DIR=/app/data",
                "-e",
                "MEDDEID_ANNOTATIONS_PATH=/input/annotations.jsonl",
                "-v",
                f"{data_dir}:/app/data",
                "-v",
                f"{source}:/input/annotations.jsonl:ro",
                BROWSER_IMAGES["subannotate"],
                "node",
                "scripts/export-bundle.js",
            ]
        )
    else:
        raise WorkflowError(
            "subannotation export requires runtime=docker or runtime=source",
            code=EXIT_NEEDS_INPUT,
        )


def _lock_adaptation_roles(root: Path, decisions: dict[str, Any]) -> None:
    development_path = Path(str(decisions["development"]))
    test_path = Path(str(decisions["test_gold"]))
    development = _read_jsonl(development_path)
    test = _read_jsonl(test_path)
    development_ids = {str(row.get("document_id")) for row in development}
    test_ids = {str(row.get("document_id")) for row in test}
    overlap = sorted(development_ids & test_ids)
    if overlap:
        raise WorkflowError(
            f"development and sealed test overlap ({len(overlap)} document IDs; first: {overlap[0]})",
            code=EXIT_BLOCKED,
        )
    project = Path(str(decisions["project"]))
    if not project.is_dir():
        raise WorkflowError(
            f"project directory does not exist: {project}", code=EXIT_BLOCKED
        )
    _atomic_json(
        root / "artifacts" / "role-lock.json",
        {
            "contract_version": "meddeid.adaptation-role-lock.v1",
            "created_at": _utc_now(),
            "development": {
                "sha256": _sha256_file(development_path),
                "documents": len(development_ids),
            },
            "sealed_test": {
                "sha256": _sha256_file(test_path),
                "documents": len(test_ids),
            },
            "overlap_documents": 0,
        },
    )


def _deployment_preflight(root: Path, decisions: dict[str, Any]) -> None:
    if (
        decisions.get("deployment_target") == "organization"
        and decisions.get("tls_boundary") is not True
    ):
        raise WorkflowError(
            "organizational deployment is blocked until the authenticated TLS boundary is documented",
            code=EXIT_BLOCKED,
        )
    if shutil.which("meddeid-server") is None:
        raise WorkflowError(
            "meddeid-server is unavailable; install meddeid[server]", code=EXIT_BLOCKED
        )
    port = int(decisions["port"])
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise WorkflowError(
                f"localhost port {port} is unavailable", code=EXIT_BLOCKED
            ) from exc
    _atomic_json(
        root / "artifacts" / "deployment-preflight.json",
        {
            "contract_version": "meddeid.deployment-preflight.v1",
            "created_at": _utc_now(),
            "target": decisions["deployment_target"],
            "localhost_only": True,
            "port": port,
            "model": decisions["model"],
            "revision": decisions.get("revision"),
            "language_profile": decisions.get("language_profile"),
            "checks": {"executable": True, "port_available": True},
        },
    )


def _start_deployment(root: Path, decisions: dict[str, Any]) -> None:
    service_path = root / "artifacts" / "service.json"
    if service_path.is_file():
        existing = json.loads(service_path.read_text(encoding="utf-8"))
        try:
            os.kill(int(existing["pid"]), 0)
        except (OSError, KeyError, TypeError, ValueError):
            pass
        else:
            return
    executable = shutil.which("meddeid-server")
    if not executable:
        raise WorkflowError(
            "meddeid-server is unavailable; install meddeid[server]", code=EXIT_BLOCKED
        )
    environment = os.environ.copy()
    environment.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(decisions["port"]),
            "MEDDEID_MODEL": str(decisions["model"]),
            "MEDDEID_DEVICE": str(decisions["device"]),
            "MEDDEID_ACCESS_LOG": "false",
        }
    )
    if decisions.get("revision"):
        environment["MEDDEID_REVISION"] = str(decisions["revision"])
    if decisions.get("language_profile"):
        environment["MEDDEID_LANGUAGE_PROFILE"] = str(decisions["language_profile"])
    log_path = _state_dir(root) / "logs" / "deployment-service.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [executable],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    time.sleep(1)
    if process.poll() is not None:
        raise WorkflowError(
            f"service exited during startup; see {log_path}", code=EXIT_FAILED
        )
    _atomic_json(
        service_path,
        {
            "contract_version": "meddeid.local-service.v1",
            "started_at": _utc_now(),
            "pid": process.pid,
            "url": f"http://127.0.0.1:{decisions['port']}",
            "log": str(log_path.relative_to(root)),
        },
    )


def _verify_deployment_health(root: Path) -> None:
    service_path = root / "artifacts" / "service.json"
    if not service_path.is_file():
        raise WorkflowError("service record is missing", code=EXIT_BLOCKED)
    service = json.loads(service_path.read_text(encoding="utf-8"))
    url = f"{service['url']}/health"
    last_error: Exception | None = None
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if response.status == 200:
                _atomic_json(
                    root / "artifacts" / "health.json",
                    {
                        "contract_version": "meddeid.deployment-health.v1",
                        "checked_at": _utc_now(),
                        "url": url,
                        "response": payload,
                    },
                )
                return
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise WorkflowError(
        f"service did not become healthy: {last_error}", code=EXIT_BLOCKED
    )


def _audit_language_resources(output: Path, *, require_files: bool = False) -> None:
    resources = output / "resources"
    files = (
        [path for path in resources.rglob("*") if path.is_file()]
        if resources.is_dir()
        else []
    )
    if require_files and not files:
        raise WorkflowError(
            f"resource mode requires files under {resources}; add licensed, provenance-recorded resources and retry",
            code=EXIT_BLOCKED,
        )
    source_locks = (
        sorted((output / "sources").glob("*.json"))
        if (output / "sources").is_dir()
        else []
    )
    required_source_fields = {
        "publisher",
        "release",
        "url",
        "retrieved_at",
        "licence",
        "attribution",
        "geographic_scope",
        "source_sha256",
        "adapter_version",
        "output_categories",
    }
    problems: list[str] = []
    sources: list[dict[str, Any]] = []
    for path in source_locks:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"{path.name}: invalid JSON: {exc}")
            continue
        missing = sorted(required_source_fields - set(payload))
        if missing:
            problems.append(f"{path.name}: missing {', '.join(missing)}")
        if not isinstance(payload.get("output_categories"), list):
            problems.append(f"{path.name}: output_categories must be a list")
        source_hash = str(payload.get("source_sha256") or "")
        if source_hash and not re.fullmatch(r"[0-9a-f]{64}", source_hash):
            problems.append(f"{path.name}: source_sha256 must be lowercase SHA-256")
        sources.append({"path": path.relative_to(output).as_posix(), "record": payload})
    report = {
        "contract_version": "meddeid.language-resources-audit.v1",
        "created_at": _utc_now(),
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(files)
        ],
        "sources": sources,
        "problems": problems,
        "passed": not problems,
        "network_used": False,
    }
    _atomic_json(output / "resources-audit.json", report)
    if problems:
        raise WorkflowError(
            "language resource audit failed: " + "; ".join(problems[:5]),
            code=EXIT_FAILED,
        )


def _test_language_conformance(output: Path, decisions: dict[str, Any]) -> None:
    failures: list[str] = []
    required_files = (
        output / "pyproject.toml",
        output / "package.json",
        output / "LICENSE",
        output / "NOTICE",
        output / "README.md",
    )
    for required in required_files:
        if not required.is_file():
            failures.append(f"missing {required.name}")
    package_name = str(decisions["package_name"])
    module_name = package_name.replace("-", "_")
    profiles = [ProfileRef.parse(value) for value in decisions["profiles"]]
    profile_module = output / "src" / module_name / "profiles.py"
    if not profile_module.is_file():
        failures.append(
            f"missing Python profile provider {profile_module.relative_to(output)}"
        )
    try:
        pyproject = (output / "pyproject.toml").read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(f"invalid pyproject.toml: {exc}")
        pyproject = ""
    if '[project.entry-points."meddeid.language_profiles"]' not in pyproject:
        failures.append("pyproject.toml does not register meddeid.language_profiles")
    try:
        package = json.loads((output / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"invalid package.json: {exc}")
        package = {}
    selections = {
        item.get("selection")
        for item in package.get("meddeid", {}).get("subannotationProfiles", [])
    }
    expected_profiles = {profile.selection for profile in profiles}
    missing_profiles = sorted(expected_profiles - selections)
    if missing_profiles:
        failures.append("missing profile registrations: " + ", ".join(missing_profiles))
    exports = package.get("exports", {})
    for profile in profiles:
        export_name = f"./subannotation/{profile.selection}"
        module_path = exports.get(export_name)
        if not module_path:
            failures.append(f"missing JavaScript export {export_name}")
        elif not (output / str(module_path).removeprefix("./")).is_file():
            failures.append(f"JavaScript export {export_name} points to a missing file")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", profile.selection).strip("-")
        manifest_path = (
            output
            / "src"
            / module_name
            / "resources"
            / "profiles"
            / slug
            / "manifest.json"
        )
        if not manifest_path.is_file():
            failures.append(f"missing profile manifest for {profile.identifier}")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"invalid manifest for {profile.identifier}: {exc}")
            continue
        if manifest.get("profile_id") != profile.selection:
            failures.append(f"manifest profile mismatch for {profile.identifier}")
    if not failures and profile_module.is_file():
        script = (
            "import json,sys; "
            f"sys.path.insert(0, {str((output / 'src').resolve())!r}); "
            f"from {module_name}.profiles import get_profile; "
            f"profiles={ [profile.selection for profile in profiles]!r}; "
            "assert all(get_profile(item).profile_id == item for item in profiles); "
            "print(json.dumps(profiles))"
        )
        try:
            environment = os.environ.copy()
            core_source = Path(str(meddeid_core.__file__)).resolve().parent.parent
            environment["PYTHONPATH"] = os.pathsep.join(
                (str((output / "src").resolve()), str(core_source))
            )
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=output,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(f"Python profile discovery failed: {exc.stderr.strip()}")
    if not failures and shutil.which("node"):
        for profile in profiles:
            module_path = exports[f"./subannotation/{profile.selection}"]
            try:
                subprocess.run(
                    [
                        "node",
                        "--input-type=module",
                        "-e",
                        (
                            f"const m=await import({('./' + str(module_path).removeprefix('./'))!r});"
                            f"if(m.profile.selection!=={profile.selection!r})process.exit(2);"
                        ),
                    ],
                    cwd=output,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                failures.append(
                    f"JavaScript profile import failed for {profile.selection}: {exc.stderr.strip()}"
                )
    package_checks: list[dict[str, Any]] = []
    if not failures and importlib.util.find_spec("build") is not None:
        with tempfile.TemporaryDirectory(prefix="meddeid-language-build-") as directory:
            try:
                subprocess.run(
                    [sys.executable, "-m", "build", "--outdir", directory, str(output)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                artifacts = sorted(Path(directory).iterdir())
                package_checks.append(
                    {
                        "kind": "python-build",
                        "artifacts": [path.name for path in artifacts],
                    }
                )
                if not any(path.suffix == ".whl" for path in artifacts) or not any(
                    path.name.endswith(".tar.gz") for path in artifacts
                ):
                    failures.append("Python build did not create both wheel and sdist")
            except subprocess.CalledProcessError as exc:
                failures.append(f"Python package build failed: {exc.stderr.strip()}")
    if not failures and shutil.which("npm"):
        try:
            result = subprocess.run(
                ["npm", "pack", "--dry-run", "--json"],
                cwd=output,
                check=True,
                capture_output=True,
                text=True,
            )
            package_checks.append(
                {"kind": "npm-pack", "result": json.loads(result.stdout)}
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            failures.append(f"npm package dry-run failed: {exc}")
    report = {
        "contract_version": "meddeid.language-conformance.v1",
        "created_at": _utc_now(),
        "profiles": [profile.identifier for profile in profiles],
        "checks": package_checks,
        "passed": not failures,
        "failures": failures,
    }
    _atomic_json(output / "conformance.json", report)
    if failures:
        raise WorkflowError(
            "language package conformance failed: " + "; ".join(failures),
            code=EXIT_FAILED,
        )


def _generate_synthetic(root: Path, decisions: dict[str, Any]) -> None:
    profile_refs = [ProfileRef.parse(value) for value in decisions["profiles"]]
    profiles = [profile.selection for profile in profile_refs]
    count = int(decisions["count"])
    seed = int(decisions["seed"])
    mode = decisions["generation_mode"]
    generated: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    scratch = _state_dir(root) / "generated"
    scratch.mkdir(parents=True, exist_ok=True)
    if mode == "remote":
        if decisions.get("allow_remote") is not True:
            raise WorkflowError(
                "remote generation was selected but not authorized", code=EXIT_BLOCKED
            )
        if set(profiles) != {"en-GB", "en-US"} or count % 2:
            raise WorkflowError(
                "remote English production currently requires en-GB,en-US and an even count",
                code=EXIT_BLOCKED,
            )
        production = scratch / "english-production"
        command = [
            "meddeid-english-production",
            "generate-batch",
            "--output-dir",
            str(production),
            "--batch-index",
            "0",
            "--batch-size",
            str(count),
            "--seed",
            str(seed),
            "--resume",
        ]
        if decisions.get("paid_model_review") is True:
            if decisions.get("reviewer_provider") != "openai":
                raise WorkflowError(
                    "paid model review requires reviewer_provider=openai",
                    code=EXIT_NEEDS_INPUT,
                )
            command.append("--model-review")
        _run_checked(command)
        generated = _read_jsonl(production / "batches" / "batch-01" / "documents.jsonl")
        usage_path = production / "batches" / "batch-01" / "usage-attempts.jsonl"
        ledger = (
            _read_jsonl(usage_path)
            if usage_path.is_file() and usage_path.stat().st_size
            else []
        )
    else:
        base, remainder = divmod(count, len(profiles))
        for index, profile in enumerate(profiles):
            profile_count = base + (1 if index < remainder else 0)
            if profile_count == 0:
                continue
            output = scratch / f"profile-{index + 1}.jsonl"
            if not output.is_file():
                _run_checked(
                    [
                        "meddeid-data",
                        "generate",
                        "--language-profile",
                        profile,
                        "--count",
                        str(profile_count),
                        "--seed",
                        str(seed + index),
                        "--output",
                        str(output),
                    ]
                )
            rows = _read_jsonl(output)
            if len(rows) != profile_count:
                raise WorkflowError(
                    f"{profile} generated {len(rows)} documents; expected {profile_count}",
                    code=EXIT_FAILED,
                )
            generated.extend(rows)
            ledger.append(
                {
                    "contract_version": "meddeid.production-attempt.v1",
                    "mode": "local",
                    "profile": profile,
                    "documents": len(rows),
                    "seed": seed + index,
                    "external_calls": 0,
                    "estimated_cost_usd": 0.0,
                    "outcome": "accepted",
                }
            )
    _write_jsonl(root / "artifacts" / "generated.jsonl", generated)
    _write_jsonl(
        root / "artifacts" / "usage-ledger.jsonl",
        ledger or [{"mode": mode, "events": 0}],
    )


def _validate_synthetic(root: Path, decisions: dict[str, Any]) -> None:
    source = root / "artifacts" / "generated.jsonl"
    rows = _read_jsonl(source)
    ids = [str(row.get("document_id") or "") for row in rows]
    problems: list[str] = []
    if len(rows) != int(decisions["count"]):
        problems.append(f"expected {decisions['count']} documents, found {len(rows)}")
    if len(ids) != len(set(ids)):
        problems.append("document IDs are not unique")
    allowed_profiles = {
        ProfileRef.parse(value).selection for value in decisions["profiles"]
    }
    for row in rows:
        document_id = str(row.get("document_id") or "unknown")
        text = row.get("text")
        if not isinstance(text, str) or not text:
            problems.append(f"{row.get('document_id')}: empty text")
            continue
        for issue in validate_record(row, strict_taxonomy=True):
            problems.append(f"{document_id}: {issue}")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        raw_profile = metadata.get("generation_profile") or metadata.get("lang")
        try:
            profile = ProfileRef.parse(str(raw_profile)).selection
        except ValueError as exc:
            problems.append(f"{document_id}: invalid profile: {exc}")
            profile = ""
        if profile not in allowed_profiles:
            problems.append(
                f"{document_id}: profile {profile!r} is outside the workflow"
            )
        for span in row.get("spans", []):
            begin, end = span.get("begin"), span.get("end")
            if (
                not isinstance(begin, int)
                or not isinstance(end, int)
                or not 0 <= begin < end <= len(text)
            ):
                problems.append(f"{row.get('document_id')}: invalid span offsets")
                break
            label = str(span.get("label") or "")
            if label == "Anonymize_Other":
                problems.append(f"{document_id}: forbidden Anonymize_Other span")
            elif label not in BERT_ENTITY_LABELS:
                problems.append(
                    f"{document_id}: generated label {label!r} is not allowed"
                )
    diversity: dict[str, Any]
    try:
        from meddeid_data.corpus_quality import (
            CorpusDiversityContract,
            audit_corpus_diversity,
        )

        families = sorted(
            {
                str((row.get("metadata") or {}).get("document_type") or "unknown")
                for row in rows
            }
        )
        diversity = audit_corpus_diversity(
            rows,
            contract=CorpusDiversityContract(
                expected_documents=int(decisions["count"]),
                profiles=tuple(sorted(allowed_profiles)),
                document_families=tuple(families),
                allowed_labels=(tuple(BERT_ENTITY_LABELS) if len(rows) >= 500 else ()),
                forbidden_labels=("Anonymize_Other",),
                require_balanced_profile_family_cells=(
                    len(rows) >= len(allowed_profiles) * max(1, len(families))
                ),
                near_duplicate_distance=3,
            ),
        )
        problems.extend(diversity["failures"])
    except ImportError as exc:
        raise WorkflowError(
            "synthetic quality requires meddeid-data; install meddeid[contributor]",
            code=EXIT_BLOCKED,
        ) from exc
    _atomic_json(
        root / "artifacts" / "quality-report.json",
        {
            "contract_version": "meddeid.synthetic-quality.v1",
            "created_at": _utc_now(),
            "documents": len(rows),
            "allowed_labels": list(BERT_ENTITY_LABELS),
            "forbidden_generated_labels": ["Anonymize_Other"],
            "diversity": diversity,
            "passed": not problems,
            "problems": list(dict.fromkeys(problems)),
        },
    )
    if problems:
        raise WorkflowError(
            "synthetic quality gate failed: " + "; ".join(problems[:5]),
            code=EXIT_FAILED,
        )


def _review_synthetic(root: Path) -> None:
    rows = _read_jsonl(root / "artifacts" / "generated.jsonl")
    report_path = root / "artifacts" / "review-report.jsonl"
    existing = (
        _read_jsonl(report_path)
        if report_path.is_file() and report_path.stat().st_size
        else []
    )

    def digest(row: dict[str, Any]) -> str:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    decisions = {str(row["document_id"]): row for row in existing}
    pending = [
        row
        for row in rows
        if decisions.get(str(row["document_id"]), {}).get("decision")
        not in {"accept", "reject"}
        or decisions.get(str(row["document_id"]), {}).get("document_sha256")
        != digest(row)
    ]
    if pending and not sys.stdin.isatty():
        raise WorkflowPause(
            "synthetic document review needs an interactive terminal; rerun this stage in a terminal. "
            "Existing per-document decisions are preserved."
        )
    for index, row in enumerate(pending, 1):
        document_id = str(row["document_id"])
        print(f"\n[{index}/{len(pending)}] {document_id}\n{row['text']}\n")
        while True:
            choice = input("Decision (accept/reject/quit): ").strip().lower()
            if choice in {"accept", "reject", "quit"}:
                break
        if choice == "quit":
            raise WorkflowPause("synthetic review paused; rerun the stage to continue")
        notes = input("Notes (optional): ").strip()
        decisions[document_id] = {
            "contract_version": "meddeid.review-decision.v1",
            "document_id": document_id,
            "document_sha256": digest(row),
            "decision": choice,
            "reviewer": os.environ.get("USER") or "personal-reviewer",
            "notes": notes,
            "reviewed_at": _utc_now(),
        }
        ordered = [
            decisions[str(item["document_id"])]
            for item in rows
            if str(item["document_id"]) in decisions
        ]
        _write_jsonl(report_path, ordered)


def _seal_synthetic(root: Path, decisions: dict[str, Any]) -> None:
    rows = _read_jsonl(root / "artifacts" / "generated.jsonl")
    reviews = _read_jsonl(root / "artifacts" / "review-report.jsonl")
    by_id = {str(item["document_id"]): item for item in reviews}

    def digest(row: dict[str, Any]) -> str:
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    if any(
        by_id.get(str(row["document_id"]), {}).get("decision")
        not in {"accept", "reject"}
        or by_id.get(str(row["document_id"]), {}).get("document_sha256") != digest(row)
        for row in rows
    ):
        raise WorkflowError(
            "every generated document needs a current content-bound review",
            code=EXIT_BLOCKED,
        )
    accepted = [
        row for row in rows if by_id[str(row["document_id"])]["decision"] == "accept"
    ]
    if len(accepted) < 2:
        raise WorkflowError(
            "at least two reviewed documents must pass to create non-empty development and benchmark splits",
            code=EXIT_BLOCKED,
        )
    benchmark_count = (
        300
        if len(accepted) == 7_000
        else min(len(accepted) - 1, max(1, round(len(accepted) * 0.2)))
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in accepted:
        metadata = row.get("metadata") or {}
        key = (
            ProfileRef.parse(
                str(metadata.get("generation_profile") or metadata.get("lang"))
            ).selection,
            str(metadata.get("document_type") or "unknown"),
        )
        grouped.setdefault(key, []).append(row)
    base, remainder = divmod(benchmark_count, len(grouped))
    benchmark_ids: set[str] = set()
    for index, key in enumerate(sorted(grouped)):
        take = base + (1 if index < remainder else 0)
        ranked = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                f"{decisions['seed']}|sealed-benchmark|{row['document_id']}".encode()
            ).hexdigest(),
        )
        if len(ranked) < take:
            raise WorkflowError(
                f"benchmark stratum {key} has fewer than {take} documents",
                code=EXIT_BLOCKED,
            )
        benchmark_ids.update(str(row["document_id"]) for row in ranked[:take])
    benchmark = [row for row in accepted if str(row["document_id"]) in benchmark_ids]
    development = [
        row for row in accepted if str(row["document_id"]) not in benchmark_ids
    ]
    development_path = root / "artifacts" / "development.jsonl"
    benchmark_path = root / "artifacts" / "benchmark.jsonl"
    _write_jsonl(development_path, development)
    _write_jsonl(benchmark_path, benchmark)
    _atomic_json(
        root / "artifacts" / "splits.manifest.json",
        {
            "contract_version": "meddeid.synthetic-splits.v1",
            "created_at": _utc_now(),
            "seed": int(decisions["seed"]),
            "allowed_labels": list(BERT_ENTITY_LABELS),
            "forbidden_generated_labels": ["Anonymize_Other"],
            "independently_authored": True,
            "accepted": len(accepted),
            "rejected": len(rows) - len(accepted),
            "development": {
                "documents": len(development),
                "sha256": _sha256_file(development_path),
            },
            "benchmark": {
                "documents": len(benchmark),
                "sha256": _sha256_file(benchmark_path),
            },
        },
    )


def _validate_model_checkpoint(root: Path, decisions: dict[str, Any]) -> None:
    selected = Path(str(decisions["checkpoint"])).expanduser().resolve()
    if not selected.exists():
        raise WorkflowError(f"checkpoint does not exist: {selected}", code=EXIT_BLOCKED)
    if selected.is_dir():
        candidates = [
            selected / "checkpoints" / "best.pt",
            selected / "best.pt",
        ]
        checkpoint = next((path for path in candidates if path.is_file()), None)
        run_root = selected
    else:
        checkpoint = selected
        run_root = (
            selected.parent.parent
            if selected.parent.name == "checkpoints"
            else selected.parent
        )
    if checkpoint is None or not checkpoint.is_file():
        raise WorkflowError(
            f"could not resolve best.pt from checkpoint selection {selected}",
            code=EXIT_BLOCKED,
        )
    metadata_path = run_root / "train_metrics.json"
    if not metadata_path.is_file():
        raise WorkflowError(
            f"checkpoint lineage is missing {metadata_path}; select a complete training run",
            code=EXIT_BLOCKED,
        )
    try:
        run = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(
            f"invalid training metadata: {exc}", code=EXIT_FAILED
        ) from exc
    config = run.get("config")
    if not isinstance(config, dict):
        raise WorkflowError(
            "training metadata has no resolved config", code=EXIT_FAILED
        )
    profile_items = config.get("language_profiles") or config.get("language_profile")
    if not isinstance(profile_items, list):
        profile_items = [profile_items] if profile_items else []
    actual_profiles = {
        ProfileRef.parse(
            str(item.get("profile_id") if isinstance(item, dict) else item)
        ).selection
        for item in profile_items
    }
    expected_profiles = {
        ProfileRef.parse(value).selection for value in decisions["profiles"]
    }
    problems: list[str] = []
    if actual_profiles != expected_profiles:
        problems.append(
            f"training profiles {sorted(actual_profiles)} do not match requested {sorted(expected_profiles)}"
        )
    actual_encoder = str(config.get("base_encoder") or config.get("model_name") or "")
    if actual_encoder != str(decisions["base_encoder"]):
        problems.append(
            f"training base encoder {actual_encoder!r} does not match {decisions['base_encoder']!r}"
        )
    actual_revision = config.get("base_revision") or config.get("model_revision")
    if str(actual_revision or "") != str(decisions["base_revision"]):
        problems.append(
            f"training base revision {actual_revision!r} does not match {decisions['base_revision']!r}"
        )
    entity_labels = tuple(run.get("entity_labels") or ())
    if entity_labels != tuple(BERT_ENTITY_LABELS):
        problems.append("training entity label order does not equal BERT_ENTITY_LABELS")
    protocol = run.get("protocol") if isinstance(run.get("protocol"), dict) else {}
    benchmark_evaluations = int(protocol.get("benchmark_evaluations", 0) or 0)
    if benchmark_evaluations > 1:
        problems.append("training metadata reports more than one benchmark evaluation")
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = (
            payload.get("model_state_dict", payload)
            if isinstance(payload, dict)
            else {}
        )
        entity_head = (
            state.get("label_classifier.weight") if isinstance(state, dict) else None
        )
        bio_head = (
            state.get("bio_classifier.weight") if isinstance(state, dict) else None
        )
        if entity_head is None or int(entity_head.shape[0]) != len(BERT_ENTITY_LABELS):
            problems.append(
                "checkpoint entity head does not match the canonical 14 labels"
            )
        if bio_head is None or int(bio_head.shape[0]) != 3:
            problems.append("checkpoint BIO head does not contain O/B/I outputs")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        problems.append(f"checkpoint tensors could not be validated: {exc}")
    report = {
        "contract_version": "meddeid.checkpoint-validation.v1",
        "created_at": _utc_now(),
        "checkpoint_sha256": _artifact_digest(checkpoint),
        "checkpoint": str(checkpoint),
        "run_metadata": str(metadata_path),
        "run_metadata_sha256": _sha256_file(metadata_path),
        "profiles": sorted(actual_profiles),
        "base_encoder": actual_encoder,
        "base_revision": actual_revision,
        "entity_labels": list(entity_labels),
        "protocol": protocol,
        "problems": problems,
        "passed": not problems,
    }
    _atomic_json(root / "artifacts" / "checkpoint-validation.json", report)
    if problems:
        raise WorkflowError(
            "model checkpoint validation failed: " + "; ".join(problems[:5]),
            code=EXIT_FAILED,
        )


def _verify_model_interfaces(root: Path, decisions: dict[str, Any]) -> None:
    model = root / "artifacts" / "model"
    checks: list[dict[str, Any]] = []
    smoke_dir = _state_dir(root) / "model-interface-smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    note = smoke_dir / "note.txt"
    _atomic_text(note, "Patient Alex Example attended on 12 January 2026.\n")
    try:
        from .api import Deidentifier

        engine = Deidentifier.from_pretrained(
            model,
            device=(
                None if decisions.get("device") == "auto" else decisions.get("device")
            ),
            local_files_only=True,
        )
        try:
            for raw_profile in decisions["profiles"]:
                profile = ProfileRef.parse(raw_profile)
                result = engine(
                    note.read_text(encoding="utf-8"),
                    metadata={"lang": profile.selection},
                )
                if not isinstance(result.deid_text, str):
                    raise TypeError("Python API returned no de-identified text")
                checks.append(
                    {
                        "interface": "python-api",
                        "profile": profile.identifier,
                        "passed": True,
                    }
                )
        finally:
            engine.close()
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkflowError(
            f"Python API smoke test failed: {exc}", code=EXIT_FAILED
        ) from exc
    for raw_profile in decisions["profiles"]:
        profile_ref = ProfileRef.parse(raw_profile)
        profile = profile_ref.selection
        common = [
            "--model",
            str(model),
            "--quiet",
            "--offline",
            "--language-profile",
            profile,
        ]
        device_args = (
            []
            if decisions.get("device") == "auto"
            else ["--device", str(decisions["device"])]
        )
        command = [
            "meddeid",
            "model-info",
            "--model",
            str(model),
            "--quiet",
            "--language-profile",
            profile,
            *device_args,
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            json.loads(result.stdout)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as exc:
            raise WorkflowError(
                f"model-info failed for {profile}: {exc}", code=EXIT_FAILED
            ) from exc
        checks.append(
            {
                "interface": "model-info",
                "profile": profile_ref.identifier,
                "passed": True,
            }
        )
        slug = re.sub(r"[^a-z0-9]+", "-", profile.lower()).strip("-")
        text_output = smoke_dir / f"{slug}.txt"
        _run_checked(
            [
                "meddeid",
                "deidentify",
                str(note),
                "--output",
                str(text_output),
                *common,
                *device_args,
            ]
        )
        checks.append(
            {
                "interface": "single-cli",
                "profile": profile_ref.identifier,
                "passed": text_output.is_file(),
            }
        )
        batch_input = smoke_dir / f"{slug}.input.jsonl"
        batch_output = smoke_dir / f"{slug}.predictions.jsonl"
        _write_jsonl(
            batch_input,
            [
                {
                    "document_id": f"smoke-{slug}",
                    "text": note.read_text(encoding="utf-8").rstrip("\n"),
                    "spans": [],
                    "metadata": {"lang": ProfileRef.parse(profile).selection},
                }
            ],
        )
        _run_checked(
            [
                "meddeid",
                "batch",
                str(batch_input),
                "--output",
                str(batch_output),
                *common,
                *device_args,
            ]
        )
        checks.append(
            {
                "interface": "batch-cli",
                "profile": profile_ref.identifier,
                "passed": batch_output.is_file(),
            }
        )
    if len(decisions["profiles"]) > 1:
        ambiguous = subprocess.run(
            [
                "meddeid",
                "model-info",
                "--model",
                str(model),
                "--quiet",
                "--language-profile",
                "en",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if ambiguous.returncode == 0:
            raise WorkflowError(
                "multi-profile bundle incorrectly accepted bare en",
                code=EXIT_FAILED,
            )
        checks.append(
            {"interface": "bare-en-rejection", "profile": "en", "passed": True}
        )
    port = _available_port(8900)
    service_log = _state_dir(root) / "logs" / "model-interface-service.log"
    service_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "MEDDEID_MODEL": str(model),
            "MEDDEID_OFFLINE": "true",
            "MEDDEID_ACCESS_LOG": "false",
            "MEDDEID_DOCS_ENABLED": "false",
            "MEDDEID_UI_ENABLED": "false",
        }
    )
    if decisions.get("device") != "auto":
        environment["MEDDEID_DEVICE"] = str(decisions["device"])
    try:
        with service_log.open("ab") as log:
            process = subprocess.Popen(
                ["meddeid-server"],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        health_url = f"http://127.0.0.1:{port}/health"
        last_error: Exception | None = None
        for _ in range(60):
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        last_error = None
                        break
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(0.5)
        else:
            raise WorkflowError(
                f"service health check timed out: {last_error}", code=EXIT_FAILED
            )
        if process.poll() is not None:
            raise WorkflowError(
                f"service exited before becoming healthy; see {service_log}",
                code=EXIT_FAILED,
            )
        for raw_profile in decisions["profiles"]:
            profile_ref = ProfileRef.parse(raw_profile)
            profile = profile_ref.selection
            payload = json.dumps(
                {
                    "text": note.read_text(encoding="utf-8").rstrip("\n"),
                    "metadata": {"lang": profile},
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/deidentify",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result.get("deid_text"), str):
                raise WorkflowError(
                    f"service returned no de-identified text for {profile}",
                    code=EXIT_FAILED,
                )
            checks.append(
                {
                    "interface": "http-service",
                    "profile": profile_ref.identifier,
                    "passed": True,
                }
            )
    except FileNotFoundError as exc:
        raise WorkflowError(
            "meddeid-server is unavailable; install meddeid[contributor]",
            code=EXIT_BLOCKED,
        ) from exc
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise WorkflowError(
            f"service interface smoke test failed: {exc}", code=EXIT_FAILED
        ) from exc
    finally:
        if "process" in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    _atomic_json(
        root / "artifacts" / "model-verification.json",
        {
            "contract_version": "meddeid.model-interface-verification.v1",
            "created_at": _utc_now(),
            "checks": checks,
            "passed": True,
        },
    )


def _scaffold_language_package(output: Path, decisions: dict[str, Any]) -> None:
    """Create a runnable two-runtime language-package contribution boundary."""

    if output.exists() and any(output.iterdir()):
        raise WorkflowError(f"language package directory is not empty: {output}")
    package_name = str(decisions["package_name"])
    if not re.fullmatch(r"meddeid-language-[a-z0-9-]+", package_name):
        raise WorkflowError(
            "language package name must match meddeid-language-[a-z0-9-]+"
        )
    module_name = package_name.replace("-", "_")
    npm_suffix = package_name.removeprefix("meddeid-")
    npm_name = f"@meddeid/{npm_suffix}"
    profiles = [ProfileRef.parse(value) for value in decisions["profiles"]]
    profile_ids = [profile.selection for profile in profiles]
    package_root = output / "src" / module_name
    (package_root / "resources" / "profiles").mkdir(parents=True, exist_ok=True)
    (output / "js").mkdir(parents=True, exist_ok=True)
    (output / "tests").mkdir(parents=True, exist_ok=True)
    (output / "tests-js").mkdir(parents=True, exist_ok=True)
    (output / "sources").mkdir(parents=True, exist_ok=True)

    _atomic_text(
        package_root / "__init__.py",
        (
            '"""MedDeID language-profile package."""\n\n'
            "from .profiles import get_profile\n\n"
            '__version__ = "0.1.0"\n'
            '__all__ = ["get_profile"]\n'
        ),
    )
    profile_literal = repr(tuple(profile_ids))
    _atomic_text(
        package_root / "profiles.py",
        f'''"""Generated profile-provider boundary; replace identity rules with locale rules."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

from meddeid_core import LanguageProfile, ProfileRef

PROFILE_IDS = {profile_literal}


def _post_process(spans, **_kwargs):
    return list(spans)


def _manifest(profile_id: str):
    path = Path(__file__).parent / "resources" / "profiles" / profile_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_profile(profile_id: str) -> LanguageProfile:
    selection = ProfileRef.parse(profile_id).selection
    if selection not in PROFILE_IDS:
        raise ValueError(f"unsupported profile {{profile_id!r}}; choose one of: {{', '.join(PROFILE_IDS)}}")
    return LanguageProfile(
        profile_id=selection,
        language_tags=(selection,),
        post_process_spans=_post_process,
        resource_manifest_provider=partial(_manifest, selection),
    )
''',
    )
    _atomic_text(
        package_root / "resources.py",
        '''"""Reproducible resource command boundary for this language pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED_SOURCE_FIELDS = {
    "publisher", "release", "url", "retrieved_at", "licence", "attribution",
    "geographic_scope", "source_sha256", "adapter_version", "output_categories",
}


def audit(root: Path) -> list[str]:
    failures = []
    for path in sorted((root / "sources").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(REQUIRED_SOURCE_FIELDS - set(record))
        if missing:
            failures.append(f"{path.name}: missing {', '.join(missing)}")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "build", "audit", "diff"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--against", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "fetch":
        parser.error("implement an official-source adapter before enabling network fetch")
    if args.command == "build":
        parser.error("implement deterministic normalization before building runtime assets")
    failures = audit(root)
    if args.command == "diff" and args.against is None:
        parser.error("diff requires --against")
    if failures:
        print("\n".join(failures))
        return 1
    print(json.dumps({"passed": True, "source_locks": len(list((root / 'sources').glob('*.json')))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''',
    )
    exports: dict[str, str] = {}
    registrations: list[dict[str, str]] = []
    js_test_imports: list[str] = []
    for profile in profiles:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", profile.selection).strip("-")
        manifest = {
            "contract_version": "meddeid.language-profile.v1",
            "profile_id": profile.selection,
            "package_version": "0.1.0",
            "resource_manifest_version": "1",
            "language_tags": [profile.selection],
            "resources": [],
            "sources": [],
        }
        _atomic_json(
            package_root / "resources" / "profiles" / slug / "manifest.json",
            manifest,
        )
        js_path = f"./js/subannotation-{slug.lower()}.js"
        export_name = f"./subannotation/{profile.selection}"
        exports[export_name] = js_path
        registrations.append(
            {"selection": profile.selection, "module": f"{npm_name}{export_name[1:]}"}
        )
        _atomic_text(
            output / js_path.removeprefix("./"),
            f"""export const profile = Object.freeze({{
  selection: {json.dumps(profile.selection)},
  packageVersion: "0.1.0",
  resourceManifestVersion: "1",
}});

export function postProcessSpans({{ spans }}) {{
  return [...spans];
}}

export default Object.freeze({{ profile, postProcessSpans }});
""",
        )
        js_test_imports.append(
            f"  const m = await import('../{js_path.removeprefix('./')}');\n"
            f"  assert.equal(m.profile.selection, {json.dumps(profile.selection)});"
        )
    _atomic_text(
        output / "pyproject.toml",
        (
            '[build-system]\nrequires = ["setuptools>=68"]\nbuild-backend = "setuptools.build_meta"\n\n'
            f'[project]\nname = "{package_name}"\nversion = "0.1.0"\n'
            'requires-python = ">=3.10"\nlicense = "AGPL-3.0-only"\n'
            'dependencies = ["meddeid-core>=0.2,<0.3"]\n\n'
            '[project.entry-points."meddeid.language_profiles"]\n'
            f'{npm_suffix.replace("-", "_")} = "{module_name}.profiles:get_profile"\n\n'
            f'[project.scripts]\n{package_name}-resources = "{module_name}.resources:main"\n\n'
            '[tool.setuptools.packages.find]\nwhere = ["src"]\n\n'
            f'[tool.setuptools.package-data]\n{module_name} = ["resources/**/*.json", "resources/**/*.jsonl", "resources/**/*.md"]\n\n'
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        ),
    )
    _atomic_json(
        output / "package.json",
        {
            "name": npm_name,
            "version": "0.1.0",
            "type": "module",
            "license": "AGPL-3.0-only",
            "exports": exports,
            "meddeid": {"subannotationProfiles": registrations},
            "files": [
                "js/",
                f"src/{module_name}/resources/",
                "LICENSE",
                "NOTICE",
                "README.md",
            ],
            "scripts": {"test": "node --test tests-js/*.test.js"},
            "engines": {"node": ">=20"},
        },
    )
    _atomic_text(
        output / "tests" / "test_profiles.py",
        f"""from {module_name} import get_profile


def test_all_profiles_resolve():
    for profile_id in {profile_literal}:
        profile = get_profile(profile_id)
        assert profile.profile_id == profile_id
        assert profile.manifest()["resources"]["profile_id"] == profile_id
""",
    )
    _atomic_text(
        output / "tests-js" / "profiles.test.js",
        (
            "import assert from 'node:assert/strict';\n"
            "import test from 'node:test';\n\n"
            "test('all subannotation profiles resolve independently', async () => {\n"
            + "\n".join(js_test_imports)
            + "\n});\n"
        ),
    )
    source_license = Path(__file__).resolve().parents[2] / "LICENSE"
    _atomic_text(
        output / "LICENSE",
        (
            source_license.read_text(encoding="utf-8")
            if source_license.is_file()
            else "SPDX-License-Identifier: AGPL-3.0-only\n"
        ),
    )
    _atomic_text(
        output / "NOTICE",
        "No third-party resources are included by this scaffold. Add source-specific attribution before building resources.\n",
    )
    _atomic_text(
        output / "README.md",
        (
            f"# {package_name}\n\n"
            f"Scaffolded MedDeID profiles: {', '.join(profile.identifier for profile in profiles)}.\n\n"
            "Implement locale rules and official-source adapters, then run fetch, build, audit, diff, and shared conformance before release.\n"
        ),
    )
    _atomic_text(
        output / "sources" / "README.md",
        (
            "# Source locks\n\n"
            "Create one JSON lock per official source. Required fields: publisher, release, URL, retrieval date, licence, attribution, geographic scope, source SHA-256, adapter version, and output categories. Never commit prohibited raw or person-level data.\n"
        ),
    )
    _atomic_text(
        output / ".gitignore",
        "/.cache/\n/dist/\n/build/\n*.egg-info/\nnode_modules/\n*.tgz\n",
    )


def _require_passed_json(path: Path, *, keys: tuple[str, ...] = ("passed",)) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not any(payload.get(key) is True for key in keys):
        raise ValueError(f"{path} does not record a passing semantic gate")


def _build_artifact_validators() -> ArtifactValidatorRegistry:
    registry = ArtifactValidatorRegistry()

    def output_json(
        context: ActionContext, keys: tuple[str, ...] = ("passed",)
    ) -> None:
        outputs = _output_paths(context.root, context.manifest, context.stage)
        if not outputs:
            raise ValueError("semantic stage declares no output")
        _require_passed_json(outputs[0], keys=keys)

    registry.register(
        "audit-language-resources",
        lambda context: output_json(context, ("passed",)),
    )
    registry.register(
        "test-language-conformance",
        lambda context: output_json(context, ("passed",)),
    )
    registry.register(
        "validate-synthetic-quality",
        lambda context: output_json(context, ("passed",)),
    )
    registry.register(
        "validate-model-checkpoint",
        lambda context: output_json(context, ("passed",)),
    )
    registry.register(
        "verify-model-interfaces",
        lambda context: output_json(context, ("passed",)),
    )

    def validate_splits(context: ActionContext) -> None:
        manifest_path = context.root / "artifacts" / "splits.manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(payload.get("allowed_labels", ())) != tuple(BERT_ENTITY_LABELS):
            raise ValueError("synthetic split manifest does not pin BERT_ENTITY_LABELS")
        if "Anonymize_Other" not in payload.get("forbidden_generated_labels", ()):
            raise ValueError("synthetic split manifest does not forbid Anonymize_Other")

    registry.register("seal-synthetic-splits", validate_splits)
    return registry


_ARTIFACT_VALIDATORS = _build_artifact_validators()


def _validate_stage_artifacts(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> bool:
    name = str(stage.get("action", {}).get("name") or "")
    return _ARTIFACT_VALIDATORS.validate(
        name, ActionContext(root=root, manifest=manifest, stage=stage)
    )


def _build_onboarding_actions() -> ActionRegistry:
    registry = ActionRegistry()
    registry.register(
        "scaffold-language-package",
        lambda context: _scaffold_language_package(
            Path(str(context.decisions["output_dir"])), context.decisions
        ),
    )

    def audit_resources(context: ActionContext) -> None:
        decisions = context.decisions
        if (
            decisions.get("resource_mode") == "remote"
            and decisions.get("allow_remote") is not True
        ):
            raise WorkflowError(
                "remote resource retrieval was not authorized", code=EXIT_BLOCKED
            )
        _audit_language_resources(
            Path(str(decisions["output_dir"])), require_files=True
        )

    registry.register("audit-language-resources", audit_resources)
    registry.register(
        "test-language-conformance",
        lambda context: _test_language_conformance(
            Path(str(context.decisions["output_dir"])), context.decisions
        ),
    )
    registry.register(
        "generate-synthetic-corpus",
        lambda context: _generate_synthetic(context.root, context.decisions),
    )
    registry.register(
        "validate-synthetic-quality",
        lambda context: _validate_synthetic(context.root, context.decisions),
    )
    registry.register(
        "review-synthetic-documents",
        lambda context: _review_synthetic(context.root),
    )
    registry.register(
        "seal-synthetic-splits",
        lambda context: _seal_synthetic(context.root, context.decisions),
    )
    registry.register(
        "validate-model-checkpoint",
        lambda context: _validate_model_checkpoint(context.root, context.decisions),
    )
    registry.register(
        "verify-model-interfaces",
        lambda context: _verify_model_interfaces(context.root, context.decisions),
    )
    return registry


_ONBOARDING_ACTIONS = _build_onboarding_actions()


def _run_internal(root: Path, manifest: dict[str, Any], stage: dict[str, Any]) -> None:
    name = stage["action"]["name"]
    decisions = manifest["decisions"]
    context = ActionContext(root=root, manifest=manifest, stage=stage)
    if _ONBOARDING_ACTIONS.handles(name):
        _ONBOARDING_ACTIONS.execute(name, context)
        return
    if name == "record-stage":
        return
    if name == "inspect-input":
        source = Path(str(decisions.get("source") or ""))
        if not source.is_file():
            raise WorkflowError(
                f"input file does not exist: {source}", code=EXIT_BLOCKED
            )
        return
    if name == "prepare-assignments":
        _prepare_assignments(root, manifest, benchmark=False)
        return
    if name == "prepare-benchmark-assignments":
        _prepare_assignments(root, manifest, benchmark=True)
        return
    if name == "prepare-adaptation-assignments":
        _prepare_assignments(
            root,
            manifest,
            benchmark=False,
            source_override=Path(str(decisions["development"])),
        )
        return
    if name == "preannotate-assignments":
        assignment_manifest = json.loads(
            (root / "assignments" / "assignment-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        device = decisions.get("device")
        for relative in assignment_manifest["assignments"]:
            target = root / relative
            temporary = target.with_suffix(".predictions.jsonl")
            command = ["meddeid", "batch", str(target), "--output", str(temporary)]
            if device and device != "auto":
                command.extend(["--device", str(device)])
            _run_checked(command)
            os.replace(temporary, target)
        _atomic_json(
            root / "artifacts" / "preannotation.json",
            {
                "contract_version": "meddeid.preannotation.v1",
                "created_at": _utc_now(),
                "model": decisions.get("model"),
                "assignments": [
                    {"path": relative, "sha256": _sha256_file(root / relative)}
                    for relative in assignment_manifest["assignments"]
                ],
            },
        )
        return
    if name == "package-authoritative":
        _package_authoritative(root, manifest)
        return
    if name == "export-subannotation-bundle":
        _export_subannotation_bundle(
            root,
            manifest,
            data_dir=root / "subannotation",
            source=root / "artifacts" / "authoritative-annotations.jsonl",
        )
        return
    if name == "export-test-subannotation-bundle":
        _export_subannotation_bundle(
            root,
            manifest,
            data_dir=root / "test-subannotation",
            source=Path(str(decisions["test_gold"])),
        )
        return
    if name == "validate-evaluation-inputs":
        gold = _read_jsonl(Path(str(decisions["gold"])))
        predictions = _read_jsonl(Path(str(decisions["predictions"])))
        gold_map = {str(row.get("document_id")): row.get("text") for row in gold}
        prediction_map = {
            str(row.get("document_id")): row.get("text") for row in predictions
        }
        if gold_map != prediction_map:
            raise WorkflowError(
                "gold and predictions do not contain identical document IDs and text",
                code=EXIT_BLOCKED,
            )
        return
    if name == "run-stability-analysis":
        config = Path(str(decisions["stability_config"]))
        _run_checked(["meddeid-eval", "stability", "run", "--config", str(config)])
        try:
            from meddeid_eval.stability.config import load_config

            output = load_config(config).output_dir
        except (ImportError, OSError, TypeError, ValueError) as exc:
            raise WorkflowError(
                f"could not validate stability output: {exc}", code=EXIT_FAILED
            ) from exc
        required = [output / "stability_analysis.json", output / "stability_report.md"]
        missing = [
            str(path)
            for path in required
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise WorkflowError(
                "stability run did not create: " + ", ".join(missing), code=EXIT_FAILED
            )
        _atomic_json(
            root / "artifacts" / "stability.json",
            {
                "contract_version": "meddeid.stability-run.v1",
                "created_at": _utc_now(),
                "config_sha256": _sha256_file(config),
                "output_dir": str(output),
                "output_sha256": _artifact_digest(output),
            },
        )
        return
    if name == "export-trained-model":
        protocol = decisions.get("training_protocol")
        run = root / "runs" / ("refit" if protocol == "select_refit" else "fit")
        command = [
            "meddeid-train",
            "export",
            "--checkpoint",
            str(run / "checkpoints" / "best.pt"),
            "--run-metadata",
            str(run / "train_metrics.json"),
            "--output",
            str(root / "artifacts" / "model"),
        ]
        _run_checked(command)
        return
    if name == "smoke-model-bundle":
        command = [
            "meddeid",
            "model-info",
            "--model",
            str(root / "artifacts" / "model"),
            "--quiet",
        ]
        device = decisions.get("device")
        if device and device != "auto":
            command.extend(["--device", str(device)])
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout)
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as exc:
            raise WorkflowError(
                f"exported model smoke test failed: {exc}", code=EXIT_FAILED
            ) from exc
        _atomic_json(root / "artifacts" / "model-smoke.json", payload)
        return
    if name == "score-exported-model":
        test = root / "prepared" / "refit" / "test.jsonl"
        if not test.is_file():
            test = root / "prepared" / "fit" / "test.jsonl"
        predictions = root / "artifacts" / "model-predictions.jsonl"
        _run_checked(
            [
                "meddeid",
                "batch",
                str(test),
                "--model",
                str(root / "artifacts" / "model"),
                "--output",
                str(predictions),
            ]
        )
        _run_checked(
            [
                "meddeid-eval",
                "score",
                "--gold",
                str(test),
                "--predictions",
                str(predictions),
                "--output",
                str(root / "artifacts" / "model-metrics.json"),
                "--name",
                "adapted",
            ]
        )
        return
    if name == "lock-adaptation-roles":
        _lock_adaptation_roles(root, decisions)
        return
    if name == "compare-adaptation":
        gold = Path(_format_context(root, manifest)["adaptation_test_gold"])
        baseline_metrics = root / "artifacts" / "baseline-metrics.json"
        adapted_metrics = root / "artifacts" / "model-metrics.json"
        _run_checked(
            [
                "meddeid-eval",
                "score",
                "--gold",
                str(gold),
                "--predictions",
                str(root / "artifacts" / "baseline-predictions.jsonl"),
                "--output",
                str(baseline_metrics),
                "--name",
                "baseline",
            ]
        )
        plots = root / "artifacts" / "plots"
        _run_checked(
            [
                "meddeid-eval",
                "plot",
                "--scores",
                str(baseline_metrics),
                str(adapted_metrics),
                "--output-dir",
                str(plots),
            ]
        )
        _atomic_json(
            root / "artifacts" / "comparison.json",
            {
                "contract_version": "meddeid.adaptation-comparison.v1",
                "created_at": _utc_now(),
                "gold_sha256": _sha256_file(gold),
                "baseline_metrics_sha256": _sha256_file(baseline_metrics),
                "adapted_metrics_sha256": _sha256_file(adapted_metrics),
            },
        )
        return
    if name == "deployment-preflight":
        _deployment_preflight(root, decisions)
        return
    if name == "start-deployment":
        _start_deployment(root, decisions)
        return
    if name == "verify-deployment-health":
        _verify_deployment_health(root)
        return
    if name == "deployment-readiness-report":
        health = json.loads(
            (root / "artifacts" / "health.json").read_text(encoding="utf-8")
        )
        _atomic_json(
            root / "artifacts" / "deployment-readiness.json",
            {
                "contract_version": "meddeid.deployment-readiness.v1",
                "created_at": _utc_now(),
                "technical_checks_passed": True,
                "target": decisions["deployment_target"],
                "localhost_only": True,
                "health_contract": health.get("contract_version"),
                "organizational_approval_required": decisions["deployment_target"]
                == "organization",
            },
        )
        return
    raise WorkflowError(
        f"stage adapter {name!r} is not executable in this installation; use "
        f"meddeid workflow run {shlex.quote(str(root))} {stage['id']} --dry-run and meddeid doctor",
        code=EXIT_BLOCKED,
    )


def _browser_action(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> dict[str, Any]:
    action = stage["action"]
    app = action["app"]
    runtime = manifest["decisions"].get("runtime")
    if runtime not in {"docker", "source"}:
        raise WorkflowError(
            f"stage {stage['id']} needs runtime; configure it with: meddeid workflow configure {shlex.quote(str(root))} --set runtime=docker",
            code=EXIT_NEEDS_INPUT,
        )
    context = _format_context(root, manifest)
    if "{next_assignment}" in action["source"]:
        source = _next_assignment(root)
    else:
        source = Path(_render(action["source"], context))
    data_dir = (
        Path(_render(action["data_dir"], context)) if action.get("data_dir") else None
    )
    if data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
    ports = {
        "annotate": {"container": 8787, "source_api": 8787, "browser": 5180},
        "curate": {"container": 8793, "source_api": 8793, "browser": 5183},
        "subannotate": {"container": 8787, "source_api": 8787, "browser": 5181},
    }
    api_port = int(ports[app]["source_api"])
    browser_port = int(ports[app]["browser"])
    commands: list[dict[str, Any]] = []
    if runtime == "source":
        suite = _suite_root(manifest)
        repo = suite / "repos" / f"meddeid-{app}"
        if app == "subannotate":
            profiles = manifest["decisions"].get("profiles") or []
            profile_command = [
                "npm",
                "--prefix",
                str(repo),
                "run",
                "profile",
                "--",
                "set",
                *profiles,
            ]
            english = suite / "repos" / "meddeid-language-en" / "js"
            for profile in profiles:
                locale = profile.split("@", 1)[0]
                module = english / f"subannotation-{locale.lower()}.js"
                if module.is_file():
                    profile_command.extend(["--module", f"{profile}={module}"])
            commands.append({"argv": profile_command, "cwd": str(suite), "env": {}})
        env: dict[str, str] = {}
        if app in {"annotate", "subannotate"}:
            env["MEDDEID_ANNOTATIONS_PATH"] = str(source)
        if data_dir:
            if app == "curate":
                env["MEDDEID_CURATE_DATA_DIR"] = str(data_dir)
            else:
                env["MEDDEID_DATA_DIR"] = str(data_dir)
        commands.append(
            {
                "argv": ["npm", "--prefix", str(repo), "run", "dev"],
                "cwd": str(suite),
                "env": env,
            }
        )
        url = f"http://127.0.0.1:{browser_port}"
    else:
        api_port = _available_port(api_port)
        image = BROWSER_IMAGES[app]
        argv = [
            "docker",
            "run",
            "--rm",
            "-p",
            f"127.0.0.1:{api_port}:{ports[app]['container']}",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
        ]
        if app == "annotate":
            argv.extend(["-v", f"{source}:/app/data/annotations.jsonl:rw"])
        elif app == "subannotate":
            argv.extend(
                [
                    "-e",
                    "MEDDEID_ANNOTATIONS_PATH=/input/annotations.jsonl",
                    "-v",
                    f"{source}:/input/annotations.jsonl:ro",
                ]
            )
        if data_dir:
            argv.extend(["-v", f"{data_dir}:/app/data"])
        if app == "subannotate" and manifest["decisions"].get("profiles"):
            profile_argv = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{data_dir}:/app/data",
                image,
                "node",
                "scripts/configure-profile.js",
                "set",
                *manifest["decisions"]["profiles"],
            ]
            commands.append({"argv": profile_argv, "cwd": str(root), "env": {}})
        argv.append(image)
        commands.append({"argv": argv, "cwd": str(root), "env": {}})
        url = f"http://127.0.0.1:{api_port}"
    return {
        "commands": commands,
        "url": url,
        "app": app,
        "source": str(source),
        "data_dir": str(data_dir) if data_dir else None,
    }


def _internal_action_preview(
    root: Path, manifest: dict[str, Any], stage: dict[str, Any]
) -> dict[str, Any]:
    """Expose component commands used by an internal orchestration adapter."""

    name = stage["action"]["name"]
    decisions = manifest["decisions"]
    commands: list[dict[str, Any]] = []
    if name == "preannotate-assignments":
        assignment_path = root / "assignments" / "assignment-manifest.json"
        if assignment_path.is_file():
            payload = json.loads(assignment_path.read_text(encoding="utf-8"))
            for relative in payload.get("assignments", []):
                target = root / relative
                argv = [
                    "meddeid",
                    "batch",
                    str(target),
                    "--output",
                    str(target.with_suffix(".predictions.jsonl")),
                ]
                if decisions.get("device") not in {None, "auto"}:
                    argv.extend(["--device", str(decisions["device"])])
                commands.append({"argv": argv, "cwd": str(root), "env": {}})
    elif name == "export-trained-model":
        run = (
            root
            / "runs"
            / (
                "refit"
                if decisions.get("training_protocol") == "select_refit"
                else "fit"
            )
        )
        commands.append(
            {
                "argv": [
                    "meddeid-train",
                    "export",
                    "--checkpoint",
                    str(run / "checkpoints" / "best.pt"),
                    "--run-metadata",
                    str(run / "train_metrics.json"),
                    "--output",
                    str(root / "artifacts" / "model"),
                ],
                "cwd": str(root),
                "env": {},
            }
        )
    elif name == "score-exported-model":
        commands.append(
            {
                "argv": [
                    "meddeid",
                    "batch",
                    "PREPARED_TEST.jsonl",
                    "--model",
                    str(root / "artifacts" / "model"),
                    "--output",
                    str(root / "artifacts" / "model-predictions.jsonl"),
                ],
                "cwd": str(root),
                "env": {},
            }
        )
    elif name in {"export-subannotation-bundle", "export-test-subannotation-bundle"}:
        data_dir = root / (
            "subannotation"
            if name == "export-subannotation-bundle"
            else "test-subannotation"
        )
        if decisions.get("runtime") == "source":
            repo = _suite_root(manifest) / "repos" / "meddeid-subannotate"
            commands.append(
                {
                    "argv": ["npm", "--prefix", str(repo), "run", "bundle"],
                    "cwd": str(root),
                    "env": {"MEDDEID_DATA_DIR": str(data_dir)},
                }
            )
        else:
            commands.append(
                {
                    "argv": [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{data_dir}:/app/data",
                        BROWSER_IMAGES["subannotate"],
                        "node",
                        "scripts/export-bundle.js",
                    ],
                    "cwd": str(root),
                    "env": {},
                }
            )
    elif name == "generate-synthetic-corpus":
        if decisions.get("generation_mode") == "remote":
            commands.append(
                {
                    "argv": [
                        "meddeid-english-production",
                        "generate-batch",
                        "--output-dir",
                        str(_state_dir(root) / "generated" / "english-production"),
                        "--batch-index",
                        "0",
                        "--batch-size",
                        str(decisions.get("count", "COUNT")),
                        "--seed",
                        str(decisions.get("seed", "SEED")),
                        "--resume",
                    ],
                    "cwd": str(root),
                    "env": {},
                }
            )
        else:
            for profile in decisions.get("profiles") or ["PROFILE"]:
                commands.append(
                    {
                        "argv": [
                            "meddeid-data",
                            "generate",
                            "--language-profile",
                            profile,
                            "--count",
                            "PROFILE_COUNT",
                            "--output",
                            "PROFILE_OUTPUT.jsonl",
                        ],
                        "cwd": str(root),
                        "env": {},
                    }
                )
    elif name == "run-stability-analysis":
        commands.append(
            {
                "argv": [
                    "meddeid-eval",
                    "stability",
                    "run",
                    "--config",
                    str(decisions.get("stability_config", "STABILITY_CONFIG.yaml")),
                ],
                "cwd": str(root),
                "env": {},
            }
        )
    elif name == "start-deployment":
        environment = {
            "HOST": "127.0.0.1",
            "PORT": str(decisions.get("port", 8000)),
            "MEDDEID_MODEL": str(decisions.get("model", "MODEL")),
            "MEDDEID_DEVICE": str(decisions.get("device", "auto")),
        }
        if decisions.get("revision"):
            environment["MEDDEID_REVISION"] = str(decisions["revision"])
        if decisions.get("language_profile"):
            environment["MEDDEID_LANGUAGE_PROFILE"] = str(decisions["language_profile"])
        commands.append(
            {
                "argv": ["meddeid-server"],
                "cwd": str(root),
                "env": environment,
            }
        )
    return {
        "kind": "internal",
        "name": name,
        "description": stage["why"],
        "commands": commands,
    }


def render_stage_action(workspace: Path | str, stage_id: str) -> dict[str, Any]:
    root, manifest = load_workflow(workspace)
    try:
        stage = next(item for item in manifest["stages"] if item["id"] == stage_id)
    except StopIteration as exc:
        raise WorkflowError(f"workflow has no stage named {stage_id!r}") from exc
    action = stage["action"]
    if action["kind"] == "command":
        context = _format_context(root, manifest)
        argv = [_render(value, context) for value in action["argv"]]
        for option in action.get("options", []):
            value = manifest["decisions"].get(option["decision"])
            if option.get("boolean"):
                if value is True:
                    argv.append(option["flag"])
            elif value is not None and str(value) != "":
                argv.extend([option["flag"], str(value)])
        cleaned: list[str] = []
        index = 0
        while index < len(argv):
            if argv[index : index + 2] == ["--device", "auto"]:
                index += 2
                continue
            cleaned.append(argv[index])
            index += 1
        environment = {
            key: _render(value, context) for key, value in action.get("env", {}).items()
        }
        for option in action.get("env_options", []):
            value = manifest["decisions"].get(option["decision"])
            if value is not None and str(value) != "":
                environment[option["name"]] = str(value)
        return {
            "commands": [{"argv": cleaned, "cwd": str(root), "env": environment}],
            "kind": "command",
        }
    if action["kind"] == "browser":
        return {**_browser_action(root, manifest, stage), "kind": "browser"}
    return _internal_action_preview(root, manifest, stage)


def _human_stage_complete(
    root: Path, stage: dict[str, Any], rendered: dict[str, Any]
) -> bool:
    app = rendered.get("app")
    if app == "annotate":
        return _all_assignments_complete(root)
    if app == "curate":
        return (root / "curation" / "exports" / "manifest.json").is_file()
    if app == "subannotate":
        data_dir = Path(str(rendered.get("data_dir")))
        return _subannotations_complete(data_dir)
    return not stage.get("outputs") or all(
        path.exists() for path in _output_paths(root, {}, stage)
    )


def _run_rendered_foreground(
    rendered: dict[str, Any], *, browser: bool, interactive_process: bool = False
) -> int:
    commands = rendered.get("commands", [])
    for index, command in enumerate(commands):
        argv = command["argv"]
        environment = os.environ.copy()
        environment.update(command.get("env", {}))
        cwd = Path(command["cwd"]) if command.get("cwd") else None
        if (browser or interactive_process) and index == len(commands) - 1:
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    start_new_session=True,
                )
            except FileNotFoundError as exc:
                raise WorkflowError(
                    f"required command is unavailable: {argv[0]}; run meddeid doctor",
                    code=EXIT_BLOCKED,
                ) from exc
            time.sleep(0.8)
            if browser:
                webbrowser.open(rendered["url"])
            try:
                return process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                return 130
        _run_checked(argv, cwd=cwd, env=environment)
    return 0


def _execute_stage(
    root: Path,
    manifest: dict[str, Any],
    stage: dict[str, Any],
    *,
    detach: bool,
) -> None:
    execution = stage["execution"]
    rendered = render_stage_action(root, stage["id"])
    if detach and (not stage.get("allows_detach") or rendered["kind"] == "browser"):
        raise WorkflowError(f"stage {stage['id']} cannot run detached")
    if detach and rendered["kind"] != "internal" and len(rendered["commands"]) != 1:
        raise WorkflowError("multi-command stages cannot run detached")
    try:
        _record_stage_inputs(root, manifest, stage)
    except WorkflowError as exc:
        execution.update(
            {"state": "blocked", "message": str(exc), "finished_at": _utc_now()}
        )
        manifest["events"].append(_event("stage_blocked", stage=stage["id"]))
        save_workflow(root, manifest)
        raise
    execution.update(
        {
            "state": "running",
            "started_at": _utc_now(),
            "attempts": int(execution.get("attempts", 0)) + 1,
        }
    )
    manifest["events"].append(_event("stage_started", stage=stage["id"]))
    save_workflow(root, manifest)
    if detach:
        if rendered["kind"] == "internal":
            command = {
                "argv": [
                    sys.executable,
                    "-m",
                    "meddeid.workflow_internal",
                    "--workspace",
                    str(root),
                    "--stage",
                    stage["id"],
                ],
                "cwd": str(root),
                "env": {},
            }
        else:
            command = rendered["commands"][0]
        log_path = _state_dir(root) / "logs" / f"{stage['id']}.log"
        result_path = _state_dir(root) / "runs" / f"{stage['id']}.result.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.unlink(missing_ok=True)
        runner = [
            sys.executable,
            "-m",
            "meddeid.workflow_runner",
            "--result",
            str(result_path),
            "--cwd",
            command["cwd"],
            "--",
            *command["argv"],
        ]
        environment = os.environ.copy()
        environment.update(command.get("env", {}))
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                runner,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        execution.update(
            {
                "detached": True,
                "pid": process.pid,
                "log": str(log_path.relative_to(root)),
                "result": str(result_path.relative_to(root)),
            }
        )
        save_workflow(root, manifest)
        return
    if rendered["kind"] == "internal":
        try:
            _run_internal(root, manifest, stage)
            _record_outputs(root, manifest, stage)
        except WorkflowPause as exc:
            execution.update(
                {"state": "pending", "message": str(exc), "finished_at": _utc_now()}
            )
            manifest["events"].append(_event("stage_paused", stage=stage["id"]))
            save_workflow(root, manifest)
            raise
        except KeyboardInterrupt:
            execution.update(
                {
                    "state": "pending",
                    "message": "stage paused; rerun it to resume",
                    "finished_at": _utc_now(),
                }
            )
            manifest["events"].append(_event("stage_paused", stage=stage["id"]))
            save_workflow(root, manifest)
            return
        except WorkflowError as exc:
            execution.update(
                {
                    "state": "failed" if exc.code == EXIT_FAILED else "blocked",
                    "message": str(exc),
                    "finished_at": _utc_now(),
                }
            )
            manifest["events"].append(
                _event("stage_failed", stage=stage["id"], detail=str(exc))
            )
            save_workflow(root, manifest)
            raise
        except Exception as exc:
            wrapped = WorkflowError(
                f"stage adapter failed unexpectedly: {exc}", code=EXIT_FAILED
            )
            execution.update(
                {"state": "failed", "message": str(wrapped), "finished_at": _utc_now()}
            )
            manifest["events"].append(
                _event("stage_failed", stage=stage["id"], detail=str(wrapped))
            )
            save_workflow(root, manifest)
            raise wrapped from exc
        execution.update({"state": "completed", "finished_at": _utc_now()})
        manifest["events"].append(_event("stage_completed", stage=stage["id"]))
        save_workflow(root, manifest)
        return
    try:
        returncode = _run_rendered_foreground(
            rendered,
            browser=rendered["kind"] == "browser",
            interactive_process=bool(stage.get("human")),
        )
        if rendered["kind"] == "browser" and not _human_stage_complete(
            root, stage, rendered
        ):
            execution.update(
                {
                    "state": "pending",
                    "message": "review is saved but not complete; rerun this stage to resume",
                    "finished_at": _utc_now(),
                }
            )
            manifest["events"].append(_event("stage_paused", stage=stage["id"]))
            save_workflow(root, manifest)
            return
        if returncode not in {0, 130}:
            raise WorkflowError(
                f"command exited with status {returncode}", code=EXIT_FAILED
            )
        _record_outputs(root, manifest, stage)
    except WorkflowError as exc:
        execution.update(
            {
                "state": (
                    "failed" if exc.code in {EXIT_FAILED, EXIT_INVALID} else "blocked"
                ),
                "message": str(exc),
                "finished_at": _utc_now(),
            }
        )
        manifest["events"].append(
            _event("stage_failed", stage=stage["id"], detail=str(exc))
        )
        save_workflow(root, manifest)
        raise
    except Exception as exc:
        wrapped = WorkflowError(
            f"stage command failed unexpectedly: {exc}", code=EXIT_FAILED
        )
        execution.update(
            {"state": "failed", "message": str(wrapped), "finished_at": _utc_now()}
        )
        manifest["events"].append(
            _event("stage_failed", stage=stage["id"], detail=str(wrapped))
        )
        save_workflow(root, manifest)
        raise wrapped from exc
    execution.update(
        {"state": "completed", "finished_at": _utc_now(), "returncode": returncode}
    )
    manifest["events"].append(_event("stage_completed", stage=stage["id"]))
    save_workflow(root, manifest)


def run_stage(
    workspace: Path | str,
    stage_id: str,
    *,
    dry_run: bool = False,
    yes: bool = False,
    detach: bool = False,
) -> dict[str, Any]:
    root, manifest = load_workflow(workspace)
    status = workflow_status(root)
    root, manifest = load_workflow(root)
    status_item = next(
        (item for item in status["stages"] if item["id"] == stage_id), None
    )
    if status_item is None:
        raise WorkflowError(f"workflow has no stage named {stage_id!r}")
    stage = next(item for item in manifest["stages"] if item["id"] == stage_id)
    if status_item["state"] != "ready" and not dry_run:
        retryable = status_item["state"] in {"failed", "blocked"} and (
            stage.get("execution", {}).get("state") in {"failed", "blocked"}
            or stage.get("action", {}).get("name") == "start-deployment"
        )
        if retryable:
            stage["execution"].update(
                {"state": "pending", "message": "explicit retry requested"}
            )
            save_workflow(root, manifest)
        else:
            raise WorkflowError(
                f"stage {stage_id} is {status_item['state']}: {status_item['message']}",
                code=(
                    EXIT_NEEDS_INPUT
                    if status_item["state"] == "needs_input"
                    else EXIT_BLOCKED
                ),
            )
    rendered = render_stage_action(root, stage_id)
    if dry_run:
        return rendered
    if (stage.get("expensive") or stage.get("external")) and not yes:
        raise WorkflowError(
            f"stage {stage_id} may use substantial compute or an external service; inspect --dry-run and rerun with --yes",
            code=EXIT_CONFIRMATION,
        )
    _execute_stage(root, manifest, stage, detach=detach)
    return workflow_status(root)


def run_next(
    workspace: Path | str,
    *,
    interactive: bool = False,
    yes: bool = False,
    detach: bool = False,
) -> dict[str, Any]:
    root, _ = load_workflow(workspace)
    status = workflow_status(root)
    if status["complete"]:
        return status
    next_item = status["next"]
    if next_item and next_item["state"] == "needs_input":
        if interactive:
            prompt_unresolved(root)
            status = workflow_status(root)
            next_item = status["next"]
        else:
            missing = next_item.get("missing_decisions", [])
            commands = " ".join(f"--set {key}=VALUE" for key in missing)
            raise WorkflowError(
                f"stage {next_item['id']} needs input: {', '.join(missing)}. Configure it with:\n"
                f"  meddeid workflow configure {shlex.quote(str(root))} {commands}",
                code=EXIT_NEEDS_INPUT,
            )
    if not next_item:
        return status
    if next_item["state"] == "running":
        return status
    if next_item["state"] in {"blocked", "failed"}:
        raise WorkflowError(
            f"stage {next_item['id']} is {next_item['state']}: {next_item['message']}",
            code=EXIT_BLOCKED if next_item["state"] == "blocked" else EXIT_FAILED,
        )
    if next_item["state"] != "ready":
        raise WorkflowError(
            f"no executable stage is ready; current state is {next_item['state']}",
            code=EXIT_BLOCKED,
        )
    return run_stage(root, next_item["id"], yes=yes, detach=detach)


def explain_workflow(workspace: Path | str) -> list[dict[str, str]]:
    status = workflow_status(workspace)
    return [
        {
            "id": stage["id"],
            "state": stage["state"],
            "why": stage["why"],
            "reason": stage["message"],
        }
        for stage in status["stages"]
    ]


def _tool_status(tool: str) -> dict[str, Any]:
    executable = shutil.which(tool)
    return {"tool": tool, "available": executable is not None, "path": executable}


def doctor(template_id: str | None = None) -> dict[str, Any]:
    try:
        templates = (
            [get_template(template_id)]
            if template_id
            else [get_template(item["id"]) for item in list_templates()]
        )
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    tools: set[str] = {"meddeid"}
    browser_needed = False
    for template in templates:
        for stage in template["stages"]:
            action = stage["action"]
            tools.update(action.get("tools", []))
            if action.get("kind") == "browser":
                browser_needed = True
                tools.update({"docker", "npm"})
        if template["id"] == "deployment":
            tools.add("meddeid-server")
        if template["id"] == "synthetic-corpus":
            tools.add("meddeid-data")
    checks = [_tool_status(tool) for tool in sorted(tools)]
    packages = _installed_versions()
    available = {item["tool"]: item["available"] for item in checks}
    required_commands_ready = all(
        item["available"] for item in checks if item["tool"] not in {"docker", "npm"}
    )
    browser_runtime_ready = (
        not browser_needed
        or available.get("docker", False)
        or available.get("npm", False)
    )
    return {
        "workflow": template_id,
        "python": {
            "version": sys.version.split()[0],
            "supported": sys.version_info >= (3, 10),
        },
        "tools": checks,
        "packages": packages,
        "browser_runtime": {
            "needed": browser_needed,
            "available": browser_runtime_ready,
            "alternatives": ["docker", "npm"],
        },
        "ready": sys.version_info >= (3, 10)
        and required_commands_ready
        and browser_runtime_ready,
    }


def guide_text(*, include_instructions: bool = True) -> str:
    lines = ["What do you want to do?", ""]
    for index, item in enumerate(list_guide_groups(), 1):
        lines.append(f"  {index}. {item['title']}")
        lines.append(f"     {item['summary']}")
    if include_instructions:
        lines.extend(
            [
                "",
                "Start the guided setup with:",
                "  meddeid start",
                "",
                "Advanced and non-interactive use remains available with:",
                "  meddeid workflow init TYPE WORKSPACE --set KEY=VALUE",
            ]
        )
    return "\n".join(lines)
