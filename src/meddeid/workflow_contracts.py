"""Typed validation for declarative ``meddeid.workflow.v1`` templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class WorkflowTemplateError(ValueError):
    """Raised before a malformed workflow template can create a workspace."""


def condition_references(condition: Mapping[str, Any] | None) -> set[str]:
    if not condition:
        return set()
    if "decision" in condition:
        return {str(condition["decision"])}
    references: set[str] = set()
    for key in ("all", "any"):
        children = condition.get(key, ())
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
            raise WorkflowTemplateError(f"condition {key!r} must contain a list")
        for child in children:
            if not isinstance(child, Mapping):
                raise WorkflowTemplateError("workflow condition child must be an object")
            references.update(condition_references(child))
    if "not" in condition:
        child = condition["not"]
        if not isinstance(child, Mapping):
            raise WorkflowTemplateError("workflow 'not' condition must be an object")
        references.update(condition_references(child))
    if not ({"decision", "all", "any", "not"} & set(condition)):
        raise WorkflowTemplateError(f"unsupported workflow condition: {dict(condition)}")
    return references


@dataclass(frozen=True)
class ActionContract:
    kind: str
    name: str | None = None

    @classmethod
    def validate(
        cls,
        action: Mapping[str, Any],
        *,
        decisions: set[str],
        location: str,
    ) -> "ActionContract":
        kind = str(action.get("kind") or "")
        if kind not in {"command", "internal", "browser"}:
            raise WorkflowTemplateError(f"{location}: unsupported action kind {kind!r}")
        name: str | None = None
        if kind == "internal":
            name = str(action.get("name") or "").strip()
            if not name:
                raise WorkflowTemplateError(f"{location}: internal action has no name")
        elif kind == "command":
            argv = action.get("argv")
            if not isinstance(argv, list) or not argv or not all(
                isinstance(item, str) and item for item in argv
            ):
                raise WorkflowTemplateError(f"{location}: command argv must be non-empty strings")
            for option in action.get("options", ()):
                decision = str(option.get("decision") or "")
                if decision not in decisions:
                    raise WorkflowTemplateError(
                        f"{location}: command option references unknown decision {decision!r}"
                    )
            for option in action.get("env_options", ()):
                decision = str(option.get("decision") or "")
                if decision not in decisions:
                    raise WorkflowTemplateError(
                        f"{location}: environment option references unknown decision {decision!r}"
                    )
        else:
            if not str(action.get("app") or "").strip():
                raise WorkflowTemplateError(f"{location}: browser action has no app")
            if not str(action.get("source") or "").strip():
                raise WorkflowTemplateError(f"{location}: browser action has no source")
        return cls(kind=kind, name=name)


def _assert_acyclic(stages: Sequence[Mapping[str, Any]], *, template_id: str) -> None:
    graph = {
        str(stage["id"]): tuple(str(item) for item in stage.get("requires", ()))
        for stage in stages
    }
    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in complete:
            return
        if stage_id in visiting:
            raise WorkflowTemplateError(
                f"workflow {template_id!r} has a dependency cycle at {stage_id!r}"
            )
        visiting.add(stage_id)
        for dependency in graph[stage_id]:
            visit(dependency)
        visiting.remove(stage_id)
        complete.add(stage_id)

    for stage_id in graph:
        visit(stage_id)


def validate_template(template: Mapping[str, Any]) -> None:
    """Validate references and executable action shapes for one template."""

    template_id = str(template.get("id") or "")
    if not template_id:
        raise WorkflowTemplateError("workflow template has no id")
    if not str(template.get("version") or ""):
        raise WorkflowTemplateError(f"workflow {template_id!r} has no version")
    decisions = template.get("decisions")
    stages = template.get("stages")
    if not isinstance(decisions, list) or not isinstance(stages, list) or not stages:
        raise WorkflowTemplateError(
            f"workflow {template_id!r} requires decision and non-empty stage lists"
        )
    decision_ids = [str(item.get("key") or "") for item in decisions]
    stage_ids = [str(item.get("id") or "") for item in stages]
    if any(not item for item in decision_ids) or len(decision_ids) != len(set(decision_ids)):
        raise WorkflowTemplateError(
            f"workflow {template_id!r} has missing or duplicate decision identifiers"
        )
    if any(not item for item in stage_ids) or len(stage_ids) != len(set(stage_ids)):
        raise WorkflowTemplateError(
            f"workflow {template_id!r} has missing or duplicate stage identifiers"
        )
    known_decisions = set(decision_ids)
    known_stages = set(stage_ids)
    for decision in decisions:
        missing = condition_references(decision.get("ask_when")) - known_decisions
        if missing:
            raise WorkflowTemplateError(
                f"workflow {template_id!r} decision {decision['key']!r} references "
                f"unknown decisions {sorted(missing)}"
            )
    for stage in stages:
        stage_id = str(stage["id"])
        missing_decisions = (
            set(stage.get("decisions", ()))
            | condition_references(stage.get("applicable_when"))
            | condition_references(stage.get("enabled_when"))
        ) - known_decisions
        missing_stages = set(stage.get("requires", ())) - known_stages
        if missing_decisions or missing_stages:
            raise WorkflowTemplateError(
                f"workflow {template_id!r} stage {stage_id!r} has invalid references: "
                f"decisions={sorted(missing_decisions)}, stages={sorted(missing_stages)}"
            )
        if stage_id in set(stage.get("requires", ())):
            raise WorkflowTemplateError(
                f"workflow {template_id!r} stage {stage_id!r} depends on itself"
            )
        ActionContract.validate(
            stage.get("action") or {},
            decisions=known_decisions,
            location=f"workflow {template_id!r} stage {stage_id!r}",
        )
        if not isinstance(stage.get("outputs"), list):
            raise WorkflowTemplateError(
                f"workflow {template_id!r} stage {stage_id!r} outputs must be a list"
            )
    _assert_acyclic(stages, template_id=template_id)


__all__ = [
    "ActionContract",
    "WorkflowTemplateError",
    "condition_references",
    "validate_template",
]
