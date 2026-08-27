"""Registries for workflow action adapters and semantic artifact validators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ActionContext:
    root: Path
    manifest: dict[str, Any]
    stage: dict[str, Any]

    @property
    def decisions(self) -> dict[str, Any]:
        return self.manifest["decisions"]


class ActionHandler(Protocol):
    def __call__(self, context: ActionContext) -> None: ...


class ArtifactValidator(Protocol):
    def __call__(self, context: ActionContext) -> None: ...


class ActionRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ActionHandler] = {}

    def register(self, name: str, handler: ActionHandler) -> None:
        if not name or name in self._handlers:
            raise ValueError(f"duplicate or empty workflow action name: {name!r}")
        self._handlers[name] = handler

    def handles(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, name: str, context: ActionContext) -> None:
        try:
            handler = self._handlers[name]
        except KeyError as exc:
            raise KeyError(f"unregistered workflow action: {name}") from exc
        handler(context)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))


class ArtifactValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, ArtifactValidator] = {}

    def register(self, action_name: str, validator: ArtifactValidator) -> None:
        if action_name in self._validators:
            raise ValueError(f"duplicate artifact validator: {action_name!r}")
        self._validators[action_name] = validator

    def validate(self, action_name: str, context: ActionContext) -> bool:
        validator = self._validators.get(action_name)
        if validator is None:
            return False
        validator(context)
        return True


__all__ = [
    "ActionContext",
    "ActionHandler",
    "ActionRegistry",
    "ArtifactValidator",
    "ArtifactValidatorRegistry",
]
