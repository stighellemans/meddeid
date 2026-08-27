from __future__ import annotations

import pytest

from meddeid.workflow import _ONBOARDING_ACTIONS, parse_decision_value
from meddeid.workflow_contracts import WorkflowTemplateError, validate_template
from meddeid.workflow_templates import get_template, list_templates


def test_all_shipped_templates_pass_typed_validation() -> None:
    for item in list_templates():
        validate_template(get_template(item["id"]))


def test_onboarding_internal_actions_use_the_adapter_registry() -> None:
    assert set(_ONBOARDING_ACTIONS.names) == {
        "audit-language-resources",
        "generate-synthetic-corpus",
        "review-synthetic-documents",
        "scaffold-language-package",
        "seal-synthetic-splits",
        "test-language-conformance",
        "validate-model-checkpoint",
        "validate-synthetic-quality",
        "verify-model-interfaces",
    }


def test_template_validation_rejects_cycles_and_bad_action_options() -> None:
    cyclic = {
        "id": "broken",
        "version": "1",
        "decisions": [],
        "stages": [
            {
                "id": "one",
                "requires": ["two"],
                "decisions": [],
                "action": {"kind": "internal", "name": "record-stage"},
                "outputs": [],
            },
            {
                "id": "two",
                "requires": ["one"],
                "decisions": [],
                "action": {"kind": "internal", "name": "record-stage"},
                "outputs": [],
            },
        ],
    }
    with pytest.raises(WorkflowTemplateError, match="dependency cycle"):
        validate_template(cyclic)

    invalid_option = {
        "id": "broken-option",
        "version": "1",
        "decisions": [],
        "stages": [
            {
                "id": "run",
                "requires": [],
                "decisions": [],
                "action": {
                    "kind": "command",
                    "argv": ["tool"],
                    "options": [{"decision": "missing", "flag": "--missing"}],
                },
                "outputs": [],
            }
        ],
    }
    with pytest.raises(WorkflowTemplateError, match="unknown decision"):
        validate_template(invalid_option)


def test_profile_decisions_are_unversioned_and_reject_bare_english() -> None:
    spec = {"kind": "profiles"}
    assert parse_decision_value(spec, "en_gb,en-US") == ["en-GB", "en-US"]
    with pytest.raises(ValueError, match="unversioned"):
        parse_decision_value(spec, "en-GB@1")
    with pytest.raises(ValueError, match="ambiguous"):
        parse_decision_value(spec, "en")
