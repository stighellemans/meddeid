from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from meddeid_core import BERT_ENTITY_LABELS

from meddeid.workflow import (
    EXIT_CONFIRMATION,
    EXIT_NEEDS_INPUT,
    WorkflowError,
    _condition_refs,
    _output_paths,
    _prompt,
    _record_outputs,
    _record_stage_inputs,
    _stage_decision_refs,
    configure_workflow,
    evaluate_condition,
    initialize_workflow,
    load_workflow,
    run_next,
    run_stage,
    save_workflow,
    workflow_status,
)
from meddeid.workflow_runner import main as runner_main
from meddeid.workflow_templates import (
    WORKFLOW_CONTRACT,
    get_template,
    list_guide_groups,
    list_templates,
)


def _jsonl(path: Path, *, document_id: str = "doc-1", annotated: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "text": "Patient Alex Example attended.",
                "spans": [],
                "annotated": annotated,
                "metadata": {"lang": "en-GB"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _states(status: dict) -> dict[str, str]:
    return {stage["id"]: stage["state"] for stage in status["stages"]}


def test_condition_language_has_three_valued_logic() -> None:
    condition = {"all": [{"decision": "review", "eq": True}, {"decision": "count", "gt": 1}]}
    assert evaluate_condition(condition, {}) is None
    assert evaluate_condition(condition, {"review": False}) is False
    assert evaluate_condition(condition, {"review": True, "count": 2}) is True


def test_every_template_has_valid_references() -> None:
    identifiers = [item["id"] for item in list_templates()]
    assert identifiers == [
        "inference",
        "dataset-review",
        "benchmark",
        "evaluation",
        "training",
        "domain-adaptation",
        "deployment",
        "language-profile",
        "synthetic-corpus",
        "model-bundle",
    ]
    for identifier in identifiers:
        template = get_template(identifier)
        assert template["version"] == "1"
        assert template["stages"]
        decision_ids = {item["key"] for item in template["decisions"]}
        stage_ids: set[str] = set()
        for decision in template["decisions"]:
            assert _condition_refs(decision.get("ask_when")) <= decision_ids
        for stage in template["stages"]:
            assert stage["id"] not in stage_ids
            assert set(stage["requires"]) <= stage_ids
            assert _stage_decision_refs(stage) <= decision_ids
            stage_ids.add(stage["id"])
    grouped = [
        workflow["id"]
        for group in list_guide_groups()
        for workflow in group["workflows"]
    ]
    assert sorted(grouped) == sorted(identifiers)


def test_decision_prompt_supports_numbered_choices_and_help(monkeypatch, capsys) -> None:
    answers = iter(["?", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt_text: next(answers))
    value = _prompt({
        "prompt": "Should reviewers see suggestions?",
        "kind": "choice",
        "choices": ["assisted", "blinded"],
        "choice_labels": {
            "assisted": "Yes — assisted",
            "blinded": "No — blinded",
        },
        "default": None,
        "required": True,
        "why": "Suggestions change the review protocol.",
    })
    assert value == "blinded"
    output = capsys.readouterr().out
    assert "1. Yes — assisted" in output
    assert "Why this is asked" in output


def test_operational_decision_is_requested_only_when_stage_is_reached(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "source.jsonl")
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "inference",
        values={"inference_mode": "batch", "source": str(source)},
    )
    initial = workflow_status(workspace)
    assert initial["next"]["id"] == "inspect_input"
    assert initial["next"]["state"] == "ready"

    run_stage(workspace, "inspect_input")
    reached = workflow_status(workspace)
    assert reached["next"]["id"] == "batch_inference"
    assert reached["next"]["state"] == "needs_input"
    assert reached["next"]["missing_decisions"] == ["device", "model"]
    with pytest.raises(WorkflowError) as raised:
        run_next(workspace, interactive=False)
    assert raised.value.code == EXIT_NEEDS_INPUT
    assert "--set device=VALUE" in str(raised.value)


def test_completed_stage_is_recomputed_from_input_hash(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "source.jsonl")
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "inference",
        values={"inference_mode": "batch", "source": str(source), "device": "cpu"},
    )
    run_stage(workspace, "inspect_input")
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    status = workflow_status(workspace)
    assert _states(status)["inspect_input"] == "blocked"
    assert "changed" in status["stages"][0]["message"]


def test_existing_gold_does_not_require_reviewer_branches(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "gold.jsonl")
    workspace = tmp_path / "benchmark"
    initialize_workflow(
        workspace,
        "benchmark",
        values={
            "source": str(source),
            "input_role": "existing_gold",
            "re_review": "false",
            "detailed_evaluation": "false",
            "score_predictions": "false",
        },
    )
    status = workflow_status(workspace)
    states = _states(status)
    assert status["next"]["id"] == "validate_source"
    assert states["prepare_primary"] == "not_applicable"
    assert states["primary_review"] == "not_applicable"
    assert states["curate"] == "not_applicable"
    assert states["subannotate"] == "skipped"
    assert states["score"] == "skipped"


@pytest.mark.parametrize(("mode", "expected"), [("blinded", "skipped"), ("assisted", "pending")])
def test_blinded_vs_assisted_review_controls_preannotation(
    tmp_path: Path, mode: str, expected: str
) -> None:
    workspace = tmp_path / mode
    initialize_workflow(
        workspace,
        "benchmark",
        values={
            "source": str(_jsonl(tmp_path / f"{mode}.jsonl", annotated=False)),
            "input_role": "unlabelled", "review_mode": mode, "reviewer_count": "1",
            "detailed_evaluation": "false", "score_predictions": "false",
        },
    )
    assert _states(workflow_status(workspace))["preannotate"] == expected


def test_detailed_benchmark_is_the_only_path_that_includes_subannotation(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "gold.jsonl")
    for detailed, expected in (("false", "skipped"), ("true", "pending")):
        values = {
            "source": str(source), "input_role": "existing_gold", "re_review": "false",
            "detailed_evaluation": detailed, "score_predictions": "false",
        }
        if detailed == "true":
            values["profiles"] = "en-GB,en-US"
        workspace = tmp_path / f"detailed-{detailed}"
        initialize_workflow(workspace, "benchmark", values=values)
        assert _states(workflow_status(workspace))["subannotate"] == expected


@pytest.mark.parametrize(
    ("reviewer_count", "gold_policy", "expected"),
    [(1, None, "not_applicable"), (2, "selected_reviewer", "skipped"), (2, "adjudicate", "pending")],
)
def test_reviewer_policy_controls_curation(
    tmp_path: Path, reviewer_count: int, gold_policy: str | None, expected: str
) -> None:
    values = {
        "source": str(_jsonl(tmp_path / f"source-{reviewer_count}-{gold_policy}.jsonl", annotated=False)),
        "input_role": "unlabelled",
        "review_mode": "blinded",
        "reviewer_count": str(reviewer_count),
        "detailed_evaluation": "false",
        "score_predictions": "false",
    }
    if gold_policy:
        values["gold_policy"] = gold_policy
    if gold_policy == "selected_reviewer":
        values.update({"selected_reviewer": "1", "selection_rationale": "pre-registered lead reviewer"})
    workspace = tmp_path / f"benchmark-{reviewer_count}-{gold_policy}"
    initialize_workflow(workspace, "benchmark", values=values)
    assert _states(workflow_status(workspace))["curate"] == expected


def test_training_protocol_selects_exactly_one_fit_path(tmp_path: Path) -> None:
    common = {
        "project": str(tmp_path),
        "development": str(_jsonl(tmp_path / "development.jsonl")),
        "test_gold": str(_jsonl(tmp_path / "test.jsonl", document_id="test-1")),
        "config": str(tmp_path / "train.yaml"),
        "device": "cpu",
    }
    (tmp_path / "train.yaml").write_text("epochs: 1\n", encoding="utf-8")
    for protocol, expected in (("fit", ("pending", "not_applicable")), ("select_refit", ("not_applicable", "pending"))):
        workspace = tmp_path / protocol
        initialize_workflow(workspace, "training", values={**common, "training_protocol": protocol})
        states = _states(workflow_status(workspace))
        assert (states["fit"], states["select_epochs"]) == expected


def test_epoch_selection_command_has_no_sealed_test_argument() -> None:
    template = get_template("training")
    selection = next(stage for stage in template["stages"] if stage["id"] == "select_epochs")
    rendered = " ".join(selection["action"]["argv"])
    assert "prepared/selection" in rendered
    assert "test_gold" not in rendered
    assert "prepared/refit/test" not in rendered


def test_configure_previews_and_archives_invalidated_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "inference",
        values={"inference_mode": "single", "source": str(first), "device": "cpu"},
    )
    root, manifest = load_workflow(workspace)
    output = root / "artifacts" / "deidentified.txt"
    output.write_text("redacted\n", encoding="utf-8")
    stage = next(item for item in manifest["stages"] if item["id"] == "single_inference")
    stage["execution"] = {
        "state": "completed",
        "attempts": 1,
        "outputs": {"artifacts/deidentified.txt": __import__("hashlib").sha256(output.read_bytes()).hexdigest()},
    }
    save_workflow(root, manifest)
    with pytest.raises(WorkflowError) as raised:
        configure_workflow(workspace, values={"source": str(second)}, reason="new revision")
    assert raised.value.code == EXIT_CONFIRMATION
    result = configure_workflow(
        workspace,
        values={"source": str(second)},
        reason="new revision",
        yes=True,
    )
    assert result["archived"]
    assert not output.exists()
    assert list(Path(result["archived"]).rglob("deidentified.txt"))


def test_detached_stage_is_reconciled_from_result_and_artifact(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Alex Example", encoding="utf-8")
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "inference",
        values={"inference_mode": "single", "source": str(source), "device": "cpu"},
    )
    root, manifest = load_workflow(workspace)
    inspect = next(item for item in manifest["stages"] if item["id"] == "inspect_input")
    inspect["execution"] = {"state": "completed", "attempts": 1, "outputs": {}}
    single = next(item for item in manifest["stages"] if item["id"] == "single_inference")
    single["execution"] = {"state": "running", "attempts": 1, "detached": True}
    save_workflow(root, manifest)
    output = root / "artifacts" / "deidentified.txt"
    output.write_text("[REDACTED]\n", encoding="utf-8")
    result = root / ".workflow" / "runs" / "single_inference.result.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({"returncode": 0, "finished_at": "2026-08-21T00:00:00Z"}), encoding="utf-8")
    assert _states(workflow_status(root))["single_inference"] == "completed"


def test_detached_runner_writes_atomic_result(tmp_path: Path, monkeypatch) -> None:
    result = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workflow-runner", "--result", str(result), "--cwd", str(tmp_path), "--",
            sys.executable, "-c", "print('done')",
        ],
    )
    assert runner_main() == 0
    assert json.loads(result.read_text(encoding="utf-8"))["returncode"] == 0


def test_interactive_initialization_shows_graph_before_any_write(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    workspace = tmp_path / "cancelled"
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(WorkflowError) as raised:
        initialize_workflow(
            workspace,
            "inference",
            values={"inference_mode": "service"},
            interactive=True,
        )
    assert raised.value.code == EXIT_CONFIRMATION
    assert "Resolved stage graph" in capsys.readouterr().out
    assert not workspace.exists()


def test_simple_interactive_initialization_shows_resolved_path(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    workspace = tmp_path / "cancelled"
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(WorkflowError) as raised:
        initialize_workflow(
            workspace,
            "inference",
            values={"inference_mode": "service"},
            interactive=True,
            simple=True,
        )
    assert raised.value.code == EXIT_CONFIRMATION
    output = capsys.readouterr().out
    assert "Your workflow" in output
    assert "Start local service" in output
    assert "single_inference" not in output
    assert not workspace.exists()


def test_optional_blank_answer_is_not_requested_again(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "cancelled"
    answers = iter(["3", "example/model", "", "", "no"])
    prompts: list[str] = []

    def answer(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", answer)
    with pytest.raises(WorkflowError) as raised:
        initialize_workflow(workspace, "inference", interactive=True, simple=True)
    assert raised.value.code == EXIT_CONFIRMATION
    assert sum("Immutable model revision" in prompt for prompt in prompts) == 1
    assert sum("Language profile" in prompt for prompt in prompts) == 1


def test_remote_generation_without_authorization_blocks_before_external_call(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "synthetic"
    initialize_workflow(
        workspace,
        "synthetic-corpus",
            values={
                "profiles": "en-GB,en-US", "count": "2",
                "generation_mode": "remote", "allow_remote": "false",
                "paid_model_review": "false",
            },
    )
    monkeypatch.setattr(
        "meddeid.workflow._run_checked",
        lambda *_args, **_kwargs: pytest.fail("external command must not run"),
    )
    with pytest.raises(WorkflowError) as raised:
        run_stage(workspace, "generate", yes=True)
    assert raised.value.code != 0
    assert "not authorized" in str(raised.value)


def test_paid_remote_review_requires_an_explicit_provider(tmp_path: Path) -> None:
    workspace = tmp_path / "synthetic"
    initialize_workflow(
        workspace,
        "synthetic-corpus",
        values={
            "profiles": "en-GB,en-US",
            "count": "2",
            "generation_mode": "remote",
            "allow_remote": "true",
            "paid_model_review": "true",
        },
    )
    status = workflow_status(workspace)
    assert status["next"]["id"] == "generate"
    assert status["next"]["state"] == "needs_input"
    assert status["next"]["missing_decisions"] == ["reviewer_provider"]


def test_language_scaffold_output_is_not_invalidated_by_conformance(tmp_path: Path) -> None:
    package_dir = tmp_path / "language-package"
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "language-profile",
        values={
            "package_name": "meddeid-language-example",
            "profiles": "en-GB",
            "output_dir": str(package_dir),
            "resource_mode": "none",
        },
    )
    run_stage(workspace, "scaffold")
    profile_manifest = json.loads(
        (
            package_dir
            / "src"
            / "meddeid_language_example"
            / "resources"
            / "profiles"
            / "en-GB"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert profile_manifest["profile_id"] == "en-GB"
    assert "version" not in profile_manifest
    assert "ruleset_version" not in profile_manifest
    javascript = (package_dir / "js" / "subannotation-en-gb.js").read_text(
        encoding="utf-8"
    )
    assert 'selection: "en-GB"' in javascript
    assert "version:" not in javascript
    assert "rulesetVersion" not in javascript
    run_stage(workspace, "conformance")
    status = workflow_status(workspace)
    assert status["complete"] is True
    assert _states(status)["scaffold"] == "completed"


def test_existing_gold_does_not_require_ui_completion_flags(tmp_path: Path) -> None:
    gold = tmp_path / "reference-items.jsonl"
    gold.write_text(
        json.dumps({
            "document_id": "gold-1",
            "text": "Patient Jane Doe.",
            "spans": [{"start": 8, "end": 16, "label": "PERSON"}],
        }) + "\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workflow"
    initialize_workflow(
        workspace,
        "benchmark",
        values={
            "source": str(gold),
            "input_role": "existing_gold",
            "re_review": "false",
            "detailed_evaluation": "false",
            "score_predictions": "false",
        },
    )

    run_stage(workspace, "select_gold")

    packaged = workspace / "artifacts" / "authoritative-annotations.jsonl"
    manifest = json.loads(
        (workspace / "artifacts" / "authoritative-annotations.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert packaged.read_bytes() == gold.read_bytes()
    assert manifest["completion_evidence"] == (
        "accepted input_role=existing_gold after validate_source"
    )


def test_workspace_cannot_mutate_a_pinned_parent_input(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    development = _jsonl(project / "development.jsonl")
    test_gold = _jsonl(project / "test.jsonl", document_id="test-1")
    config = project / "config.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")
    workspace = project / "workflow"
    initialize_workflow(
        workspace,
        "training",
        values={
            "project": str(project),
            "development": str(development),
            "test_gold": str(test_gold),
            "config": str(config),
            "training_protocol": "fit",
            "device": "cpu",
        },
    )
    with pytest.raises(WorkflowError, match="inside input directory"):
        run_stage(workspace, "prepare")


def test_inference_service_dry_run_pins_environment(tmp_path: Path) -> None:
    workspace = tmp_path / "service"
    initialize_workflow(
        workspace,
        "inference",
        values={
            "inference_mode": "service",
            "model": "example/model",
            "revision": "abc123",
            "language_profile": "en-GB",
            "offline": "true",
            "device": "cpu",
        },
    )
    rendered = run_stage(workspace, "local_service", dry_run=True)
    assert rendered["commands"][0]["env"] == {
        "MEDDEID_DEVICE": "cpu",
        "MEDDEID_MODEL": "example/model",
        "MEDDEID_OFFLINE": "True",
        "MEDDEID_REVISION": "abc123",
        "MEDDEID_LANGUAGE_PROFILE": "en-GB",
    }


def test_deployment_dry_run_propagates_locale(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "deployment"
    initialize_workflow(
        workspace,
        "deployment",
        values={
            "model": "example/model",
            "revision": "abc123",
            "language_profile": "en-US",
            "deployment_target": "local",
            "device": "cpu",
            "port": 8123,
        },
    )

    rendered = run_stage(workspace, "start", dry_run=True)

    assert rendered["commands"][0]["env"]["MEDDEID_LANGUAGE_PROFILE"] == "en-US"


def test_mixed_english_benchmark_acceptance_graph(tmp_path: Path) -> None:
    source = tmp_path / "mixed.jsonl"
    rows = [
        {"document_id": "gb-1", "text": "Mr Alex Example", "spans": [], "annotated": True, "metadata": {"lang": "en-GB"}},
        {"document_id": "us-1", "text": "Mr Alex Example", "spans": [], "annotated": True, "metadata": {"lang": "en-US"}},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    workspace = tmp_path / "mixed-workflow"
    initialize_workflow(
        workspace,
        "benchmark",
        values={
            "source": str(source), "input_role": "existing_gold", "re_review": "false",
            "detailed_evaluation": "true", "profiles": "en-GB,en-US",
            "score_predictions": "false",
        },
    )
    root, manifest = load_workflow(workspace)
    assert manifest["decisions"]["profiles"] == ["en-GB", "en-US"]
    assert workflow_status(root)["next"]["id"] == "validate_source"


def test_one_resolved_end_to_end_fixture_per_workflow(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    project = inputs / "project"
    project.mkdir(parents=True)
    gold = _jsonl(inputs / "gold.jsonl")
    predictions = _jsonl(inputs / "predictions.jsonl")
    development = _jsonl(inputs / "development.jsonl", document_id="development-1")
    test = _jsonl(inputs / "test.jsonl", document_id="test-1")
    config = inputs / "config.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")
    checkpoint = inputs / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    package_dir = tmp_path / "language-package"
    examples = {
        "inference": {
            "inference_mode": "batch", "source": str(gold), "device": "cpu",
            "model": "example/model",
        },
        "dataset-review": {
            "source": str(gold), "namespace": "fixture", "language_profile": "en-GB",
            "create_split": "false", "review_mode": "blinded", "reviewer_count": "1",
            "runtime": "docker",
        },
        "benchmark": {
            "source": str(gold), "input_role": "existing_gold", "re_review": "false",
            "detailed_evaluation": "false", "score_predictions": "false",
        },
        "evaluation": {"gold": str(gold), "predictions": str(predictions), "plots": "false", "stability": "false"},
        "training": {
            "project": str(project), "development": str(development), "test_gold": str(test),
            "config": str(config), "training_protocol": "fit", "device": "cpu",
        },
        "domain-adaptation": {
            "project": str(project), "development": str(development), "test_gold": str(test),
            "review_development": "false", "baseline_model": "example/model",
            "baseline_revision": "abc123", "detailed_evaluation": "false",
            "training_protocol": "fit", "config": str(config), "device": "cpu",
        },
        "deployment": {
            "model": "example/model", "deployment_target": "local", "device": "cpu",
        },
        "language-profile": {
            "package_name": "meddeid-language-example", "profiles": "en-GB",
            "output_dir": str(package_dir), "resource_mode": "none",
        },
        "synthetic-corpus": {"profiles": "en-GB,en-US", "count": "2", "generation_mode": "local"},
        "model-bundle": {
            "checkpoint": str(checkpoint), "profiles": "en-GB", "base_encoder": "example/base",
            "base_revision": "abc123", "device": "cpu",
        },
    }
    for template_id, values in examples.items():
        workspace = tmp_path / "workflows" / template_id
        manifest = initialize_workflow(workspace, template_id, values=values)
        assert manifest["contract_version"] == WORKFLOW_CONTRACT
        for _ in range(len(manifest["stages"]) + 1):
            status = workflow_status(workspace)
            if status["complete"]:
                break
            assert status["next"] is not None
            assert status["next"]["state"] == "ready", (
                template_id,
                status["next"],
            )
            root, current = load_workflow(workspace)
            stage = next(
                item for item in current["stages"]
                if item["id"] == status["next"]["id"]
            )
            _record_stage_inputs(root, current, stage)
            stage["execution"]["state"] = "running"
            for output in _output_paths(root, current, stage):
                action_name = stage.get("action", {}).get("name")
                if output.name == "service.json":
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({"pid": os.getpid()}) + "\n", encoding="utf-8")
                elif action_name in {
                    "audit-language-resources",
                    "test-language-conformance",
                    "validate-synthetic-quality",
                    "validate-model-checkpoint",
                    "verify-model-interfaces",
                } and output == _output_paths(root, current, stage)[0]:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({"passed": True}) + "\n", encoding="utf-8")
                elif action_name == "seal-synthetic-splits" and output.name == "splits.manifest.json":
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(
                        json.dumps({
                            "allowed_labels": list(BERT_ENTITY_LABELS),
                            "forbidden_generated_labels": ["Anonymize_Other"],
                        }) + "\n",
                        encoding="utf-8",
                    )
                elif output.suffix:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text("fixture\n", encoding="utf-8")
                else:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / ".fixture").write_text("fixture\n", encoding="utf-8")
            _record_outputs(root, current, stage)
            stage["execution"]["state"] = "completed"
            save_workflow(root, current)
        assert workflow_status(workspace)["complete"] is True, template_id
