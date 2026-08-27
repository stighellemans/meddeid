from __future__ import annotations

import json
from pathlib import Path

from meddeid.cli import main
from meddeid.workflow_cli import _choose_workflow_interactively


def test_workflow_cli_lists_templates(capsys) -> None:
    assert main(["workflow", "list"]) == 0
    output = capsys.readouterr().out
    assert "dataset-review" in output
    assert "domain-adaptation" in output
    assert "model-bundle" in output


def test_guide_groups_internal_workflows_behind_six_goals(capsys) -> None:
    assert main(["guide"]) == 0
    output = capsys.readouterr().out
    assert "1. De-identify clinical text" in output
    assert "6. Contribute a language or model package" in output
    assert "domain-adaptation:" not in output


def test_nested_guide_selects_a_workflow(monkeypatch, capsys) -> None:
    answers = iter(["2", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert _choose_workflow_interactively() == "benchmark"
    assert "what best matches your situation" in capsys.readouterr().out


def test_noninteractive_next_returns_stable_needs_input_code(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"document_id": "one", "text": "Example", "spans": []}) + "\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "guided"
    assert main([
        "workflow", "init", "inference", str(workspace), "--non-interactive",
        "--set", "inference_mode=batch", "--set", f"source={source}",
    ]) == 0
    assert main(["workflow", "next", str(workspace)]) == 0
    assert main(["workflow", "next", str(workspace)]) == 3
    error = capsys.readouterr().err
    assert "--set device=VALUE" in error


def test_status_json_is_machine_readable(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "guided"
    assert main([
        "workflow", "init", "inference", str(workspace), "--non-interactive",
        "--set", "inference_mode=service",
    ]) == 3
    captured = capsys.readouterr()
    assert "--set device=VALUE" in captured.err
    assert main(["workflow", "status", str(workspace), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract_version"] == "meddeid.workflow.v1"
    assert payload["next"]["state"] == "needs_input"


def test_simple_start_status_and_next_aliases(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "guided"
    package = tmp_path / "language-package"
    assert main([
        "start", str(workspace), "--workflow", "language-profile",
        "--non-interactive", "--set", "package_name=meddeid-language-example",
        "--set", "profiles=en-GB", "--set", f"output_dir={package}",
        "--set", "resource_mode=none",
    ]) == 0
    started = capsys.readouterr().out
    assert "0 of 2 stages complete" in started
    assert "Resource mode: none" in started

    assert main(["next", str(workspace)]) == 0
    assert "1 of 2 stages complete" in capsys.readouterr().out
    assert main(["next", str(workspace)]) == 0
    assert "Complete. All included outputs validate." in capsys.readouterr().out


def test_simple_status_discovers_workspace_from_child_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    workspace = tmp_path / "guided"
    assert main([
        "workflow", "init", "inference", str(workspace), "--non-interactive",
        "--set", "inference_mode=service",
    ]) == 3
    capsys.readouterr()
    monkeypatch.chdir(workspace / "artifacts")
    assert main(["status"]) == 0
    output = capsys.readouterr().out
    assert "Local inference" in output
    assert "Decision needed before: Start local service" in output
