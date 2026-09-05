from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from meddeid import cli


PROVENANCE = {
    "contract_version": "meddeid.inference-provenance.v1",
    "software": {"name": "meddeid", "version": "test"},
    "model": {
        "name": "test-model",
        "version": "1",
        "resolved_revision": "abc123",
        "bundle_sha256": "a" * 64,
    },
    "language_profile": {"profile_id": "en-GB"},
}


class FakeEngine:
    def model_info(self) -> dict:
        return {"runtime": {"checked": True}}

    def close(self) -> None:
        return None

    def __call__(self, text, *, metadata=None):
        result = SimpleNamespace(
            text=text,
            deid_text=text,
            spans=[],
            language_profile="en-GB",
            warnings=[],
            processing={},
            provenance=PROVENANCE,
        )
        result.to_contract = lambda: {
            "deid_text": result.deid_text,
            "spans": result.spans,
            "processing": result.processing,
            "warnings": result.warnings,
            "provenance": result.provenance,
        }
        return result


def test_cli_requires_explicit_model(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["model-info", "--quiet"])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "no model selected" in stderr
    assert "meddeid models" in stderr


def test_cli_passes_explicit_model_settings(monkeypatch) -> None:
    captured = {}

    def fake_load_engine(args):
        captured.update(vars(args))
        return FakeEngine()

    monkeypatch.setattr(cli, "_load_engine", fake_load_engine)

    assert cli.main([
        "model-info",
        "--model",
        "stighellemans/meddeid-dutch-synth",
        "--verify-runtime",
        "--quiet",
    ]) == 0
    assert captured["model"] == "stighellemans/meddeid-dutch-synth"
    assert captured["revision"] is None
    assert captured["offline"] is False
    assert captured["device"] is None
    assert captured["language_profile"] is None
    assert captured["age_granularity_config"] is None
    assert captured["min_recommended_date_shift_days"] == 366


def test_model_info_default_inspects_without_loading_runtime(
    monkeypatch, capsys
) -> None:
    from meddeid import model_inspection

    captured = {}

    def fake_inspect(model, **kwargs):
        captured.update(model=model, **kwargs)
        return {"runtime": {"checked": False}}

    monkeypatch.setattr(model_inspection, "inspect_model", fake_inspect)
    monkeypatch.setattr(
        cli,
        "_load_engine",
        lambda _args: pytest.fail("the default inspection must not load a runtime"),
    )

    assert cli.main([
        "model-info",
        "--model",
        "stighellemans/meddeid-dutch-synth",
        "--quiet",
    ]) == 0

    assert captured["model"] == "stighellemans/meddeid-dutch-synth"
    assert json.loads(capsys.readouterr().out) == {
        "runtime": {"checked": False}
    }


def test_importing_cli_does_not_import_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import meddeid.cli; print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "False"


def test_cli_json_reports_bundle_pinned_profile(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Example", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_engine", lambda _args: FakeEngine())

    assert cli.main([
        "deidentify",
        str(source),
        "--model",
        "stighellemans/meddeid-dutch-synth",
        "--json",
        "--quiet",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert list(payload) == [
        "deid_text",
        "spans",
        "processing",
        "warnings",
        "provenance",
    ]
    assert "text" not in payload
    assert payload["deid_text"] == "Example"
    assert payload["provenance"]["language_profile"] == {"profile_id": "en-GB"}


def test_cli_accepts_explicit_input_option(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text('{"text":"Example"}\n', encoding="utf-8")
    captured = {}

    monkeypatch.setattr(cli, "_load_engine", lambda _args: FakeEngine())

    def fake_run_batch(engine, input_path, output_path, **kwargs):
        captured.update(
            engine=engine,
            input_path=input_path,
            output_path=output_path,
            kwargs=kwargs,
        )
        return {"counts": {"documents": 1}}

    monkeypatch.setattr(cli, "_run_batch", fake_run_batch)

    assert cli.main([
        "batch",
        "--input",
        str(source),
        "--output",
        str(output),
        "--model",
        "stighellemans/meddeid-dutch-synth",
        "--quiet",
    ]) == 0

    assert captured["input_path"] == source
    assert captured["output_path"] == output
    assert json.loads(capsys.readouterr().out) == {"documents": 1}


def test_cli_rejects_missing_input_before_loading_model(
    monkeypatch, tmp_path, capsys
) -> None:
    model_loaded = False

    def fake_load_engine(_args):
        nonlocal model_loaded
        model_loaded = True
        return FakeEngine()

    monkeypatch.setattr(cli, "_load_engine", fake_load_engine)
    missing = tmp_path / "missing.jsonl"

    with pytest.raises(SystemExit) as error:
        cli.main([
            "batch",
            str(missing),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--model",
            "stighellemans/meddeid-dutch-synth",
        ])

    assert error.value.code == 2
    assert model_loaded is False
    assert f"input file not found: {missing}" in capsys.readouterr().err


def test_cli_rejects_both_input_forms_before_loading_model(
    monkeypatch, tmp_path, capsys
) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"text":"Example"}\n', encoding="utf-8")
    model_loaded = False

    def fake_load_engine(_args):
        nonlocal model_loaded
        model_loaded = True
        return FakeEngine()

    monkeypatch.setattr(cli, "_load_engine", fake_load_engine)

    with pytest.raises(SystemExit) as error:
        cli.main([
            "batch",
            str(source),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "output.jsonl"),
            "--model",
            "stighellemans/meddeid-dutch-synth",
        ])

    assert error.value.code == 2
    assert model_loaded is False
    assert "either positionally or with `--input`, not both" in capsys.readouterr().err


def test_cli_lists_public_model_scope(capsys) -> None:
    assert cli.main(["models"]) == 0

    output = capsys.readouterr().out
    assert "stighellemans/meddeid-dutch-synth" in output
    assert "stighellemans/meddeid-english-synth" in output
    assert "trained only on synthetic notes" in output
    assert "representative local data" in output
    assert "use one of the model IDs above as the --model value" in output
    assert "meddeid deidentify, batch, or model-info" in output


def test_cli_rejects_ambiguous_public_english_profile() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([
            "deidentify",
            "note.txt",
            "--model",
            "stighellemans/meddeid-english-synth",
            "--quiet",
        ])

    assert error.value.code == 2


def test_cli_status_is_flushed_immediately(monkeypatch) -> None:
    from meddeid.api import Deidentifier

    writes = []

    def fake_print(*values, **kwargs):
        writes.append((values, kwargs))

    monkeypatch.setattr(cli, "print", fake_print, raising=False)
    monkeypatch.setattr(
        Deidentifier,
        "from_pretrained",
        lambda *_args, on_status, **_kwargs: (on_status("checking cache"), FakeEngine())[1],
    )
    args = argparse.Namespace(
        model="stighellemans/meddeid-dutch-synth",
        language_profile=None,
        quiet=False,
        revision=None,
        cache_dir=None,
        offline=False,
        device="cpu",
        backend="torch",
        triton_url=None,
        triton_timeout=30.0,
        window_batch_size=None,
        age_granularity_config=None,
        min_recommended_date_shift_days=366,
    )

    cli._load_engine(args)

    assert writes
    assert all(kwargs.get("flush") is True for _values, kwargs in writes)
