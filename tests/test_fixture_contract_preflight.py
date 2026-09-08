"""Catch invalid fixture requests before spending time on GPU image builds."""
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

preflight = runpy.run_path(str(Path(__file__).resolve().parents[1] / "deploy/preflight_benchmark_contract.py"))
validate = preflight["validate_fixture"]


def fixture(tmp_path, **updates):
    path = tmp_path / "fixture.jsonl"
    row = {"document_id": "english-1", "text": "Synthetic note",
           "metadata": {"lang": "en-GB", "generation_method": "benchmark-only"}}
    path.write_text(json.dumps({**row, **updates}) + "\n")
    return path


def test_real_api_accepts_normalized_provenance_and_metadata_json(tmp_path):
    path = fixture(tmp_path, metadata=None, metadata_json=json.dumps({
        "lang": "en-US", "patient": {"given_name": "Ada"}, "synthetic": True}))
    report = validate(path, profiles=["en-GB", "en-US"])
    assert report["passed"] and report["checked_requests"] == 3
    assert report["scope"] == "request-contract-only-no-inference"


@pytest.mark.parametrize("updates,options,status", [
    ({"metadata": {"patient": {"unsupported": "secret"}}}, {}, 422),
    ({"metadata": {"date_shift_days": "366"}}, {}, 422),
    ({"metadata": {"known_values": [{"value": "Ada", "label": "invalid"}]}}, {}, 422),
    ({"metadata": {"lang": "nl-BE"}}, {}, 422),
    ({"text": "   "}, {}, 422),
    ({}, {"max_input_chars": 3}, 422),
    ({}, {"max_batch_chars": 3}, 413),
    ({}, {"max_request_bytes": 3}, 413),
])
def test_rejects_invalid_requests_before_inference(tmp_path, updates, options, status):
    report = validate(fixture(tmp_path, **updates), profiles=["en-GB"], **options)
    assert not report["passed"]
    assert report["errors"][0]["status"] == status
    assert "secret" not in json.dumps(report)


def test_rejects_duplicate_document_ids(tmp_path):
    path = fixture(tmp_path)
    path.write_text(path.read_text() * 2)
    report = validate(path, profiles=["en-GB"])
    assert not report["passed"] and report["checked_requests"] == 0


def test_detects_loader_drift(tmp_path, monkeypatch):
    monkeypatch.setitem(validate.__globals__, "load_batches", lambda *a, **kw: [])
    report = validate(fixture(tmp_path), profiles=["en-GB"])
    assert not report["passed"]
    assert "loaders disagree" in report["errors"][0]["message"]


def test_direct_cli_runs_without_loading_a_model(tmp_path):
    output = tmp_path / "report.json"
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "deploy/preflight_benchmark_contract.py"),
        str(fixture(tmp_path)), "--language-profiles", "en-GB", "--output", str(output),
    ], check=True, capture_output=True, timeout=30)
    assert json.loads(output.read_text())["passed"]
