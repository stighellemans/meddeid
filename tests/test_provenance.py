from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from meddeid.api import DeidentificationResult, Deidentifier


def test_result_contract_is_flat_and_logically_ordered() -> None:
    result = DeidentificationResult(
        text="Jan",
        deid_text="[Name:Patient]",
        spans=[{"begin": 0, "end": 3, "label": "Name:Patient"}],
        processing={"date_replacement": {}},
        warnings=[],
        provenance={"contract_version": "meddeid.inference-provenance.v1"},
    )

    contract = result.to_contract()

    assert list(contract) == [
        "deid_text",
        "spans",
        "processing",
        "warnings",
        "provenance",
    ]
    assert "output" not in contract
    assert "language_profile" not in contract


def test_consumer_provenance_contains_only_essential_identity() -> None:
    engine = object.__new__(Deidentifier)
    engine._provenance_packages = {"meddeid": "0.3.0"}
    engine.resolved_revision = "abc123"
    engine.bundle = SimpleNamespace(
        name="test-model",
        model_version="1",
        contract_hash=lambda: "a" * 64,
    )

    provenance = engine.inference_provenance("nl-BE")

    assert provenance == {
        "contract_version": "meddeid.inference-provenance.v1",
        "software": {"name": "meddeid", "version": "0.3.0"},
        "model": {
            "name": "test-model",
            "version": "1",
            "resolved_revision": "abc123",
            "bundle_sha256": "a" * 64,
        },
        "language_profile": {"profile_id": "nl-BE"},
    }
    assert "runtime" not in provenance
    assert "source" not in provenance["model"]
    assert "model_files" not in provenance


def test_model_file_inventory_is_an_administrator_view(tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.json"
    weights = tmp_path / "model.safetensors"
    tokenizer = tmp_path / "tokenizer.json"
    manifest.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"weights")
    tokenizer.write_text("{}", encoding="utf-8")
    engine = object.__new__(Deidentifier)
    engine.bundle = SimpleNamespace(
        manifest_path=manifest,
        checkpoint_path=weights,
    )
    engine.model_source_is_local = False
    engine.offline = True

    inventory = engine._model_file_inventory()

    assert inventory["loaded_from"] == "huggingface-cache"
    assert inventory["offline_enforced"] is True
    assert inventory["root_path"] == str(tmp_path)
    assert inventory["manifest_path"] == str(manifest)
    assert inventory["weights_path"] == str(weights)
    assert inventory["weights_present"] is True
    assert [item["relative_path"] for item in inventory["files"]] == [
        "bundle.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    assert all(Path(item["path"]).is_absolute() for item in inventory["files"])


def test_requested_revision_distinguishes_latest_and_local() -> None:
    engine = object.__new__(Deidentifier)
    engine.requested_revision = None
    engine.model_source_is_local = False
    assert engine._requested_revision() == "latest"

    engine.model_source_is_local = True
    assert engine._requested_revision() == "local"

    engine.requested_revision = "abc123"
    assert engine._requested_revision() == "abc123"
