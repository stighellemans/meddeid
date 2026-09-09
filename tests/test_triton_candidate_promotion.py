from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from meddeid import triton_artifact

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "deploy/triton/release.json"


def load_verifier():
    path = ROOT / "deploy/verify_triton_candidate_artifact.py"
    spec = importlib.util.spec_from_file_location("candidate_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


def describe(path: Path, repository: Path) -> dict[str, object]:
    contents = path.read_bytes()
    return {
        "path": str(path.relative_to(repository)),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes": len(contents),
    }


def write_candidate(repository: Path, evidence: Path, source_commit: str) -> None:
    catalog = triton_artifact.load_catalog(CATALOG)
    plan = next(
        item
        for item in catalog["plans"]
        if item["target"] == "t4-sm75"
        and item["model_key"] == "dutch_synthetic"
    )
    model_root = repository / "meddeid-dutch-synth"
    paths = {
        "config": model_root / "config.pbtxt",
        "latency": model_root / "configs/latency.pbtxt",
        "throughput": model_root / "configs/throughput.pbtxt",
        "plan": model_root / "1/model.plan",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"candidate-{name}".encode())
    manifest = {
        "schema": triton_artifact.BUILD_MANIFEST_SCHEMA,
        "release": {"suite_version": "0.3.0", "meddeid_version": "0.4.0"},
        "runtime": {"triton_stack": "26.07"},
        "model": {
            "id": plan["model"],
            "revision": plan["revision"],
            "bundle_sha256": plan["bundle_sha256"],
            "language_profiles": plan["language_profiles"],
        },
        "target": {
            "id": "t4-sm75",
            "release_status": "ready",
            "artifact_repository": plan["artifact"].rsplit(":", 1)[0],
            "gpu_name": "NVIDIA T4",
            "compute_capability": "7.5",
        },
        "artifacts": {
            "config": describe(paths["config"], repository),
            "config_profiles": {
                "latency": describe(paths["latency"], repository),
                "throughput": describe(paths["throughput"], repository),
            },
            "plan": describe(paths["plan"], repository),
        },
    }
    (repository / "build-manifest.json").write_text(json.dumps(manifest))

    evidence.mkdir(parents=True)
    fixture_sha256 = "f" * 64
    parity = {
        "passed": True,
        "strict_passed": False,
        "complete": True,
        "documents": 300,
        "fixture_documents": 300,
        "checked_documents": 300,
        "fixture_sha256": fixture_sha256,
        "errors": [],
        "model_identity_matches": True,
        "source_commit": source_commit,
        "semantic_differences": [{"document_id": "report-only"}],
    }
    preflight = {
        "passed": True,
        "documents": 300,
        "fixture_sha256": fixture_sha256,
        "errors": [],
        "source_commit": source_commit,
    }
    (evidence / "parity-report.json").write_text(json.dumps(parity))
    (evidence / "fixture-contract-preflight.json").write_text(
        json.dumps(preflight)
    )


def test_candidate_verifier_accepts_report_only_semantic_differences(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    evidence = tmp_path / "evidence"
    source_commit = "a" * 40
    write_candidate(repository, evidence, source_commit)

    result = verifier.verify_candidate(
        catalog_path=CATALOG,
        repository=repository,
        evidence=evidence,
        target="t4-sm75",
        model_key="dutch_synthetic",
        source_commit=source_commit,
        minimum_documents=300,
    )

    assert result["strict_passed"] is False
    assert result["semantic_differences"] == 1


def test_candidate_verifier_rejects_technical_errors(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    evidence = tmp_path / "evidence"
    source_commit = "a" * 40
    write_candidate(repository, evidence, source_commit)
    parity_path = evidence / "parity-report.json"
    parity = json.loads(parity_path.read_text())
    parity["errors"] = ["request failed"]
    parity_path.write_text(json.dumps(parity))

    with pytest.raises(ValueError, match="technical errors"):
        verifier.verify_candidate(
            catalog_path=CATALOG,
            repository=repository,
            evidence=evidence,
            target="t4-sm75",
            model_key="dutch_synthetic",
            source_commit=source_commit,
            minimum_documents=300,
        )
