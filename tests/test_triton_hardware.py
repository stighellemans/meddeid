import json
import runpy
import sys
from pathlib import Path

import pytest

from meddeid.triton_hardware import family_for_gpu, plan_supports_gpu, validate_family
from meddeid import triton_artifact
from test_triton_artifact import CATALOG, write_repository


@pytest.mark.parametrize("cc,family", [
    ("7.5", "turing"), ("8.0", "ampere-plus"), ("8.6", "ampere-plus"),
    ("8.9", "ampere-plus"), ("9.0", "ampere-plus"), ("12.0", "ampere-plus"),
])
def test_family_detection(cc, family):
    assert family_for_gpu(cc) == family


@pytest.mark.parametrize("cc", ["7.0", "6.1", "7.9", "", "8", "invalid"])
def test_unsupported_or_invalid_capabilities(cc):
    with pytest.raises(ValueError):
        family_for_gpu(cc)


def family_target(family):
    return {
        "id": family, "family": family,
        "compute_capability": "7.5" if family == "turing" else "8.0",
        "gpu_name": "NVIDIA T4" if family == "turing" else "NVIDIA A100",
        "compatibility_mode": "sameComputeCapability" if family == "turing" else "ampere+",
    }


def test_native_plan_is_not_implicitly_portable():
    assert not plan_supports_gpu(
        {"id": "t4-sm75", "gpu_name": "NVIDIA T4", "compute_capability": "7.5"},
        "NVIDIA GeForce RTX 2080", "7.5",
    )


def test_family_requires_explicit_mode():
    target = family_target("turing")
    target.pop("compatibility_mode")
    with pytest.raises(ValueError, match="compatibility mode"):
        validate_family(target)


@pytest.mark.parametrize("family,name,cc", [
    ("turing", "NVIDIA GeForce RTX 2080", "7.5"),
    ("ampere-plus", "NVIDIA GeForce RTX 4090", "8.9"),
    ("ampere-plus", "NVIDIA GeForce RTX 5090", "12.0"),
])
def test_local_family_plan_accepts_covered_gpu(tmp_path, family, name, cc):
    write_repository(tmp_path, model="/input-model", revision="")
    manifest_path = tmp_path / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["target"] = family_target(family)
    manifest_path.write_text(json.dumps(manifest))
    assert triton_artifact.prepare_plan(
        triton_artifact.load_catalog(CATALOG), hardware=None,
        model="/input-model", revision="", language_profile="nl-BE",
        output=tmp_path, gpu=triton_artifact.NvidiaGpu(name, cc),
    ) == "local"


def test_turing_does_not_accept_ampere_and_vice_versa():
    assert not plan_supports_gpu(family_target("turing"), "NVIDIA A100", "8.0")
    assert not plan_supports_gpu(family_target("ampere-plus"), "NVIDIA T4", "7.5")


def test_family_plan_selection_for_previously_unknown_rtx():
    catalog = triton_artifact.load_catalog(CATALOG)
    plan = {**catalog["plans"][0], "hardware": "ampere-plus", "target": "ampere-plus"}
    catalog["plans"].append(plan)
    selection = triton_artifact.select_plan(
        catalog, hardware="nvidia-geforce-rtx-4090", model=plan["model"],
        revision=plan["revision"], language_profile="nl-BE",
        gpu=triton_artifact.NvidiaGpu("NVIDIA GeForce RTX 4090", "8.9"),
    )
    assert selection.family == "ampere-plus"
    assert selection.target == "ampere-plus"


def test_unbuilt_family_has_no_published_artifact():
    catalog = triton_artifact.load_catalog(CATALOG)
    assert not any(p["hardware"] in {"turing", "ampere-plus"} for p in catalog["plans"])


@pytest.mark.parametrize("family,status", [("native", "ready"), ("turing", "local")])
def test_manifest_does_not_inherit_native_release_approval(tmp_path, monkeypatch, family, status):
    root = Path(__file__).resolve().parents[1]
    model = tmp_path / "model"
    (model / "1").mkdir(parents=True)
    (model / "configs").mkdir()
    (model / "1/model.plan").write_bytes(b"manifest-test-only")
    for config in ("config.pbtxt", "configs/latency.pbtxt", "configs/throughput.pbtxt"):
        (model / config).write_text("test config")
    arguments = {
        "repository": str(tmp_path), "model-name": "model", "model-version": "1",
        "model-id": "owner/model", "model-revision": "a" * 40,
        "bundle-sha256": "b" * 64, "language-profile": "nl-BE",
        "suite-version": "0.2.0", "meddeid-version": "0.3.0",
        "triton-stack": "26.07", "triton-server-version": "2.71.0",
        "tensorrt-version": "11.1.0.106", "builder-image": "test-builder",
        "gpu-target": "t4-sm75", "hardware-family": family,
        "gpu-name": "NVIDIA T4", "compute-capability": "7.5",
        "driver-version": "610.57.04", "precision": "fp16",
        "output-precision": "fp32", "min-shape": "1x8", "opt-shape": "16x256",
        "max-shape": "64x512", "throughput-dynamic-batching": "false",
    }
    script = root / "deploy/write_triton_manifest.py"
    monkeypatch.syspath_prepend(str(root / "deploy"))
    monkeypatch.setattr(sys, "argv", [str(script), *[
        item for key, value in arguments.items() for item in (f"--{key}", value)
    ]])
    runpy.run_path(str(script), run_name="__main__")
    target = json.loads((tmp_path / "build-manifest.json").read_text())["target"]
    assert target["release_status"] == status
    if family != "native":
        assert target["artifact_repository"] == ""
        assert target["family"] == family
        assert target["compatibility_mode"] == "sameComputeCapability"
