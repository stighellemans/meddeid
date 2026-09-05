import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


targets = load_script("triton_targets", "deploy/triton_targets.py")
lengths = load_script("make_benchmark_lengths", "deploy/make_benchmark_lengths.py")


def test_target_catalog_has_one_ready_default_and_requestable_candidates() -> None:
    catalog = targets.load_catalog()
    by_id = {target["id"]: target for target in catalog["targets"]}

    assert catalog["default_target"] == "t4-sm75"
    assert by_id["t4-sm75"]["release_status"] == "ready"
    assert by_id["t4-sm75"]["compute_capability"] == "7.5"
    assert by_id["a10g-sm86"]["release_status"] == "on-request"
    assert by_id["l4-sm89"]["release_status"] == "on-request"
    assert [
        target["id"]
        for target in catalog["targets"]
        if target["release_status"] == "ready"
    ] == ["t4-sm75"]


def test_target_host_validation_checks_name_and_compute_capability() -> None:
    target = targets.get_target(targets.load_catalog(), "t4-sm75")

    targets.verify_host(
        target,
        gpu_name="NVIDIA Tesla T4",
        compute_capability="7.5",
    )
    with pytest.raises(ValueError, match="compute capability"):
        targets.verify_host(
            target,
            gpu_name="NVIDIA L4",
            compute_capability="8.9",
        )
    with pytest.raises(ValueError, match="GPU name"):
        targets.verify_host(
            target,
            gpu_name="NVIDIA RTX 2080",
            compute_capability="7.5",
        )


def test_target_image_tag_keeps_gpu_and_runtime_identity_visible() -> None:
    target = targets.get_target(targets.load_catalog(), "l4-sm89")

    assert targets.image_tag(
        target,
        version="0.3.0",
        triton_stack="26.07",
        precision="fp16",
    ) == ("ghcr.io/stighellemans/meddeid-triton-l4-sm89:0.3.0-trt26.07-fp16")


def test_manifest_is_bound_to_the_reviewed_target_spec(tmp_path: Path) -> None:
    target = targets.get_target(targets.load_catalog(), "a10g-sm86")
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": targets.BUILD_MANIFEST_SCHEMA,
                "target": {
                    "id": target["id"],
                    "display_name": target["display_name"],
                    "catalog_spec_sha256": targets.target_spec_sha256(target),
                    "release_status": target["release_status"],
                    "image_repository": target["image_repository"],
                    "compute_capability": target["compute_capability"],
                },
            }
        ),
        encoding="utf-8",
    )

    targets.verify_manifest(target, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["target"]["release_status"] = "ready"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="release_status"):
        targets.verify_manifest(target, manifest)


def test_length_fixture_generator_produces_exact_public_test_shapes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "document_id": "synthetic",
                "text": "Synthetic clinical note.",
                "metadata": {"lang": "nl-BE"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = lengths.read_rows(source)
    generated = [lengths.repeat_to_length(rows[0]["text"], size) for size in (8, 31)]

    assert [len(text) for text in generated] == [8, 31]
    assert generated[0] == "Syntheti"


def test_gpu_workflow_is_target_driven_and_publishes_only_ready_targets() -> None:
    workflow = (ROOT / ".github/workflows/triton-gpu-validation.yml").read_text()
    preflight = (ROOT / "deploy/preflight_triton_host.sh").read_text()
    manifest_writer = (ROOT / "deploy/write_triton_manifest.py").read_text()

    assert "inputs.gpu_target || 't4-sm75'" in workflow
    assert '"${GPU_TARGET}" 0' in workflow
    assert '"${target_status}" != ready' in workflow
    assert 'triton_name="${TRITON_IMAGE_REPOSITORY}"' in workflow
    assert 'triton_targets.py" verify-host' in preflight
    assert 'case "${target}"' not in preflight
    assert "BUILD_MANIFEST_SCHEMA" in manifest_writer
    assert '"catalog_spec_sha256"' in manifest_writer


def test_mps_summary_records_parity_and_measured_recommendations() -> None:
    summary = json.loads(
        (ROOT / "deploy/mps/benchmark-summary.json").read_text(encoding="utf-8")
    )

    assert summary["schema"] == "meddeid.mps-benchmark-summary.v1"
    assert summary["host"]["chip"] == "Apple M4 Pro"
    assert summary["parity"] == {
        "passed": True,
        "documents": 300,
        "semantic_differences": 0,
        "confidence_fields_excluded": True,
    }
    eager = summary["eager_medians"]
    assert all(case["mps_speedup"] > 1 for case in eager.values())
    assert (
        eager["etl_batch_16"]["mps_documents_per_second"]
        > eager["etl_batch_32"]["mps_documents_per_second"]
    )
    assert (
        summary["throughput_microbatch_medians"][
            "interactive_short_burst_change_percent"
        ]
        > 0
    )
    assert summary["compile_experiment"]["recommendation"] == "off"


def test_t4_summary_records_target_bound_parity_size_and_performance() -> None:
    summary = json.loads(
        (ROOT / "deploy/triton/t4-benchmark-summary.json").read_text(
            encoding="utf-8"
        )
    )
    catalog_target = targets.get_target(targets.load_catalog(), "t4-sm75")

    assert summary["schema"] == "meddeid.triton-benchmark-summary.v1"
    assert summary["target"]["id"] == catalog_target["id"]
    assert summary["target"]["compute_capability"] == catalog_target[
        "compute_capability"
    ]
    assert catalog_target["release_status"] == "ready"
    assert summary["parity"] == {
        "passed": True,
        "documents": 300,
        "semantic_differences": 0,
        "confidence_fields_excluded": True,
    }
    artifacts = summary["artifacts"]
    assert (
        artifacts["tensorrt_gateway_pair"]["documents_per_second"]
        > artifacts["pytorch_cuda_fp16_eager"]["documents_per_second"]
        > artifacts["cpu"]["documents_per_second"]
    )
    assert (
        artifacts["tensorrt_gateway_pair"]["pull_size_proxy_gb"]
        < artifacts["pytorch_cuda_fp16_eager"]["pull_size_proxy_gb"]
    )
    assert summary["compile_experiment"]["recommendation"] == "fp16-eager"
    assert summary["scheduler_recommendation"]["triton_dynamic_batching"] is False
