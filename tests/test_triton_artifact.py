from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from meddeid import triton_artifact

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "deploy/triton/release.json"


def write_repository(path: Path, *, model: str, revision: str) -> None:
    files = {
        "meddeid-dutch-synth/config.pbtxt": b"name: test\n",
        "meddeid-dutch-synth/configs/latency.pbtxt": b"name: latency\n",
        "meddeid-dutch-synth/configs/throughput.pbtxt": b"name: throughput\n",
        "meddeid-dutch-synth/1/model.plan": b"compiled-weights",
    }
    for relative, contents in files.items():
        destination = path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

    def description(relative: str) -> dict[str, object]:
        contents = files[relative]
        return {
            "path": relative,
            "sha256": hashlib.sha256(contents).hexdigest(),
            "bytes": len(contents),
        }

    manifest = {
        "schema": triton_artifact.BUILD_MANIFEST_SCHEMA,
        "release": {"suite_version": "0.3.0", "meddeid_version": "0.4.0"},
        "runtime": {"triton_stack": "26.07", "tensorrt_version": "11.1.0.106"},
        "model": {
            "id": model,
            "revision": revision,
            "bundle_sha256": "0" * 64,
            "language_profiles": ["nl-BE"],
        },
        "target": {
            "id": "t4-sm75",
            "gpu_name": "NVIDIA T4",
            "compute_capability": "7.5",
        },
        "artifacts": {
            "config": description("meddeid-dutch-synth/config.pbtxt"),
            "config_profiles": {
                "latency": description("meddeid-dutch-synth/configs/latency.pbtxt"),
                "throughput": description(
                    "meddeid-dutch-synth/configs/throughput.pbtxt"
                ),
            },
            "plan": description("meddeid-dutch-synth/1/model.plan"),
        },
    }
    (path / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_release_catalog_resolves_hardware_model_and_revision_together() -> None:
    catalog = triton_artifact.load_catalog(CATALOG)
    selection = triton_artifact.select_plan(
        catalog,
        hardware="t4",
        model="stighellemans/meddeid-dutch-synth",
        revision="1f20655454dcbd042647cacdfff6b6802a970959",
        language_profile="nl-BE",
    )

    assert selection.suite_version == "0.3.0"
    assert selection.meddeid_version == "0.4.0"
    assert selection.target == "t4-sm75"
    assert selection.artifact.endswith(":0.4.0-trt26.07-fp16-dutch-synthetic")
    assert selection.bundle_sha256 == (
        "001123dbc1184751e4bae3c29633d14a5e041acae0c173773f186127ae3fd944"
    )


def test_release_catalog_exposes_both_public_models_on_t4() -> None:
    catalog = triton_artifact.load_catalog(CATALOG)
    selection = triton_artifact.select_plan(
        catalog,
        hardware="t4",
        model="stighellemans/meddeid-english-synth",
        revision="e519d454250ba67bef62c0063dbbf242c329b750",
        language_profile="en-US",
    )

    assert selection.target == "t4-sm75"
    assert selection.language_profiles == ("en-GB", "en-US")
    assert selection.artifact.endswith(":0.4.0-trt26.07-fp16-english-synthetic")


@pytest.mark.parametrize("gpu_name", ["NVIDIA T4", "Tesla T4", "NVIDIA Tesla T4"])
def test_release_catalog_detects_t4_from_nvidia_name(gpu_name: str) -> None:
    catalog = triton_artifact.load_catalog(CATALOG)

    hardware = triton_artifact.detect_hardware(
        catalog,
        triton_artifact.NvidiaGpu(gpu_name, "7.5"),
    )
    assert hardware == gpu_name.lower().replace(" ", "-")
    selection = triton_artifact.select_plan(
        catalog, hardware=hardware,
        model=catalog["plans"][0]["model"],
        revision=catalog["plans"][0]["revision"], language_profile="nl-BE",
    )
    assert selection.target == "t4-sm75"  # Existing published artifact still resolves.


def test_unknown_gpu_is_recognized_without_a_catalog_entry() -> None:
    catalog = triton_artifact.load_catalog(CATALOG)

    assert triton_artifact.detect_hardware(
        catalog,
        triton_artifact.NvidiaGpu("NVIDIA RTX 6000 Ada Generation", "8.9"),
    ) == "nvidia-rtx-6000-ada-generation"


def test_public_model_on_unpublished_gpu_only_offers_local_build(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="detected NVIDIA A10G.*Build a local plan",
    ):
        triton_artifact.prepare_plan(
            triton_artifact.load_catalog(CATALOG),
            hardware=None,
            model="stighellemans/meddeid-dutch-synth",
            revision="1f20655454dcbd042647cacdfff6b6802a970959",
            language_profile="nl-BE",
            output=tmp_path / "model_repository",
            gpu=triton_artifact.NvidiaGpu("NVIDIA A10G", "8.6"),
        )


def test_private_model_on_unlisted_gpu_only_offers_local_build(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as captured:
        triton_artifact.prepare_plan(
            triton_artifact.load_catalog(CATALOG),
            hardware=None,
            model="hospital/private-model",
            revision="a" * 40,
            language_profile="nl-BE",
            output=tmp_path / "model_repository",
            gpu=triton_artifact.NvidiaGpu(
                "NVIDIA RTX 6000 Ada Generation",
                "8.9",
            ),
        )

    assert "Build a local plan" in str(captured.value)
    assert "help add optimized support" not in str(captured.value)


def test_unknown_public_model_revision_does_not_invite_gpu_contribution(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as captured:
        triton_artifact.prepare_plan(
            triton_artifact.load_catalog(CATALOG),
            hardware=None,
            model="stighellemans/meddeid-dutch-synth",
            revision="a" * 40,
            language_profile="nl-BE",
            output=tmp_path / "model_repository",
            gpu=triton_artifact.NvidiaGpu("NVIDIA A10G", "8.6"),
        )

    assert "Build a local plan" in str(captured.value)
    assert "help add optimized support" not in str(captured.value)


def test_release_catalog_and_compose_pin_the_same_weight_free_images() -> None:
    catalog = triton_artifact.load_catalog(CATALOG)
    compose = (ROOT / "compose.triton.yaml").read_text(encoding="utf-8")

    assert catalog["images"]["gateway"] in compose
    assert set(catalog["images"]["runtimes"].values()) == {
        "ghcr.io/stighellemans/meddeid-triton-runtime:0.4.0-trt26.07"
    }
    assert next(iter(catalog["images"]["runtimes"].values())) in compose


def test_release_catalog_rejects_unpublished_model_plan() -> None:
    catalog = triton_artifact.load_catalog(CATALOG)
    with pytest.raises(ValueError, match="Build a local plan on this server"):
        triton_artifact.select_plan(
            catalog,
            hardware="t4",
            model="hospital/private-model",
            revision="a" * 40,
            language_profile="nl-BE",
        )


def test_local_repository_is_verified_without_a_published_plan(tmp_path: Path) -> None:
    repository = tmp_path / "model_repository"
    repository.mkdir()
    write_repository(
        repository,
        model="hospital/private-model",
        revision="a" * 40,
    )
    result = triton_artifact.prepare_plan(
        triton_artifact.load_catalog(CATALOG),
        hardware="t4",
        model="hospital/private-model",
        revision="a" * 40,
        language_profile="nl-BE",
        output=repository,
    )

    assert result == "local"


def test_local_repository_for_an_unlisted_gpu_is_accepted(tmp_path: Path) -> None:
    repository = tmp_path / "model_repository"
    repository.mkdir()
    write_repository(
        repository,
        model="hospital/private-model",
        revision="a" * 40,
    )
    manifest_path = repository / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target"] = {
        "id": "local-nvidia-rtx-6000-ada-generation-sm89",
        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
        "compute_capability": "8.9",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = triton_artifact.prepare_plan(
        triton_artifact.load_catalog(CATALOG),
        hardware=None,
        model="hospital/private-model",
        revision="a" * 40,
        language_profile="nl-BE",
        output=repository,
        gpu=triton_artifact.NvidiaGpu("NVIDIA RTX 6000 Ada Generation", "8.9"),
    )

    assert result == "local"


def test_repository_pack_is_reproducible_and_verified(tmp_path: Path) -> None:
    repository = tmp_path / "model_repository"
    repository.mkdir()
    write_repository(
        repository,
        model="stighellemans/meddeid-dutch-synth",
        revision="1f20655454dcbd042647cacdfff6b6802a970959",
    )
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"

    triton_artifact.verify_repository(repository)
    assert triton_artifact.pack_repository(repository, first) == (
        triton_artifact.pack_repository(repository, second)
    )
    assert first.read_bytes() == second.read_bytes()


def test_plan_install_preserves_mount_directory(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    write_repository(source, model="hospital/model", revision="a" * 40)
    archive = tmp_path / "plan.tar.gz"
    triton_artifact.pack_repository(source, archive)
    monkeypatch.setattr(
        triton_artifact.RegistryClient, "pull",
        lambda self, ref: (archive.read_bytes(), "sha256:" + "a" * 64),
    )
    selection = triton_artifact.request_selection(
        triton_artifact.load_catalog(CATALOG), hardware="t4",
        model="hospital/model", revision="a" * 40, language_profile="nl-BE",
    )
    output = tmp_path / "mounted-models"
    output.mkdir()
    inode = output.stat().st_ino
    triton_artifact.fetch_plan(selection, output, "ghcr.io/example/model:test")
    assert output.stat().st_ino == inode
    triton_artifact.verify_repository(output, selection)


def test_local_plan_rejects_runtime_from_another_release(tmp_path: Path):
    repository = tmp_path / "models"
    repository.mkdir()
    write_repository(repository, model="hospital/model", revision="a" * 40)
    manifest_file = repository / "build-manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["runtime"]["tensorrt_version"] = "incompatible"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match the runtime stack"):
        triton_artifact.prepare_plan(
            triton_artifact.load_catalog(CATALOG), hardware=None,
            model="hospital/model", revision="a" * 40, language_profile="nl-BE",
            output=repository, gpu=triton_artifact.NvidiaGpu("NVIDIA T4", "7.5"),
        )


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (
            "ghcr.io/stighellemans/plan:0.3.0",
            ("ghcr.io", "stighellemans/plan", "0.3.0"),
        ),
        (
            f"ghcr.io/stighellemans/plan@sha256:{'a' * 64}",
            ("ghcr.io", "stighellemans/plan", f"sha256:{'a' * 64}"),
        ),
    ],
)
def test_oci_reference_parsing(reference: str, expected: tuple[str, str, str]) -> None:
    assert triton_artifact.parse_oci_reference(reference) == expected
