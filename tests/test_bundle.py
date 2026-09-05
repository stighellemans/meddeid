from __future__ import annotations

import json

import pytest

from meddeid.bundle import load_model_bundle, load_model_release_input
from meddeid_core.taxonomy import BERT_ENTITY_LABELS


def write_bundle(path, **overrides) -> None:
    payload = {
        "bundle_version": "deid-bundle.v1",
        "artifact_version": "2026-04-20",
        "model_version": "1",
        "name": "deid-token-classifier",
        "base_encoder": "example/encoder",
        "task": "dual_head_token_classification",
        "hidden_size": 768,
        "labels": {
            "bio": ["O", "B", "I"],
            "entity": list(BERT_ENTITY_LABELS),
        },
        "inference": {
            "max_length": 256,
            "overlap": 64,
            "min_entity_score": 0.0,
        },
        "postprocess": {
            "profiles": [{"profile_id": "nl-BE"}],
            "profile_selection": "bundle_default",
        },
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_model_bundle_reads_contract_without_source_validation(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path)

    bundle = load_model_bundle(bundle_path, validate_package=False)

    assert bundle.name == "deid-token-classifier"
    assert bundle.model_version == "1"
    assert bundle.bio_labels == ["O", "B", "I"]
    assert bundle.entity_labels == list(BERT_ENTITY_LABELS)
    assert bundle.inference.max_length == 256
    assert bundle.postprocess.profile_id == "nl-BE"
    assert bundle.checkpoint_path == tmp_path / "model.pt"
    assert bundle.weights_format == "pt"
    assert bundle.encoder_config_path is None
    assert bundle.tokenizer_path is None
    assert len(bundle.contract_hash()) == 64


def test_load_model_bundle_supports_two_explicit_postprocess_profiles(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path, postprocess={
        "profiles": [
            {"profile_id": "en-GB"},
            {"profile_id": "en-US"},
        ],
        "profile_selection": "explicit",
    })

    bundle = load_model_bundle(bundle_path, validate_package=False)

    assert bundle.postprocess.profile_id is None
    assert [item.profile_id for item in bundle.postprocess.profiles] == ["en-GB", "en-US"]
    assert bundle.postprocess.profile_selection == "explicit"


def test_multi_profile_bundle_rejects_an_implicit_default(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path, postprocess={
        "profiles": [
            {"profile_id": "en-GB"},
            {"profile_id": "en-US"},
        ],
        "profile_selection": "bundle_default",
    })
    with pytest.raises(ValueError, match="requires explicit profile selection"):
        load_model_bundle(bundle_path, validate_package=False)


def test_bundle_rejects_duplicate_normalized_locales(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path, postprocess={
        "profiles": [
            {"profile_id": "en-GB"},
            {"profile_id": "en_GB"},
        ],
        "profile_selection": "explicit",
    })
    with pytest.raises(ValueError, match="duplicate normalized locales"):
        load_model_bundle(bundle_path, validate_package=False)


def test_source_model_dir_contract_allows_hub_documentation_files(tmp_path) -> None:
    source_root = tmp_path / "source"
    model_dir = source_root / "model"
    (source_root / "src" / "meddeid").mkdir(parents=True)
    model_dir.mkdir()
    write_bundle(model_dir / "bundle.json")
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    (model_dir / "README.md").write_text("# model", encoding="utf-8")

    bundle = load_model_bundle(model_dir / "bundle.json")
    assert bundle.checkpoint_path == model_dir / "model.pt"


def test_source_model_dir_contract_requires_one_checkpoint(tmp_path) -> None:
    source_root = tmp_path / "source"
    model_dir = source_root / "model"
    (source_root / "src" / "meddeid").mkdir(parents=True)
    model_dir.mkdir()
    write_bundle(model_dir / "bundle.json")

    with pytest.raises(ValueError, match="missing declared weights"):
        load_model_bundle(model_dir / "bundle.json")


def test_source_model_dir_contract_accepts_onnx_source(tmp_path) -> None:
    source_root = tmp_path / "source"
    model_dir = source_root / "model"
    (source_root / "src" / "meddeid").mkdir(parents=True)
    model_dir.mkdir()
    write_bundle(model_dir / "bundle.json")
    (model_dir / "model.onnx").write_bytes(b"onnx")

    bundle = load_model_bundle(model_dir / "bundle.json")

    assert bundle.checkpoint_path == model_dir / "model.onnx"


def test_model_name_and_version_are_triton_safe(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.json"
    write_bundle(bundle_path, name="bad/name")
    with pytest.raises(ValueError, match="path separators"):
        load_model_bundle(bundle_path, validate_package=False)

    write_bundle(bundle_path, model_version="latest")
    with pytest.raises(ValueError, match="must be numeric"):
        load_model_bundle(bundle_path, validate_package=False)


def test_model_release_input_accepts_one_pt_or_onnx(tmp_path) -> None:
    write_bundle(tmp_path / "bundle.json")
    (tmp_path / "model.onnx").write_bytes(b"onnx")

    release_input = load_model_release_input(tmp_path)

    assert release_input.source_type == "onnx"
    assert release_input.source_path == tmp_path / "model.onnx"
    assert release_input.bundle.name == "deid-token-classifier"


def test_model_release_input_rejects_ambiguous_sources(tmp_path) -> None:
    write_bundle(tmp_path / "bundle.json")
    (tmp_path / "model.pt").write_bytes(b"pt")
    (tmp_path / "model.onnx").write_bytes(b"onnx")

    with pytest.raises(ValueError, match="exactly one model weights"):
        load_model_release_input(tmp_path)


def test_model_release_input_allows_model_card(tmp_path) -> None:
    write_bundle(tmp_path / "bundle.json")
    (tmp_path / "model.pt").write_bytes(b"pt")
    (tmp_path / "README.md").write_text("# model", encoding="utf-8")

    assert load_model_release_input(tmp_path).source_type == "pt"


def test_self_contained_safetensors_bundle_requires_config_and_tokenizer(tmp_path) -> None:
    write_bundle(
        tmp_path / "bundle.json",
        weights={"filename": "model.safetensors", "format": "safetensors"},
        encoder_config="config.json",
        tokenizer_path=".",
    )
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    with pytest.raises(ValueError, match="missing config.json"):
        load_model_bundle(tmp_path / "bundle.json", validate_package=True)

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing tokenizer files"):
        load_model_bundle(tmp_path / "bundle.json", validate_package=True)

    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    bundle = load_model_bundle(tmp_path / "bundle.json", validate_package=True)
    assert bundle.weights_format == "safetensors"
    assert bundle.encoder_config_path == tmp_path / "config.json"
    assert bundle.tokenizer_path == tmp_path


def test_gateway_bundle_can_validate_metadata_without_local_weights(tmp_path) -> None:
    write_bundle(
        tmp_path / "bundle.json",
        weights={"filename": "model.safetensors", "format": "safetensors"},
        encoder_config="config.json",
        tokenizer_path=".",
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    bundle = load_model_bundle(
        tmp_path / "bundle.json",
        validate_package=True,
        require_weights=False,
    )

    assert not bundle.checkpoint_path.exists()
    assert bundle.contract_hash()


def test_hugging_face_style_symlinked_manifest_keeps_snapshot_root(tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / "abc123"
    blobs = tmp_path / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    write_bundle(
        blobs / "manifest-content-hash",
        weights={"filename": "model.safetensors", "format": "safetensors"},
        encoder_config="config.json",
        tokenizer_path=".",
    )
    (snapshot / "bundle.json").symlink_to(blobs / "manifest-content-hash")
    (blobs / "weights-content-hash").write_bytes(b"weights")
    (blobs / "config-content-hash").write_text("{}", encoding="utf-8")
    (blobs / "tokenizer-content-hash").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").symlink_to(blobs / "weights-content-hash")
    (snapshot / "config.json").symlink_to(blobs / "config-content-hash")
    (snapshot / "tokenizer.json").symlink_to(blobs / "tokenizer-content-hash")

    bundle = load_model_bundle(snapshot / "bundle.json", validate_package=True)

    assert bundle.manifest_path == snapshot / "bundle.json"
    assert bundle.checkpoint_path == snapshot / "model.safetensors"
    assert bundle.encoder_config_path == snapshot / "config.json"
    assert bundle.tokenizer_path == snapshot
