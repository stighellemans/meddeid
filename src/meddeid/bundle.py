from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from meddeid_core.taxonomy import BERT_ENTITY_LABELS


@dataclass(frozen=True)
class InferenceConfig:
    """Windowing and threshold values that must match model training/evaluation."""

    max_length: int
    overlap: int
    min_entity_score: float


@dataclass(frozen=True)
class LanguageProfileRef:
    """One supported postprocessing profile."""

    profile_id: str


@dataclass(frozen=True)
class PostprocessConfig:
    """Language profiles supported by one model bundle."""

    profiles: tuple[LanguageProfileRef, ...]
    profile_selection: str = "bundle_default"

    @property
    def profile_id(self) -> str | None:
        return self.profiles[0].profile_id if len(self.profiles) == 1 else None

@dataclass(frozen=True)
class ModelBundle:
    """Complete metadata contract required to make a raw checkpoint serveable.

    The weights file only tells the runtime which tensor values to load. This
    bundle supplies the operational context: local encoder/tokenizer assets,
    head sizes, label order and decoding settings.
    """

    manifest_path: Path
    bundle_version: str
    artifact_version: str
    model_version: str
    name: str
    task: str
    base_encoder: str
    base_encoder_revision: str | None
    hidden_size: int
    checkpoint_path: Path
    weights_format: str
    encoder_config_path: Path | None
    tokenizer_path: Path | None
    bio_labels: list[str]
    entity_labels: list[str]
    inference: InferenceConfig
    postprocess: PostprocessConfig

    @property
    def usable_tokens(self) -> int:
        return self.inference.max_length - 2

    def to_metadata(self) -> dict[str, Any]:
        return {
            "bundle_version": self.bundle_version,
            "artifact_version": self.artifact_version,
            "model_version": self.model_version,
            "name": self.name,
            "task": self.task,
            "base_encoder": self.base_encoder,
            "base_encoder_revision": self.base_encoder_revision,
            "hidden_size": self.hidden_size,
            "weights_format": self.weights_format,
            "bio_labels": list(self.bio_labels),
            "entity_labels": list(self.entity_labels),
            "max_length": self.inference.max_length,
            "overlap": self.inference.overlap,
            "min_entity_score": self.inference.min_entity_score,
            "postprocess_profiles": [
                {"profile_id": item.profile_id}
                for item in self.postprocess.profiles
            ],
            "postprocess_profile_selection": self.postprocess.profile_selection,
        }

    def contract_hash(self) -> str:
        payload = self.to_metadata()
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ModelReleaseInput:
    """Validated model input directory used to prepare runtime artifacts."""

    root: Path
    bundle: ModelBundle
    source_path: Path
    source_type: str


def _validate_model_name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("bundle field 'name' may not be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError("bundle field 'name' may not contain path separators")
    return normalized


def _validate_model_version(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("bundle field 'model_version' may not be empty")
    if not normalized.isdigit():
        raise ValueError("bundle field 'model_version' must be numeric for Triton")
    return normalized


def _is_source_repo_model_dir(model_dir: Path) -> bool:
    return (model_dir.parent / "src" / "meddeid").is_dir()


def _weight_contract(
    payload: dict[str, Any],
    model_dir: Path | None = None,
) -> tuple[str, str]:
    weights = payload.get("weights")
    if isinstance(weights, dict):
        filename = str(weights.get("filename", "")).strip()
        weights_format = str(weights.get("format", "")).strip().lower()
        if not filename or Path(filename).name != filename:
            raise ValueError("bundle weights.filename must be one file in the model root")
        if weights_format not in {"safetensors", "pt", "onnx"}:
            raise ValueError("bundle weights.format must be safetensors, pt, or onnx")
        return filename, weights_format
    if model_dir is not None:
        legacy = [
            path
            for path in model_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".safetensors", ".pt", ".onnx"}
        ]
        if len(legacy) > 1:
            raise ValueError("legacy bundle must contain exactly one model weights file")
        if len(legacy) == 1:
            suffix = legacy[0].suffix.lower()
            return legacy[0].name, {
                ".safetensors": "safetensors",
                ".pt": "pt",
                ".onnx": "onnx",
            }[suffix]
    return "model.pt", "pt"


def _validate_model_dir_contract(manifest_path: Path, payload: dict[str, Any]) -> Path:
    model_dir = manifest_path.parent
    if manifest_path.name != "bundle.json":
        raise ValueError("model package manifest must be named bundle.json")

    filename, weights_format = _weight_contract(payload, model_dir)
    source = model_dir / filename
    expected_suffix = {
        "safetensors": ".safetensors",
        "pt": ".pt",
        "onnx": ".onnx",
    }[weights_format]
    if source.suffix.lower() != expected_suffix:
        raise ValueError("bundle weights filename extension does not match weights.format")
    if not source.is_file():
        raise ValueError(f"model package is missing declared weights file: {filename}")

    if weights_format == "safetensors":
        config_name = str(payload.get("encoder_config", "config.json"))
        tokenizer_name = str(payload.get("tokenizer_path", "."))
        if not (model_dir / config_name).is_file():
            raise ValueError(f"self-contained model package is missing {config_name}")
        tokenizer_dir = (model_dir / tokenizer_name).resolve()
        if not any((tokenizer_dir / name).is_file() for name in ("tokenizer.json", "vocab.json")):
            raise ValueError("self-contained model package is missing tokenizer files")
    # Preserve the declared filename for format detection. Hub snapshots use a
    # symlink whose target is a suffix-less content hash.
    return source.absolute()


def load_model_bundle(path: str | Path, *, validate_package: bool | None = None) -> ModelBundle:
    """Load and validate the model bundle manifest used by all runtimes.

    Current Hub bundles use Safetensors plus local encoder/tokenizer assets.
    Legacy `.pt` and `.onnx` handoffs remain readable when unambiguous.
    """

    # Keep the package-facing path instead of resolving the final symlink.
    # Hugging Face snapshots expose files such as bundle.json as symlinks into
    # a content-addressed blobs directory.  Resolving that link changes both
    # the filename and parent directory, which breaks the package contract and
    # makes sibling weights/tokenizer files impossible to find.
    manifest_path = Path(path).expanduser().absolute()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    for omitted_key in ("source", "serving"):
        if omitted_key in payload:
            raise ValueError(
                f"bundle field '{omitted_key}' is no longer part of the model contract"
            )

    if validate_package is None:
        validate_package = _is_source_repo_model_dir(manifest_path.parent)
    if validate_package:
        checkpoint_path = _validate_model_dir_contract(manifest_path, payload)
    else:
        weights_filename, _ = _weight_contract(payload, manifest_path.parent)
        checkpoint_path = (manifest_path.parent / weights_filename).absolute()

    inference = payload["inference"]
    postprocess = payload["postprocess"]
    labels = payload["labels"]
    name = _validate_model_name(str(payload["name"]))
    model_version = _validate_model_version(str(payload["model_version"]))
    _, weights_format = _weight_contract(payload, manifest_path.parent)
    encoder_config = payload.get("encoder_config")
    tokenizer_path = payload.get("tokenizer_path")
    if weights_format == "safetensors":
        encoder_config = encoder_config or "config.json"
        tokenizer_path = tokenizer_path or "."

    entity_labels = [str(item) for item in labels["entity"]]
    if tuple(entity_labels) != BERT_ENTITY_LABELS:
        raise ValueError(
            "bundle entity labels must exactly match the canonical ordered 14-label head"
        )

    if not isinstance(postprocess.get("profiles"), list):
        raise ValueError("bundle postprocess.profiles must be a list")
    profile_refs = tuple(
        LanguageProfileRef(profile_id=str(item["profile_id"]))
        for item in postprocess["profiles"]
    )
    if not profile_refs:
        raise ValueError("bundle postprocess.profiles must not be empty")
    if len({item.profile_id for item in profile_refs}) != len(profile_refs):
        raise ValueError("bundle postprocess.profiles contains duplicates")
    normalized_profile_ids = {
        item.profile_id.strip().replace("_", "-").lower()
        for item in profile_refs
    }
    if len(normalized_profile_ids) != len(profile_refs):
        raise ValueError("bundle postprocess.profiles contains duplicate normalized locales")
    profile_selection = str(postprocess.get("profile_selection", "explicit"))
    if len(profile_refs) > 1 and profile_selection != "explicit":
        raise ValueError("a multi-profile bundle requires explicit profile selection")

    return ModelBundle(
        manifest_path=manifest_path,
        bundle_version=str(payload["bundle_version"]),
        artifact_version=str(payload["artifact_version"]),
        model_version=model_version,
        name=name,
        task=str(payload["task"]),
        base_encoder=str(payload["base_encoder"]),
        base_encoder_revision=(
            str(payload["base_encoder_revision"])
            if payload.get("base_encoder_revision") is not None
            else None
        ),
        hidden_size=int(payload["hidden_size"]),
        checkpoint_path=checkpoint_path,
        weights_format=weights_format,
        encoder_config_path=(
            (manifest_path.parent / str(encoder_config)).absolute()
            if encoder_config is not None
            else None
        ),
        tokenizer_path=(
            (manifest_path.parent / str(tokenizer_path)).absolute()
            if tokenizer_path is not None
            else None
        ),
        bio_labels=[str(item) for item in labels["bio"]],
        entity_labels=entity_labels,
        inference=InferenceConfig(
            max_length=int(inference["max_length"]),
            overlap=int(inference["overlap"]),
            min_entity_score=float(inference["min_entity_score"]),
        ),
        postprocess=PostprocessConfig(
            profiles=profile_refs,
            profile_selection=profile_selection,
        ),
    )


def load_model_release_input(path: str | Path) -> ModelReleaseInput:
    """Validate a model runtime preparation input directory.

    Current inputs contain `bundle.json`, Safetensors weights, local encoder
    configuration and tokenizer files. Legacy `.pt` and `.onnx` source bundles
    remain accepted when their source file is unambiguous.
    """

    root = Path(path).expanduser().resolve()
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise ValueError(f"model input directory must contain bundle.json: {root}")

    bundle = load_model_bundle(manifest_path, validate_package=True)
    source_path = bundle.checkpoint_path
    source_type = bundle.weights_format
    return ModelReleaseInput(
        root=root,
        bundle=bundle,
        source_path=source_path,
        source_type=source_type,
    )
