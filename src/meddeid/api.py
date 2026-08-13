from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
from threading import Lock
from typing import Any, Callable

from meddeid_core.language import LanguageProfile
from meddeid_core.normalize import normalize_metadata
from meddeid_core.taxonomy import BERT_ENTITY_LABELS

from .bundle import ModelBundle, load_model_bundle
from .language import resolve_language_profile
from .pipeline import DeidentificationPipeline
from .runtime import InferenceRuntime, TorchRuntime, TritonRuntime


StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class DeidentificationResult:
    text: str
    spans: list[dict[str, Any]]
    deid_text: str


def redact(text: str, spans: list[dict[str, Any]]) -> str:
    output = text
    for span in sorted(spans, key=lambda item: (int(item["begin"]), int(item["end"])), reverse=True):
        begin, end = int(span["begin"]), int(span["end"])
        output = output[:begin] + f"[{span['label']}]" + output[end:]
    return output


class Deidentifier:
    def __init__(
        self,
        bundle: ModelBundle,
        runtime: InferenceRuntime,
        *,
        model_source: str,
        requested_revision: str | None,
        resolved_revision: str | None,
        offline: bool,
    ) -> None:
        self.bundle = bundle
        self.runtime = runtime
        self.model_source = model_source
        self.requested_revision = requested_revision
        self.resolved_revision = resolved_revision
        self.offline = offline
        self._inference_lock = Lock()
        self.pipeline = DeidentificationPipeline(bundle)
        self.language_profile: LanguageProfile = resolve_language_profile(
            bundle.postprocess.profile_id,
            version=bundle.postprocess.profile_version,
        )

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        device: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
        backend: str = "torch",
        triton_url: str | None = None,
        triton_timeout_seconds: float = 30.0,
        max_windows_per_batch: int | None = None,
        on_status: StatusCallback | None = None,
    ) -> "Deidentifier":
        def emit(message: str) -> None:
            if on_status is not None:
                on_status(message)

        backend = backend.strip().lower()
        if backend not in {"torch", "triton"}:
            raise ValueError("backend must be 'torch' or 'triton'")
        emit("resolving model")
        root = Path(model).expanduser()
        if not root.exists():
            emit("downloading model bundle")
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("the meddeid installation is missing huggingface-hub") from exc
            try:
                root = Path(snapshot_download(
                    repo_id=str(model),
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        "bundle.json",
                        "config.json",
                        "model.safetensors",
                        "model.pt",
                        "tokenizer*",
                        "special_tokens_map.json",
                        "vocab.json",
                        "merges.txt",
                        "README.md",
                        "LICENSE*",
                        "NOTICE*",
                    ],
                ))
            except Exception as exc:
                if local_files_only:
                    recovery = (
                        f"model {model!s} is not available in the local cache; "
                        f"download it first with `hf download {model!s} --local-dir ./meddeid-model`"
                    )
                else:
                    recovery = (
                        f"model {model!s} could not be downloaded; verify the repository name "
                        "and visibility, or authenticate with `hf auth login` if it is private"
                    )
                raise RuntimeError(recovery) from exc
        else:
            emit("using local model bundle")
        emit("validating model bundle")
        bundle = load_model_bundle(root / "bundle.json", validate_package=True)
        resolved_revision = (
            root.name if root.parent.name == "snapshots" and len(root.name) >= 7 else revision
        )
        if backend == "torch":
            emit("loading PyTorch model")
            runtime: InferenceRuntime = TorchRuntime(
                bundle,
                device=device,
                max_windows_per_batch=max_windows_per_batch or 32,
            )
        else:
            emit("connecting to TensorRT through Triton")
            runtime = TritonRuntime(
                bundle,
                base_url=triton_url or "",
                timeout_seconds=triton_timeout_seconds,
                max_windows_per_batch=max_windows_per_batch or 16,
            )
        emit("loading tokenizer and language profile")
        engine = cls(
            bundle,
            runtime,
            model_source=str(model),
            requested_revision=revision,
            resolved_revision=resolved_revision,
            offline=local_files_only,
        )
        emit("ready")
        return engine

    def __call__(self, text: str, *, metadata: dict[str, Any] | None = None) -> DeidentificationResult:
        return self.deidentify_many([(text, metadata)])[0]

    def deidentify_many(
        self,
        documents: list[tuple[str, dict[str, Any] | None]],
    ) -> list[DeidentificationResult]:
        """De-identify documents in one bounded runtime call.

        Tokenization and post-processing remain document-specific, including
        metadata. Model windows are flattened across documents so PyTorch or
        Triton can batch them efficiently without allowing spans to cross a
        document boundary.
        """

        if not documents:
            return []
        normalized: list[tuple[str, dict[str, Any]]] = []
        for text, metadata in documents:
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            if metadata is not None and not isinstance(metadata, dict):
                raise TypeError("metadata must be an object")
            document_metadata = normalize_metadata(metadata or {})
            known_values = document_metadata.get("known_values")
            if known_values is not None:
                if not isinstance(known_values, list):
                    raise ValueError("metadata.known_values must be a list")
                for index, entry in enumerate(known_values):
                    if not isinstance(entry, dict):
                        raise ValueError(
                            f"metadata.known_values[{index}] must be an object"
                        )
                    value = entry.get("value")
                    label = entry.get("label")
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError(
                            f"metadata.known_values[{index}].value must be a non-empty string"
                        )
                    if label not in BERT_ENTITY_LABELS:
                        raise ValueError(
                            f"metadata.known_values[{index}].label must be a canonical label"
                        )
            self.language_profile.validate_language(document_metadata.get("lang"))
            normalized.append((text, document_metadata))

        with self._inference_lock:
            states = []
            all_windows = []
            for index, (text, document_metadata) in enumerate(normalized):
                prepared = self.pipeline.prepare_document(
                    record_id=f"document-{index}", text=text, metadata=document_metadata
                )
                state = self.pipeline.prepare_state(index, prepared)
                states.append(state)
                all_windows.extend(state.windows)

            all_predictions = self.runtime.infer_windows(all_windows)
            results: list[DeidentificationResult] = []
            prediction_offset = 0
            for state, (text, document_metadata) in zip(states, normalized, strict=True):
                next_offset = prediction_offset + len(state.windows)
                self.pipeline.apply_predictions(
                    state, all_predictions[prediction_offset:next_offset]
                )
                prediction_offset = next_offset
                raw = self.pipeline.decode_document(state)
                spans = self.language_profile.post_process_spans(
                    raw.spans, text, document_metadata
                )
                results.append(
                    DeidentificationResult(
                        text=text,
                        spans=spans,
                        deid_text=redact(text, spans),
                    )
                )

        if prediction_offset != len(all_predictions):
            raise RuntimeError("runtime returned predictions for unexpected windows")
        return results

    def model_info(self) -> dict[str, Any]:
        packages: dict[str, str | None] = {}
        for package in (
            "meddeid",
            "meddeid-core",
            "meddeid-language-nl",
            "torch",
            "transformers",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        return {
            "model": {
                "source": self.model_source,
                "name": self.bundle.name,
                "version": self.bundle.model_version,
                "requested_revision": self.requested_revision,
                "resolved_revision": self.resolved_revision,
                "bundle_sha256": self.bundle.contract_hash(),
                "weights_format": self.bundle.weights_format,
                "local_files_only": self.offline,
                "offline_ready": True,
            },
            "contracts": {
                "language_profile": self.bundle.postprocess.profile_id,
                "language_profile_version": self.bundle.postprocess.profile_version,
                "max_length": self.bundle.inference.max_length,
                "overlap": self.bundle.inference.overlap,
                "entity_labels": len(self.bundle.entity_labels),
            },
            "runtime": self.runtime.healthcheck(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
            },
        }

    def close(self) -> None:
        self.runtime.close()
