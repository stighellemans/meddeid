from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import nullcontext
from pathlib import Path
import platform
from threading import Lock
from typing import Any, Callable, Mapping

from meddeid_core.age_policy import (
    AgeGranularityPolicy,
    load_age_granularity_policy,
)
from meddeid_core.language import LanguageProfile
from meddeid_core.normalize import normalize_metadata
from meddeid_core.taxonomy import BERT_ENTITY_LABELS

from .bundle import ModelBundle, load_model_bundle
from .language import resolve_language_profile
from .model_inspection import (
    model_file_inventory,
    package_versions,
    requested_revision_label,
)
from .model_source import resolve_model_source
from .pipeline import DeidentificationPipeline
from .runtime.base import InferenceRuntime


StatusCallback = Callable[[str], None]
INFERENCE_PROVENANCE_CONTRACT = "meddeid.inference-provenance.v1"


@dataclass(frozen=True)
class DeidentificationResult:
    text: str
    spans: list[dict[str, Any]]
    deid_text: str
    provenance: dict[str, Any]
    warnings: list[dict[str, str]] = field(default_factory=list)
    processing: dict[str, Any] = field(default_factory=dict)

    def to_contract(self) -> dict[str, Any]:
        return {
            "deid_text": self.deid_text,
            "spans": self.spans,
            "processing": self.processing,
            "warnings": self.warnings,
            "provenance": self.provenance,
        }


def choose_language_profile(
    profiles: tuple[LanguageProfile, ...],
    *,
    document_language: str | None,
    explicit_default: LanguageProfile | None,
) -> LanguageProfile:
    """Apply the public per-document profile-resolution contract.

    Trusted document metadata wins over a load-time default.  A single-profile
    bundle is its own safe default; a multi-profile bundle without either input
    fails rather than guessing.
    """

    if document_language is not None:
        matches = [
            profile
            for profile in profiles
            if profile.accepts_language(str(document_language))
        ]
        if len(matches) == 1:
            return matches[0]
        supported = ", ".join(profile.profile_id for profile in profiles)
        raise ValueError(
            f"metadata.lang={document_language!r} is not a supported unambiguous locale; "
            f"expected one of: {supported}"
        )
    if explicit_default is not None:
        return explicit_default
    if len(profiles) == 1:
        return profiles[0]
    supported = ", ".join(profile.profile_id for profile in profiles)
    raise ValueError(
        "metadata.lang is required for this multi-profile model, unless a "
        f"default is selected while loading it; supported profiles: {supported}"
    )


def redact(text: str, spans: list[dict[str, Any]]) -> str:
    output = text
    for span in sorted(
        spans, key=lambda item: (int(item["begin"]), int(item["end"])), reverse=True
    ):
        begin, end = int(span["begin"]), int(span["end"])
        replacement = str(span.get("replacement", f"[{span['label']}]"))
        output = output[:begin] + replacement + output[end:]
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
        language_profile: str | None = None,
        age_granularity_config: (
            AgeGranularityPolicy | str | Path | Mapping[str, Any] | None
        ) = None,
        min_recommended_date_shift_days: int = 366,
    ) -> None:
        if (
            isinstance(min_recommended_date_shift_days, bool)
            or not isinstance(min_recommended_date_shift_days, int)
            or min_recommended_date_shift_days <= 0
        ):
            raise ValueError(
                "min_recommended_date_shift_days must be a positive integer"
            )
        self.bundle = bundle
        self.runtime = runtime
        self.model_source = model_source
        self.model_source_is_local = Path(model_source).expanduser().exists()
        self.requested_revision = requested_revision
        self.resolved_revision = resolved_revision
        self.offline = offline
        self.age_granularity_policy = load_age_granularity_policy(
            age_granularity_config
        )
        self.min_recommended_date_shift_days = min_recommended_date_shift_days
        self._provenance_packages = self._package_versions()
        self._inference_lock = Lock()
        self.pipeline = DeidentificationPipeline(bundle)
        self.language_profiles: tuple[LanguageProfile, ...] = tuple(
            resolve_language_profile(item.profile_id)
            for item in bundle.postprocess.profiles
        )
        self.language_profile: LanguageProfile | None = None
        if language_profile is not None:
            requested = language_profile.strip().replace("_", "-")
            matches = [
                profile
                for profile in self.language_profiles
                if profile.accepts_language(requested)
            ]
            if len(matches) != 1:
                supported = ", ".join(
                    profile.profile_id for profile in self.language_profiles
                )
                raise ValueError(
                    f"language profile {requested!r} is not uniquely supported by this "
                    f"model; expected one of: {supported}"
                )
            self.language_profile = matches[0]
        elif len(self.language_profiles) == 1:
            self.language_profile = self.language_profiles[0]

    @staticmethod
    def _package_versions() -> dict[str, str | None]:
        return package_versions()

    def _requested_revision(self) -> str:
        return requested_revision_label(
            self.requested_revision,
            source_is_local=self.model_source_is_local,
        )

    def inference_provenance(self, language_profile: str) -> dict[str, Any]:
        return {
            "contract_version": INFERENCE_PROVENANCE_CONTRACT,
            "software": {
                "name": "meddeid",
                "version": self._provenance_packages.get("meddeid"),
            },
            "model": {
                "name": self.bundle.name,
                "version": self.bundle.model_version,
                "resolved_revision": self.resolved_revision,
                "bundle_sha256": self.bundle.contract_hash(),
            },
            "language_profile": {"profile_id": language_profile},
        }

    def _model_file_inventory(self) -> dict[str, Any]:
        return model_file_inventory(
            self.bundle,
            source_is_local=self.model_source_is_local,
            offline=self.offline,
        )

    def _profile_for_document(self, metadata: dict[str, Any]) -> LanguageProfile:
        return choose_language_profile(
            self.language_profiles,
            document_language=metadata.get("lang"),
            explicit_default=self.language_profile,
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
        triton_transport: str = "json",
        max_windows_per_batch: int | None = None,
        torch_precision: str = "fp32",
        torch_compile_mode: str = "off",
        torch_compile_dynamic: bool | None = True,
        on_status: StatusCallback | None = None,
        language_profile: str | None = None,
        age_granularity_config: (
            AgeGranularityPolicy | str | Path | Mapping[str, Any] | None
        ) = None,
        min_recommended_date_shift_days: int = 366,
    ) -> "Deidentifier":
        def emit(message: str) -> None:
            if on_status is not None:
                on_status(message)

        backend = backend.strip().lower()
        if backend not in {"torch", "triton"}:
            raise ValueError("backend must be 'torch' or 'triton'")
        policy = load_age_granularity_policy(age_granularity_config)
        if (
            isinstance(min_recommended_date_shift_days, bool)
            or not isinstance(min_recommended_date_shift_days, int)
            or min_recommended_date_shift_days <= 0
        ):
            raise ValueError(
                "min_recommended_date_shift_days must be a positive integer"
            )
        resolved = resolve_model_source(
            model,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            include_weights=backend == "torch",
            on_status=on_status,
        )
        root = resolved.root
        emit("validating model bundle")
        bundle = load_model_bundle(
            root / "bundle.json",
            validate_package=True,
            require_weights=backend == "torch",
        )
        if backend == "torch":
            emit("loading PyTorch model")
            from .runtime.torch import TorchRuntime

            runtime: InferenceRuntime = TorchRuntime(
                bundle,
                device=device,
                max_windows_per_batch=max_windows_per_batch or 32,
                precision=torch_precision,
                compile_mode=torch_compile_mode,
                compile_dynamic=torch_compile_dynamic,
            )
        else:
            emit("connecting to TensorRT through Triton")
            from .runtime.triton import TritonRuntime

            runtime = TritonRuntime(
                bundle,
                base_url=triton_url or "",
                timeout_seconds=triton_timeout_seconds,
                max_windows_per_batch=max_windows_per_batch or 16,
                transport=triton_transport,
            )
        emit("loading tokenizer and language profile")
        engine = cls(
            bundle,
            runtime,
            model_source=str(model),
            requested_revision=revision,
            resolved_revision=resolved.resolved_revision,
            offline=local_files_only,
            language_profile=language_profile,
            age_granularity_config=policy,
            min_recommended_date_shift_days=min_recommended_date_shift_days,
        )
        emit("ready")
        return engine

    def __call__(
        self, text: str, *, metadata: dict[str, Any] | None = None
    ) -> DeidentificationResult:
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
        normalized: list[tuple[str, dict[str, Any], LanguageProfile]] = []
        for text, metadata in documents:
            if not isinstance(text, str):
                raise TypeError("text must be a string")
            if metadata is not None and not isinstance(metadata, dict):
                raise TypeError("metadata must be an object")
            document_metadata = normalize_metadata(metadata or {})
            deployment_only = sorted(
                set(document_metadata)
                & {
                    "age_granularity_config",
                    "age_granularity_policy",
                    "min_recommended_date_shift_days",
                }
            )
            if deployment_only:
                raise ValueError(
                    f"deployment-only setting(s) {deployment_only} cannot be selected per request"
                )
            profile = self._profile_for_document(document_metadata)
            profile.validate_language(document_metadata.get("lang"))
            document_metadata = self._with_patient_birth_date_variants(
                document_metadata, profile
            )
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
            normalized.append((text, document_metadata, profile))

        inference_guard = (
            nullcontext()
            if getattr(self.runtime, "supports_concurrent_requests", False)
            else self._inference_lock
        )
        with inference_guard:
            states = []
            all_windows = []
            for index, (text, document_metadata, _) in enumerate(normalized):
                prepared = self.pipeline.prepare_document(
                    record_id=f"document-{index}", text=text, metadata=document_metadata
                )
                state = self.pipeline.prepare_state(index, prepared)
                states.append(state)
                all_windows.extend(state.windows)

            all_predictions = self.runtime.infer_windows(all_windows)
            results: list[DeidentificationResult] = []
            prediction_offset = 0
            for state, (text, document_metadata, profile) in zip(
                states, normalized, strict=True
            ):
                next_offset = prediction_offset + len(state.windows)
                self.pipeline.apply_predictions(
                    state, all_predictions[prediction_offset:next_offset]
                )
                prediction_offset = next_offset
                raw = self.pipeline.decode_document(state)
                spans = profile.post_process_spans(raw.spans, text, document_metadata)
                spans, warnings, processing = self._apply_replacements(
                    text, spans, document_metadata, profile
                )
                results.append(
                    DeidentificationResult(
                        text=text,
                        spans=spans,
                        deid_text=redact(text, spans),
                        warnings=warnings,
                        processing=processing,
                        provenance=self.inference_provenance(profile.profile_id),
                    )
                )

        if prediction_offset != len(all_predictions):
            raise RuntimeError("runtime returned predictions for unexpected windows")
        return results

    def _with_patient_birth_date_variants(
        self,
        metadata: dict[str, Any],
        profile: LanguageProfile,
    ) -> dict[str, Any]:
        patient = metadata.get("patient")
        if not isinstance(patient, dict) or patient.get("birth_date") is None:
            return metadata
        birth_date = patient.get("birth_date")
        if not isinstance(birth_date, str) or not birth_date.strip():
            raise ValueError("metadata.patient.birth_date must be a non-empty string")
        variants = profile.birth_date_variants(birth_date)
        known_values = list(metadata.get("known_values") or [])
        existing = {
            (entry.get("value"), entry.get("label"))
            for entry in known_values
            if isinstance(entry, dict)
        }
        for variant in variants:
            key = (variant, "Age_Birthdate")
            if key not in existing:
                known_values.append({"value": variant, "label": "Age_Birthdate"})
                existing.add(key)
        return {**metadata, "known_values": known_values}

    def _apply_replacements(
        self,
        text: str,
        spans: list[dict[str, Any]],
        metadata: dict[str, Any],
        profile: LanguageProfile,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
        raw_shift = metadata.get("date_shift_days")
        if isinstance(raw_shift, bool) or (
            raw_shift is not None and not isinstance(raw_shift, int)
        ):
            raise ValueError("metadata.date_shift_days must be an integer or null")
        date_shift_days = int(raw_shift) if raw_shift is not None else None
        minimum_shift = getattr(self, "min_recommended_date_shift_days", 366)
        age_policy = getattr(self, "age_granularity_policy", None)
        if age_policy is None:  # lightweight custom/test engines built without __init__
            age_policy = load_age_granularity_policy()
        counters = {
            "detected_spans": 0,
            "shifted_spans": 0,
            "age_generalized_spans": 0,
            "year_fallback_spans": 0,
            "placeholder_spans": 0,
        }
        warning_messages: dict[str, str] = {}
        if date_shift_days == 0:
            warning_messages["zero_date_shift_placeholder"] = (
                "date_shift_days=0 would reproduce original dates; placeholders were used"
            )
        elif date_shift_days is not None and abs(date_shift_days) < minimum_shift:
            warning_messages["date_shift_below_recommended_minimum"] = (
                f"absolute date shift is below the configured recommended minimum of "
                f"{minimum_shift} days"
            )

        output: list[dict[str, Any]] = []
        for source_span in spans:
            span = dict(source_span)
            label = str(span["label"])
            span["replacement"] = f"[{label}]"
            if label not in {"Date", "Age_Birthdate"}:
                output.append(span)
                continue
            counters["detected_spans"] += 1
            if date_shift_days is None or date_shift_days == 0:
                counters["placeholder_spans"] += 1
                output.append(span)
                continue
            if profile.date_replacement_provider is None:
                counters["placeholder_spans"] += 1
                warning_messages["date_replacement_unsupported"] = (
                    f"language profile {profile.profile_id!r} does not provide date replacement"
                )
                output.append(span)
                continue

            begin, end = int(span["begin"]), int(span["end"])
            try:
                candidate = profile.replace_date(
                    str(span.get("text", text[begin:end])),
                    label=label,
                    date_shift_days=date_shift_days,
                    context_before=text[max(0, begin - 64) : begin],
                    context_after=text[end : min(len(text), end + 64)],
                    document_creation_date=metadata.get("document_creation_date"),
                    age_granularity_policy=age_policy,
                )
            except OverflowError as exc:
                raise ValueError(
                    "metadata.date_shift_days produces a date outside the supported range"
                ) from exc
            if candidate is None:
                counters["placeholder_spans"] += 1
                warning_messages["date_parse_fallback"] = (
                    "one or more date or age spans could not be parsed; placeholders were used"
                )
                output.append(span)
                continue
            span["replacement"] = f"[{candidate.body}]"
            if candidate.kind == "shifted_date":
                counters["shifted_spans"] += 1
            elif candidate.kind == "age_generalized":
                counters["age_generalized_spans"] += 1
            else:
                counters["year_fallback_spans"] += 1
                warning_messages["birthdate_year_fallback"] = (
                    "a birthdate could not be converted to age without a usable document date; "
                    "a shifted year was used"
                )
            output.append(span)

        warnings = [
            {"code": code, "message": message}
            for code, message in warning_messages.items()
        ]
        processing = {
            "date_replacement": {
                "mode": "shift" if date_shift_days not in {None, 0} else "placeholder",
                "requested_shift_days": date_shift_days,
                "minimum_recommended_abs_shift_days": (minimum_shift),
                **counters,
            },
            "age_granularity_policy": age_policy.identity.to_dict(),
        }
        return output, warnings, processing

    def model_info(self) -> dict[str, Any]:
        packages = self._package_versions()
        return {
            "model": {
                "source": self.model_source,
                "name": self.bundle.name,
                "version": self.bundle.model_version,
                "requested_revision": self._requested_revision(),
                "resolved_revision": self.resolved_revision,
                "bundle_sha256": self.bundle.contract_hash(),
                "weights_format": self.bundle.weights_format,
            },
            "model_files": self._model_file_inventory(),
            "contracts": {
                "language_profiles": [
                    {"profile_id": profile.profile_id}
                    for profile in self.language_profiles
                ],
                "default_language_profile": (
                    self.language_profile.profile_id
                    if self.language_profile is not None
                    else None
                ),
                "max_length": self.bundle.inference.max_length,
                "overlap": self.bundle.inference.overlap,
                "entity_labels": len(self.bundle.entity_labels),
                "age_granularity_policy": (
                    self.age_granularity_policy.identity.to_dict()
                ),
                "minimum_recommended_abs_date_shift_days": (
                    self.min_recommended_date_shift_days
                ),
            },
            "runtime": {"checked": True, **self.runtime.healthcheck()},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
            },
        }

    def health_info(self) -> dict[str, Any]:
        runtime = self.runtime.healthcheck()
        ready = bool(runtime.get("ready", False))
        return {
            "status": "ok" if ready else "unavailable",
            "ready": ready,
            "model": {
                "name": self.bundle.name,
                "version": self.bundle.model_version,
            },
            "contracts": {
                "language_profiles": [
                    {"profile_id": profile.profile_id}
                    for profile in self.language_profiles
                ],
                "default_language_profile": (
                    self.language_profile.profile_id
                    if self.language_profile is not None
                    else None
                ),
            },
        }

    def close(self) -> None:
        self.runtime.close()
