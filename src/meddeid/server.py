import argparse
import asyncio
import os
import re
import secrets
import sys
from collections.abc import Collection
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from . import __version__
from .api import Deidentifier

DEFAULT_MAX_INPUT_CHARS = 20_000
DEFAULT_MAX_BATCH_DOCUMENTS = 32
DEFAULT_MAX_BATCH_CHARS = 200_000
DEFAULT_MAX_REQUEST_BYTES = 2_000_000
DEFAULT_MAX_CONCURRENT_REQUESTS = 1
DEFAULT_QUEUE_TIMEOUT_SECONDS = 30.0


SERVER_ENVIRONMENT_KEYS = frozenset(
    {
        "MEDDEID_ACCESS_LOG",
        "MEDDEID_AGE_GRANULARITY_CONFIG",
        "MEDDEID_ALLOWED_LANGUAGE_PROFILES",
        "MEDDEID_ALLOWED_MODELS",
        "MEDDEID_API_KEY",
        "MEDDEID_BACKEND",
        "MEDDEID_BIND_ADDRESS",
        "MEDDEID_DEVICE",
        "MEDDEID_DOCS_ENABLED",
        "MEDDEID_FORWARDED_ALLOW_IPS",
        "MEDDEID_LANGUAGE_PROFILE",
        "MEDDEID_MAX_BATCH_CHARS",
        "MEDDEID_MAX_BATCH_DOCUMENTS",
        "MEDDEID_MAX_CONCURRENT_REQUESTS",
        "MEDDEID_MAX_INPUT_CHARS",
        "MEDDEID_MAX_REQUEST_BYTES",
        "MEDDEID_MICROBATCH_ENABLED",
        "MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS",
        "MEDDEID_MODEL",
        "MEDDEID_OFFLINE",
        "MEDDEID_PORT",
        "MEDDEID_PROXY_HEADERS",
        "MEDDEID_QUEUE_TIMEOUT_SECONDS",
        "MEDDEID_REVISION",
        "MEDDEID_REQUIRE_API_KEY",
        "MEDDEID_SEQUENCE_LENGTH_BUCKETS",
        "MEDDEID_SERVING_PROFILE",
        "MEDDEID_MICROBATCH_MAX_TOKENS",
        "MEDDEID_MICROBATCH_MAX_WAIT_MS",
        "MEDDEID_MICROBATCH_MAX_WINDOWS",
        "MEDDEID_MICROBATCH_QUEUE_MAX_REQUESTS",
        "MEDDEID_MICROBATCH_QUEUE_MAX_WINDOWS",
        "MEDDEID_TRITON_TIMEOUT",
        "MEDDEID_TRITON_TRANSPORT",
        "MEDDEID_TRITON_URL",
        "MEDDEID_TORCH_COMPILE_MODE",
        "MEDDEID_TORCH_COMPILE_DYNAMIC",
        "MEDDEID_TORCH_PRECISION",
        "MEDDEID_UI_ENABLED",
        "MEDDEID_WINDOW_BATCH_SIZE",
        "MEDDEID_WORKERS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


LOCAL_UI = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MedDeID local de-identification</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #f4f7f6; color: #17231f; }
    main { max-width: 860px; margin: 0 auto; padding: 2rem 1rem 4rem; }
    h1 { margin-bottom: .35rem; }
    .notice { background: #fff5d7; border-left: 4px solid #a96b00; padding: .85rem 1rem; }
    .card { background: white; border: 1px solid #d7e0dd; border-radius: 12px; padding: 1rem; margin-top: 1rem; box-shadow: 0 2px 8px #10251c0d; }
    label { display: block; font-weight: 650; margin: .8rem 0 .35rem; }
    textarea, input, select { box-sizing: border-box; width: 100%; padding: .75rem; border: 1px solid #93a39d; border-radius: 7px; font: inherit; }
    textarea { min-height: 190px; resize: vertical; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; }
    .actions { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1rem; }
    button { border: 0; border-radius: 7px; padding: .7rem 1rem; font: inherit; font-weight: 650; cursor: pointer; }
    button.primary { background: #075e54; color: white; }
    button.secondary { background: #e4ebe8; color: #17231f; }
    button:disabled { opacity: .6; cursor: wait; }
    #status { min-height: 1.5rem; margin-top: .8rem; }
    #result { white-space: pre-wrap; overflow-wrap: anywhere; background: #f4f7f6; padding: 1rem; border-radius: 7px; min-height: 3rem; }
    .error { color: #a01818; }
    .muted { color: #4e625b; font-size: .92rem; }
    @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>MedDeID</h1>
  <p>De-identify one clinical note using this server's model.</p>
  <p class="notice"><strong>Important:</strong> Model output can still contain identifying information. Validate it and use human review where an error could cause harm.</p>
  <section class="card" aria-labelledby="input-title">
    <h2 id="input-title">Note</h2>
    <label for="api-key">API key</label>
    <input id="api-key" type="password" autocomplete="off" spellcheck="false" aria-describedby="key-help">
    <div id="key-help" class="muted">Stored only in this browser tab and sent to this MedDeID server.</div>
    <div class="grid">
      <div>
        <label for="current-model">Active model</label>
        <input id="current-model" value="Loading…" readonly aria-describedby="model-help">
        <div id="model-help" class="muted">Fixed for this server instance; select another with the local launcher.</div>
      </div>
      <div id="language-field">
        <label for="language">Language profile</label>
        <select id="language" aria-describedby="language-help"></select>
        <div id="language-help" class="muted">Selectable when the active model supports multiple profiles.</div>
      </div>
    </div>
    <div class="grid">
      <div><label for="given-name">Known patient given name (optional)</label><input id="given-name" autocomplete="off"></div>
      <div><label for="family-name">Known patient family name (optional)</label><input id="family-name" autocomplete="off"></div>
    </div>
    <label for="note">Clinical text</label>
    <textarea id="note" required spellcheck="false" placeholder="Paste a note here"></textarea>
    <div class="actions">
      <button id="run" class="primary" type="button">De-identify</button>
      <button id="clear" class="secondary" type="button">Clear</button>
    </div>
    <div id="status" role="status" aria-live="polite"></div>
  </section>
  <section class="card" aria-labelledby="result-title">
    <h2 id="result-title">De-identified text</h2>
    <div id="result" aria-live="polite">No result yet.</div>
    <div class="actions"><button id="copy" class="secondary" type="button" disabled>Copy result</button></div>
  </section>
</main>
<script>
  const byId = (id) => document.getElementById(id);
  const run = byId('run'), copy = byId('copy'), status = byId('status'), result = byId('result');
  const languageField = byId('language-field'), language = byId('language'), currentModel = byId('current-model');
  const launchParameters = new URLSearchParams(window.location.hash.slice(1));
  const launchApiKey = launchParameters.get('api_key');
  if (launchApiKey !== null) {
    byId('api-key').value = launchApiKey;
    history.replaceState(null, '', window.location.pathname + window.location.search);
  }
  let output = '';
  async function configureLanguages() {
    try {
      const response = await fetch('/health');
      const body = await response.json();
      if (!response.ok) return;
      const model = body.model || {};
      currentModel.value = [model.name, model.version].filter(Boolean).join(' · ') || 'Unknown';
      const contracts = body.contracts || {};
      const profiles = contracts.language_profiles || [];
      const profileLabels = {
        'nl-BE': 'Dutch — Belgium',
        'nl-NL': 'Dutch — Netherlands',
        'en-GB': 'English — United Kingdom',
        'en-US': 'English — United States',
      };
      if (!profiles.length) { languageField.hidden = true; return; }
      const defaultProfile = contracts.default_language_profile || '';
      language.replaceChildren();
      if (!defaultProfile) {
        const prompt = document.createElement('option');
        prompt.value = ''; prompt.textContent = 'Choose a language and region';
        language.appendChild(prompt);
      }
      for (const profile of profiles) {
        const option = document.createElement('option');
        option.value = profile.profile_id;
        option.textContent = profileLabels[profile.profile_id] || profile.profile_id;
        option.selected = profile.profile_id === defaultProfile;
        language.appendChild(option);
      }
      language.disabled = profiles.length === 1;
      languageField.hidden = false;
    } catch (_) {
      // The inference request will return an actionable error if profile input is required.
    }
  }
  configureLanguages();
  run.addEventListener('click', async () => {
    const text = byId('note').value;
    const key = byId('api-key').value;
    if (!text.trim()) { status.textContent = 'Enter a note first.'; status.className = 'error'; return; }
    const patient = {};
    if (byId('given-name').value.trim()) patient.given_name = byId('given-name').value.trim();
    if (byId('family-name').value.trim()) patient.family_name = byId('family-name').value.trim();
    if (!languageField.hidden && !language.value) {
      status.textContent = 'Choose the document language and region first.';
      status.className = 'error';
      return;
    }
    const metadata = {};
    if (Object.keys(patient).length) metadata.patient = patient;
    if (!languageField.hidden) metadata.lang = language.value;
    run.disabled = true; copy.disabled = true; status.className = ''; status.textContent = 'Processing locally…';
    try {
      const headers = {'Content-Type': 'application/json'};
      if (key) headers.Authorization = `Bearer ${key}`;
      const response = await fetch('/deidentify', {method: 'POST', headers, body: JSON.stringify({text, metadata})});
      const body = await response.json();
      if (!response.ok) {
        const detail = body.detail || {};
        throw new Error(detail.message || detail.code || `Request failed (HTTP ${response.status})`);
      }
      output = body.deid_text;
      result.textContent = output;
      copy.disabled = false;
      const usedModel = body.provenance && body.provenance.model && body.provenance.model.name;
      const usedProfile = body.provenance && body.provenance.language_profile && body.provenance.language_profile.profile_id;
      const selection = [usedModel, usedProfile].filter(Boolean).join(' · ');
      status.textContent = `Done${selection ? ` using ${selection}` : ''}. ${body.spans.length} identifying span${body.spans.length === 1 ? '' : 's'} detected.`;
    } catch (error) {
      output = ''; result.textContent = 'No result.'; status.className = 'error'; status.textContent = error.message;
    } finally { run.disabled = false; }
  });
  byId('clear').addEventListener('click', () => {
    byId('note').value = ''; byId('given-name').value = ''; byId('family-name').value = '';
    output = ''; result.textContent = 'No result yet.'; status.textContent = ''; status.className = ''; copy.disabled = true;
  });
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(output); status.textContent = 'Result copied.'; }
    catch (_) { status.className = 'error'; status.textContent = 'Copy was blocked by the browser. Select the result manually.'; }
  });
</script>
</body>
</html>"""


def _enabled(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _allowlist(
    values: Collection[str] | None,
    environment_value: str | None,
    *,
    setting: str,
) -> frozenset[str] | None:
    if values is None:
        if environment_value is None or not environment_value.strip():
            return None
        candidates = environment_value.split(",")
    else:
        candidates = list(values)
    normalized = frozenset(
        str(value).strip() for value in candidates if str(value).strip()
    )
    if not normalized:
        raise ValueError(f"{setting} must contain at least one non-empty value")
    return normalized


def _load_server_environment(path: Path) -> None:
    """Load a small, non-expanding KEY=VALUE server configuration file.

    Existing process variables take precedence, which lets an orchestrator
    inject secrets without writing them into the reproducible configuration.
    """

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read server environment file {path}: {exc}") from exc

    seen: set[str] = set()
    for line_number, source_line in enumerate(lines, start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _ENVIRONMENT_KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid variable name {key!r}")
        if key not in SERVER_ENVIRONMENT_KEYS:
            raise ValueError(f"{path}:{line_number}: unknown server setting {key!r}")
        if key in seen:
            raise ValueError(f"{path}:{line_number}: duplicate server setting {key!r}")
        seen.add(key)

        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ValueError(
                    f"{path}:{line_number}: unterminated quoted value for {key}"
                )
            value = value[1:-1]
        os.environ.setdefault(key, value)


def create_app(
    model: str | None = None,
    *,
    backend: str | None = None,
    device: str | None = None,
    triton_url: str | None = None,
    triton_transport: str | None = None,
    torch_precision: str | None = None,
    torch_compile_mode: str | None = None,
    torch_compile_dynamic: bool | None = None,
    serving_profile: str | None = None,
    language_profile: str | None = None,
    age_granularity_config: str | Path | None = None,
    min_recommended_date_shift_days: int | None = None,
    max_input_chars: int | None = None,
    max_batch_documents: int | None = None,
    max_batch_chars: int | None = None,
    max_request_bytes: int | None = None,
    max_concurrent_requests: int | None = None,
    queue_timeout_seconds: float | None = None,
    api_key: str | None = None,
    require_api_key: bool | None = None,
    allowed_models: Collection[str] | None = None,
    allowed_language_profiles: Collection[str] | None = None,
    docs_enabled: bool | None = None,
    ui_enabled: bool | None = None,
    engine: Any | None = None,
):
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, Security
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.security import (
            APIKeyHeader,
            HTTPAuthorizationCredentials,
            HTTPBearer,
        )
        from pydantic import (
            BaseModel,
            ConfigDict,
            Field,
            field_validator,
            model_validator,
        )
    except ImportError as exc:
        raise RuntimeError("install meddeid[server]") from exc

    selected_serving_profile = (
        (
            serving_profile
            if serving_profile is not None
            else os.environ.get("MEDDEID_SERVING_PROFILE", "latency")
        )
        .strip()
        .lower()
    )
    if selected_serving_profile not in {"latency", "throughput"}:
        raise ValueError("MEDDEID_SERVING_PROFILE must be 'latency' or 'throughput'")
    selected_backend = (
        (backend or os.environ.get("MEDDEID_BACKEND", "torch")).strip().lower()
    )
    if selected_backend not in {"torch", "triton"}:
        raise ValueError("MEDDEID_BACKEND must be 'torch' or 'triton'")
    raw_microbatching = os.environ.get("MEDDEID_MICROBATCH_ENABLED", "auto")
    normalized_microbatching = raw_microbatching.strip().lower()
    if normalized_microbatching == "auto":
        microbatching_policy = "auto"
    elif normalized_microbatching in {"1", "true", "yes", "on"}:
        microbatching_policy = "on"
    elif normalized_microbatching in {"0", "false", "no", "off"}:
        microbatching_policy = "off"
    else:
        raise ValueError("MEDDEID_MICROBATCH_ENABLED must be auto, true, or false")
    if selected_serving_profile == "latency" and microbatching_policy == "on":
        raise ValueError(
            "MEDDEID_MICROBATCH_ENABLED=true requires the throughput serving profile"
        )
    if torch_compile_dynamic is None:
        raw_compile_dynamic = (
            os.environ.get("MEDDEID_TORCH_COMPILE_DYNAMIC", "true").strip().lower()
        )
        if raw_compile_dynamic in {"1", "true", "yes"}:
            selected_compile_dynamic: bool | None = True
        elif raw_compile_dynamic in {"0", "false", "no"}:
            selected_compile_dynamic = False
        elif raw_compile_dynamic in {"auto", "none", ""}:
            selected_compile_dynamic = None
        else:
            raise ValueError(
                "MEDDEID_TORCH_COMPILE_DYNAMIC must be true, false, or auto"
            )
    else:
        selected_compile_dynamic = torch_compile_dynamic

    configured_limit = (
        max_input_chars
        if max_input_chars is not None
        else os.environ.get("MEDDEID_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS)
    )
    limit = int(configured_limit)
    if limit < 1:
        raise ValueError("MEDDEID_MAX_INPUT_CHARS must be positive")

    document_limit = int(
        max_batch_documents
        if max_batch_documents is not None
        else os.environ.get("MEDDEID_MAX_BATCH_DOCUMENTS", DEFAULT_MAX_BATCH_DOCUMENTS)
    )
    batch_char_limit = int(
        max_batch_chars
        if max_batch_chars is not None
        else os.environ.get("MEDDEID_MAX_BATCH_CHARS", DEFAULT_MAX_BATCH_CHARS)
    )
    if document_limit < 1 or batch_char_limit < 1:
        raise ValueError("batch limits must be positive")

    request_byte_limit = int(
        max_request_bytes
        if max_request_bytes is not None
        else os.environ.get("MEDDEID_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES)
    )
    configured_concurrency: int | str = (
        max_concurrent_requests
        if max_concurrent_requests is not None
        else os.environ.get("MEDDEID_MAX_CONCURRENT_REQUESTS", "auto")
    )
    automatic_concurrency = str(configured_concurrency).strip().lower() == "auto"
    concurrent_limit = (
        DEFAULT_MAX_CONCURRENT_REQUESTS
        if automatic_concurrency
        else int(configured_concurrency)
    )
    queue_timeout = float(
        queue_timeout_seconds
        if queue_timeout_seconds is not None
        else os.environ.get(
            "MEDDEID_QUEUE_TIMEOUT_SECONDS", DEFAULT_QUEUE_TIMEOUT_SECONDS
        )
    )
    if request_byte_limit < 1:
        raise ValueError("MEDDEID_MAX_REQUEST_BYTES must be positive")
    if concurrent_limit < 1:
        raise ValueError("MEDDEID_MAX_CONCURRENT_REQUESTS must be positive")
    if queue_timeout <= 0:
        raise ValueError("MEDDEID_QUEUE_TIMEOUT_SECONDS must be positive")

    configured_api_key = (
        api_key if api_key is not None else os.environ.get("MEDDEID_API_KEY", "")
    ).strip()
    key_required = (
        require_api_key
        if require_api_key is not None
        else _enabled(os.environ.get("MEDDEID_REQUIRE_API_KEY"))
    )
    if key_required and not configured_api_key:
        raise ValueError(
            "MEDDEID_REQUIRE_API_KEY is enabled but MEDDEID_API_KEY is empty"
        )
    expose_docs = (
        docs_enabled
        if docs_enabled is not None
        else _enabled(os.environ.get("MEDDEID_DOCS_ENABLED"), default=True)
    )
    expose_ui = (
        ui_enabled
        if ui_enabled is not None
        else _enabled(os.environ.get("MEDDEID_UI_ENABLED"), default=expose_docs)
    )
    configured_allowed_models = _allowlist(
        allowed_models,
        os.environ.get("MEDDEID_ALLOWED_MODELS"),
        setting="MEDDEID_ALLOWED_MODELS",
    )
    raw_allowed_profiles = _allowlist(
        allowed_language_profiles,
        os.environ.get("MEDDEID_ALLOWED_LANGUAGE_PROFILES"),
        setting="MEDDEID_ALLOWED_LANGUAGE_PROFILES",
    )
    configured_allowed_profiles = (
        frozenset(value.replace("_", "-") for value in raw_allowed_profiles)
        if raw_allowed_profiles is not None
        else None
    )

    class StrictRequestModel(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class PersonMetadata(StrictRequestModel):
        given_name: str | None = None
        family_name: str | None = None

    class PatientMetadata(PersonMetadata):
        birth_date: str | None = Field(
            default=None,
            description=(
                "Trusted full patient birth date. The selected language profile locates "
                "equivalent full-year representations as Age_Birthdate spans."
            ),
        )

    class KnownValueMetadata(StrictRequestModel):
        value: str = Field(min_length=1)
        label: str = Field(min_length=1)

        @field_validator("label")
        @classmethod
        def canonical_label(cls, value: str) -> str:
            from meddeid_core.taxonomy import BERT_ENTITY_LABELS

            if value not in BERT_ENTITY_LABELS:
                raise ValueError("label must be one of the 14 canonical MedDeID labels")
            return value

    class RecordMetadata(StrictRequestModel):
        lang: str | None = None
        patient: PatientMetadata | None = None
        caregivers: list[PersonMetadata] | None = None
        document_creation_date: str | None = Field(
            default=None,
            description="Reference date used to convert birthdates to generalized ages.",
        )
        date_shift_days: int | None = Field(
            default=None,
            strict=True,
            description=(
                "Explicit date shift in days. Omit for placeholders; zero also uses "
                "placeholders and produces a warning."
            ),
        )
        known_values: list[KnownValueMetadata] | None = None

        @model_validator(mode="before")
        @classmethod
        def reject_retired_identity_keys(cls, value):
            if isinstance(value, dict):
                retired = [
                    key for key in ("patient_name", "caregiver_names") if key in value
                ]
                if retired:
                    raise ValueError(
                        f"retired metadata key(s) {retired}; use 'patient' and 'caregivers'"
                    )
                deployment_only = [
                    key
                    for key in (
                        "age_granularity_config",
                        "age_granularity_policy",
                        "min_recommended_date_shift_days",
                    )
                    if key in value
                ]
                if deployment_only:
                    raise ValueError(
                        f"deployment-only setting(s) {deployment_only} cannot be selected per request"
                    )
            return value

    class DeidentifyRequest(StrictRequestModel):
        text: str = Field(min_length=1, max_length=limit)
        metadata: RecordMetadata = Field(default_factory=RecordMetadata)

    class InferenceSpan(BaseModel):
        model_config = ConfigDict(extra="forbid")

        begin: int = Field(description="Inclusive character offset in the input text.")
        end: int = Field(description="Exclusive character offset in the input text.")
        label: str = Field(description="Detected MedDeID category or category:subtype.")
        text: str = Field(description="Input text covered by begin:end.")
        category: str = Field(description="Top-level MedDeID entity category.")
        subtype: str | None = Field(
            default=None,
            description="Entity subtype when the label has one.",
        )
        score: float | None = Field(
            default=None,
            description="Model confidence when the span originated from model inference.",
        )
        replacement: str = Field(
            description="Exact bracketed value inserted into deid_text."
        )

    class ProcessingWarning(BaseModel):
        code: str
        message: str

    class DateReplacementProcessing(BaseModel):
        mode: Literal["placeholder", "shift"]
        requested_shift_days: int | None
        minimum_recommended_abs_shift_days: int
        detected_spans: int
        shifted_spans: int
        age_generalized_spans: int
        year_fallback_spans: int
        placeholder_spans: int

    class AgeGranularityPolicyProcessing(BaseModel):
        policy_id: str
        policy_version: str
        sha256: str

    class ProcessingInfo(BaseModel):
        date_replacement: DateReplacementProcessing
        age_granularity_policy: AgeGranularityPolicyProcessing

    class ModelProvenance(BaseModel):
        name: str
        version: str
        resolved_revision: str | None
        bundle_sha256: str

    class LanguageProfileProvenance(BaseModel):
        profile_id: str

    class SoftwareProvenance(BaseModel):
        name: Literal["meddeid"]
        version: str | None

    class InferenceProvenance(BaseModel):
        contract_version: Literal["meddeid.inference-provenance.v1"]
        software: SoftwareProvenance
        model: ModelProvenance
        language_profile: LanguageProfileProvenance

    class DeidentifyResponse(BaseModel):
        deid_text: str
        spans: list[InferenceSpan]
        processing: ProcessingInfo | None = None
        warnings: list[ProcessingWarning] = Field(default_factory=list)
        provenance: InferenceProvenance

    class BatchDocumentRequest(DeidentifyRequest):
        document_id: str = Field(min_length=1, max_length=200)

    class BatchRequest(StrictRequestModel):
        documents: list[BatchDocumentRequest] = Field(
            min_length=1, max_length=document_limit
        )

    class BatchDocumentResponse(BaseModel):
        document_id: str
        deid_text: str
        spans: list[InferenceSpan]
        processing: ProcessingInfo | None = None
        warnings: list[ProcessingWarning] = Field(default_factory=list)
        provenance: InferenceProvenance

    class BatchResponse(BaseModel):
        documents: list[BatchDocumentResponse]

    def result_response(result) -> dict[str, Any]:
        provenance = getattr(result, "provenance", None)
        if not provenance:
            raise RuntimeError(
                "inference result is missing required provenance metadata"
            )
        return {
            "deid_text": result.deid_text,
            "spans": result.spans,
            "processing": getattr(result, "processing", None) or None,
            "warnings": getattr(result, "warnings", []),
            "provenance": provenance,
        }

    configured_model = model or os.environ.get("MEDDEID_MODEL")
    configured_language_profile = (
        language_profile
        if language_profile is not None
        else os.environ.get("MEDDEID_LANGUAGE_PROFILE") or None
    )
    if (
        configured_language_profile is None
        and configured_allowed_profiles is not None
        and len(configured_allowed_profiles) == 1
    ):
        configured_language_profile = next(iter(configured_allowed_profiles))
    if (
        configured_language_profile is not None
        and configured_allowed_profiles is not None
        and configured_language_profile.replace("_", "-")
        not in configured_allowed_profiles
    ):
        raise ValueError(
            f"configured language profile {configured_language_profile!r} is not "
            "permitted by MEDDEID_ALLOWED_LANGUAGE_PROFILES"
        )

    gateway_microbatching = False
    if engine is None:
        if not configured_model:
            raise ValueError(
                "no model selected; set MEDDEID_MODEL to a Hub model ID or local "
                "bundle directory (run `meddeid models` to review public baselines)"
            )
        if (
            configured_allowed_models is not None
            and configured_model not in configured_allowed_models
        ):
            allowed = ", ".join(sorted(configured_allowed_models))
            raise ValueError(
                f"selected model {configured_model!r} is not permitted by "
                f"MEDDEID_ALLOWED_MODELS; allowed values: {allowed}"
            )
        configured_age_policy = (
            age_granularity_config
            if age_granularity_config is not None
            else os.environ.get("MEDDEID_AGE_GRANULARITY_CONFIG") or None
        )
        configured_shift_minimum = (
            min_recommended_date_shift_days
            if min_recommended_date_shift_days is not None
            else int(os.environ.get("MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS", "366"))
        )
        configured_device = device or os.environ.get("MEDDEID_DEVICE")
        selected_device = (
            configured_device.strip().lower() if configured_device is not None else None
        )
        selected_compile_mode = (
            (torch_compile_mode or os.environ.get("MEDDEID_TORCH_COMPILE_MODE", "off"))
            .strip()
            .lower()
        )
        engine = Deidentifier.from_pretrained(
            configured_model,
            revision=os.environ.get("MEDDEID_REVISION") or None,
            local_files_only=os.environ.get("MEDDEID_OFFLINE", "").lower()
            in {"1", "true", "yes"},
            backend=selected_backend,
            device=configured_device,
            triton_url=triton_url or os.environ.get("MEDDEID_TRITON_URL"),
            triton_timeout_seconds=float(
                os.environ.get("MEDDEID_TRITON_TIMEOUT", "30")
            ),
            triton_transport=(
                triton_transport or os.environ.get("MEDDEID_TRITON_TRANSPORT", "json")
            ),
            max_windows_per_batch=(
                int(os.environ["MEDDEID_WINDOW_BATCH_SIZE"])
                if os.environ.get("MEDDEID_WINDOW_BATCH_SIZE")
                else None
            ),
            torch_precision=(
                torch_precision or os.environ.get("MEDDEID_TORCH_PRECISION", "fp32")
            ),
            torch_compile_mode=selected_compile_mode,
            torch_compile_dynamic=selected_compile_dynamic,
            language_profile=configured_language_profile,
            age_granularity_config=configured_age_policy,
            min_recommended_date_shift_days=configured_shift_minimum,
            on_status=lambda message: print(f"[meddeid] {message}", flush=True),
        )
        runtime_device = getattr(getattr(engine, "runtime", None), "device", None)
        runtime_device_type = getattr(runtime_device, "type", None)
        if runtime_device_type is None and runtime_device is not None:
            runtime_device_type = str(runtime_device).split(":", 1)[0]
        torch_uses_gpu_acceleration = selected_backend == "torch" and (
            selected_device in {"cuda", "mps"} or runtime_device_type in {"cuda", "mps"}
        )
        gateway_microbatching = selected_serving_profile == "throughput" and (
            microbatching_policy == "on"
            or (microbatching_policy == "auto" and torch_uses_gpu_acceleration)
        )
        if gateway_microbatching:
            from .runtime.microbatch import MicroBatchRuntime

            runtime_max_windows = int(
                getattr(engine.runtime, "max_windows_per_batch", 32)
            )
            default_microbatch_windows = (
                min(16, runtime_max_windows)
                if selected_backend == "triton"
                else runtime_max_windows
            )
            microbatch_windows = int(
                os.environ.get(
                    "MEDDEID_MICROBATCH_MAX_WINDOWS",
                    default_microbatch_windows,
                )
            )
            if microbatch_windows > runtime_max_windows:
                engine.close()
                raise ValueError(
                    "MEDDEID_MICROBATCH_MAX_WINDOWS cannot exceed "
                    "MEDDEID_WINDOW_BATCH_SIZE"
                )
            max_length = int(engine.bundle.inference.max_length)
            raw_buckets = os.environ.get("MEDDEID_SEQUENCE_LENGTH_BUCKETS")
            if raw_buckets is None or raw_buckets.strip().lower() == "auto":
                if selected_backend == "triton" or selected_compile_mode != "off":
                    sequence_buckets = tuple(
                        sorted(
                            {
                                min(candidate, max_length)
                                for candidate in (64, 128, 256, max_length)
                            }
                        )
                    )
                else:
                    sequence_buckets = ()
            elif raw_buckets.strip().lower() in {"", "none", "off"}:
                sequence_buckets = ()
            else:
                try:
                    sequence_buckets = tuple(
                        int(item.strip())
                        for item in raw_buckets.split(",")
                        if item.strip()
                    )
                except ValueError as exc:
                    engine.close()
                    raise ValueError(
                        "MEDDEID_SEQUENCE_LENGTH_BUCKETS must be auto, off, or "
                        "comma-separated integers"
                    ) from exc
                if not sequence_buckets or max(sequence_buckets) < max_length:
                    engine.close()
                    raise ValueError(
                        "MEDDEID_SEQUENCE_LENGTH_BUCKETS must include a bucket at least "
                        "as large as the model maximum length"
                    )
            try:
                microbatch_runtime = MicroBatchRuntime(
                    engine.runtime,
                    max_windows=microbatch_windows,
                    max_tokens=int(
                        os.environ.get(
                            "MEDDEID_MICROBATCH_MAX_TOKENS",
                            microbatch_windows * max_length,
                        )
                    ),
                    max_wait_ms=float(
                        os.environ.get("MEDDEID_MICROBATCH_MAX_WAIT_MS", "1")
                    ),
                    queue_max_windows=int(
                        os.environ.get("MEDDEID_MICROBATCH_QUEUE_MAX_WINDOWS", "8192")
                    ),
                    queue_max_requests=int(
                        os.environ.get("MEDDEID_MICROBATCH_QUEUE_MAX_REQUESTS", "256")
                    ),
                    sequence_buckets=sequence_buckets,
                )
            except Exception:
                engine.close()
                raise
            engine.runtime = microbatch_runtime

    if automatic_concurrency:
        if selected_serving_profile == "latency":
            concurrent_limit = 1
        elif selected_backend == "triton":
            concurrent_limit = 8
        elif gateway_microbatching:
            concurrent_limit = 16
        else:
            concurrent_limit = 1

    administrative_info = engine.model_info()
    model_details = administrative_info.get("model", {})
    selected_model_source = model_details.get("source") or configured_model
    if configured_allowed_models is not None:
        if selected_model_source not in configured_allowed_models:
            allowed = ", ".join(sorted(configured_allowed_models))
            engine.close()
            raise ValueError(
                f"loaded model source {selected_model_source!r} is not permitted by "
                f"MEDDEID_ALLOWED_MODELS; allowed values: {allowed}"
            )

    contract_details = administrative_info.get("contracts", {})
    supported_profile_items = contract_details.get("language_profiles") or []
    supported_profile_ids = tuple(
        str(item["profile_id"]).replace("_", "-")
        for item in supported_profile_items
        if isinstance(item, dict) and item.get("profile_id")
    )
    if configured_allowed_profiles is not None:
        unknown_profiles = sorted(
            configured_allowed_profiles - set(supported_profile_ids)
        )
        if unknown_profiles:
            engine.close()
            raise ValueError(
                "MEDDEID_ALLOWED_LANGUAGE_PROFILES contains profile(s) not "
                f"declared by the selected model: {', '.join(unknown_profiles)}"
            )
        permitted_profile_ids = tuple(
            profile_id
            for profile_id in supported_profile_ids
            if profile_id in configured_allowed_profiles
        )
    else:
        permitted_profile_ids = supported_profile_ids
    raw_default_profile_id = contract_details.get("default_language_profile")
    default_profile_id = (
        str(raw_default_profile_id).replace("_", "-")
        if raw_default_profile_id is not None
        else None
    )
    if (
        default_profile_id is not None
        and default_profile_id not in permitted_profile_ids
    ):
        engine.close()
        raise ValueError(
            f"default language profile {default_profile_id!r} is not permitted by "
            "MEDDEID_ALLOWED_LANGUAGE_PROFILES"
        )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        engine.close()

    app = FastAPI(
        title="MedDeID",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if expose_docs else None,
        redoc_url="/redoc" if expose_docs else None,
        openapi_url="/openapi.json" if expose_docs else None,
    )

    bearer_scheme = HTTPBearer(auto_error=False)
    api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize(
        bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
        header_key: str | None = Security(api_key_scheme),
    ) -> None:
        if not configured_api_key:
            return
        candidate = bearer.credentials if bearer is not None else header_key
        if candidate is None or not secrets.compare_digest(
            candidate.encode("utf-8"), configured_api_key.encode("utf-8")
        ):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "unauthorized",
                    "message": "provide a valid bearer token or X-API-Key",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

    def enforce_language_profile(
        requested_profile: str | None,
        *,
        document_id: str | None = None,
    ) -> None:
        if requested_profile is None or configured_allowed_profiles is None:
            return
        normalized = requested_profile.strip().replace("_", "-")
        if normalized in permitted_profile_ids:
            return
        context = f" for document {document_id!r}" if document_id is not None else ""
        raise HTTPException(
            status_code=422,
            detail={
                "code": "language_profile_not_allowed",
                "message": (
                    f"language profile {requested_profile!r}{context} is not "
                    "enabled on this server"
                ),
            },
        )

    inference_slot = asyncio.Semaphore(concurrent_limit)

    def add_security_headers(response, request_id: str):
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.middleware("http")
    async def operational_limits(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "").strip()
        if not request_id or len(request_id) > 128:
            request_id = str(uuid4())

        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                supplied_bytes = int(content_length)
            except ValueError:
                supplied_bytes = -1
            if supplied_bytes < 0:
                return add_security_headers(
                    JSONResponse(
                        status_code=400,
                        content={
                            "detail": {
                                "code": "invalid_content_length",
                                "message": "Content-Length must be a non-negative integer",
                            }
                        },
                    ),
                    request_id,
                )
            if supplied_bytes > request_byte_limit:
                return add_security_headers(
                    JSONResponse(
                        status_code=413,
                        content={
                            "detail": {
                                "code": "request_too_large",
                                "message": (
                                    f"request contains {supplied_bytes} bytes; "
                                    f"limit is {request_byte_limit}"
                                ),
                            }
                        },
                    ),
                    request_id,
                )

        acquired = False
        if request.url.path in {"/deidentify", "/deidentify-batch"}:
            try:
                await asyncio.wait_for(inference_slot.acquire(), timeout=queue_timeout)
                acquired = True
            except asyncio.TimeoutError:
                return add_security_headers(
                    JSONResponse(
                        status_code=503,
                        content={
                            "detail": {
                                "code": "server_busy",
                                "message": "inference queue wait limit exceeded",
                            }
                        },
                        headers={"Retry-After": "1"},
                    ),
                    request_id,
                )
        try:
            response = await call_next(request)
        finally:
            if acquired:
                inference_slot.release()
        return add_security_headers(response, request_id)

    @app.get("/health")
    def health():
        if hasattr(engine, "health_info"):
            runtime_health = engine.health_info()
            ready = bool(runtime_health.get("ready", False))
        else:
            current_info = engine.model_info()
            ready = bool(current_info.get("runtime", {}).get("ready", False))
        payload = {
            "status": "ok" if ready else "unavailable",
            "ready": ready,
            "serving": {
                "profile": selected_serving_profile,
                "gateway_microbatching": gateway_microbatching,
                "max_concurrent_requests": concurrent_limit,
            },
            "model": {
                key: model_details.get(key)
                for key in ("name", "version")
                if model_details.get(key) is not None
            },
            "contracts": {
                "language_profiles": [
                    {"profile_id": profile_id} for profile_id in permitted_profile_ids
                ],
                "default_language_profile": (
                    default_profile_id
                    if default_profile_id in permitted_profile_ids
                    else None
                ),
            },
        }
        if not ready:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/live", include_in_schema=False)
    def live():
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root():
        payload = {
            "service": "MedDeID",
            "status": "running",
            "health": "/health",
        }
        if expose_docs:
            payload["docs"] = "/docs"
        if expose_ui:
            payload["ui"] = "/ui"
        return payload

    if expose_ui:

        @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
        def local_ui():
            return HTMLResponse(
                LOCAL_UI,
                headers={
                    "Content-Security-Policy": (
                        "default-src 'none'; style-src 'unsafe-inline'; "
                        "script-src 'unsafe-inline'; connect-src 'self'; "
                        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
                    )
                },
            )

    @app.post(
        "/deidentify",
        response_model=DeidentifyResponse,
        response_model_exclude_unset=True,
        dependencies=[Depends(authorize)],
    )
    def deidentify(payload: DeidentifyRequest):
        if not payload.text.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "empty_text",
                    "message": "text must contain non-whitespace characters",
                },
            )
        enforce_language_profile(payload.metadata.lang)
        try:
            result = engine(
                payload.text,
                metadata=payload.metadata.model_dump(exclude_none=True),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "inference_unavailable", "message": str(exc)},
            ) from exc
        return result_response(result)

    @app.post(
        "/deidentify-batch",
        response_model=BatchResponse,
        response_model_exclude_unset=True,
        dependencies=[Depends(authorize)],
    )
    def deidentify_batch(payload: BatchRequest):
        total_chars = sum(len(document.text) for document in payload.documents)
        if total_chars > batch_char_limit:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "batch_too_large",
                    "message": (
                        f"batch contains {total_chars} characters; "
                        f"limit is {batch_char_limit}"
                    ),
                },
            )
        for document in payload.documents:
            if not document.text.strip():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "empty_text",
                        "message": f"document {document.document_id!r} contains only whitespace",
                    },
                )
            enforce_language_profile(
                document.metadata.lang,
                document_id=document.document_id,
            )
        try:
            inputs = [
                (
                    document.text,
                    document.metadata.model_dump(exclude_none=True),
                )
                for document in payload.documents
            ]
            if hasattr(engine, "deidentify_many"):
                results = engine.deidentify_many(inputs)
            else:  # Compatibility for small custom engines.
                results = [engine(text, metadata=metadata) for text, metadata in inputs]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_request", "message": str(exc)},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "inference_unavailable", "message": str(exc)},
            ) from exc
        return {
            "documents": [
                {
                    "document_id": document.document_id,
                    **result_response(result),
                }
                for document, result in zip(payload.documents, results, strict=True)
            ]
        }

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="meddeid-server",
        description="Run the MedDeID HTTP inference service.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help=(
            "load server settings from a KEY=VALUE file; existing process "
            "environment variables take precedence"
        ),
    )
    parser.add_argument(
        "--serving-profile",
        choices=("latency", "throughput"),
        help="select the latency or throughput deployment preset",
    )
    args = parser.parse_args(argv)
    if args.env_file is not None:
        try:
            _load_server_environment(args.env_file)
        except ValueError as exc:
            parser.exit(2, f"meddeid-server: error: {exc}\n")
    if args.serving_profile is not None:
        os.environ["MEDDEID_SERVING_PROFILE"] = args.serving_profile

    if not os.environ.get("MEDDEID_MODEL"):
        print(
            "meddeid-server: error: MEDDEID_MODEL is required. Set it to a Hub "
            "model ID or local bundle directory; run `meddeid models` to review "
            "the public baselines.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    uvicorn.run(
        "meddeid.server:create_app",
        factory=True,
        host=os.environ.get("MEDDEID_BIND_ADDRESS", os.environ.get("HOST", "0.0.0.0")),
        port=int(os.environ.get("MEDDEID_PORT", os.environ.get("PORT", "8000"))),
        workers=int(os.environ.get("MEDDEID_WORKERS", "1")),
        access_log=_enabled(os.environ.get("MEDDEID_ACCESS_LOG"), default=True),
        proxy_headers=_enabled(os.environ.get("MEDDEID_PROXY_HEADERS")),
        forwarded_allow_ips=os.environ.get("MEDDEID_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
