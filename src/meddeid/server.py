from contextlib import asynccontextmanager
import asyncio
import os
import secrets
from typing import Any
from uuid import uuid4

from .api import Deidentifier


DEFAULT_MODEL = "stighellemans/meddeid-dutch-synth"
DEFAULT_MAX_INPUT_CHARS = 20_000
DEFAULT_MAX_BATCH_DOCUMENTS = 32
DEFAULT_MAX_BATCH_CHARS = 200_000
DEFAULT_MAX_REQUEST_BYTES = 2_000_000
DEFAULT_MAX_CONCURRENT_REQUESTS = 1
DEFAULT_QUEUE_TIMEOUT_SECONDS = 30.0


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
    textarea, input { box-sizing: border-box; width: 100%; padding: .75rem; border: 1px solid #93a39d; border-radius: 7px; font: inherit; }
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
  <p>De-identify one Dutch clinical note on this server.</p>
  <p class="notice"><strong>Important:</strong> Model output can still contain identifying information. Validate it and use human review where an error could cause harm.</p>
  <section class="card" aria-labelledby="input-title">
    <h2 id="input-title">Note</h2>
    <label for="api-key">API key</label>
    <input id="api-key" type="password" autocomplete="off" spellcheck="false" aria-describedby="key-help">
    <div id="key-help" class="muted">Stored only in this browser tab and sent to this MedDeID server.</div>
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
  let output = '';
  run.addEventListener('click', async () => {
    const text = byId('note').value;
    const key = byId('api-key').value;
    if (!text.trim()) { status.textContent = 'Enter a note first.'; status.className = 'error'; return; }
    const patient = {};
    if (byId('given-name').value.trim()) patient.given_name = byId('given-name').value.trim();
    if (byId('family-name').value.trim()) patient.family_name = byId('family-name').value.trim();
    const metadata = Object.keys(patient).length ? {patient} : {};
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
      status.textContent = `Done. ${body.spans.length} identifying span${body.spans.length === 1 ? '' : 's'} detected.`;
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


def create_app(
    model: str | None = None,
    *,
    backend: str | None = None,
    device: str | None = None,
    triton_url: str | None = None,
    max_input_chars: int | None = None,
    max_batch_documents: int | None = None,
    max_batch_chars: int | None = None,
    max_request_bytes: int | None = None,
    max_concurrent_requests: int | None = None,
    queue_timeout_seconds: float | None = None,
    api_key: str | None = None,
    require_api_key: bool | None = None,
    docs_enabled: bool | None = None,
    ui_enabled: bool | None = None,
    engine: Any | None = None,
):
    try:
        from fastapi import Depends, FastAPI, HTTPException, Request, Security
        from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
        from fastapi.responses import HTMLResponse, JSONResponse
        from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
    except ImportError as exc:
        raise RuntimeError("install meddeid[server]") from exc

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
    concurrent_limit = int(
        max_concurrent_requests
        if max_concurrent_requests is not None
        else os.environ.get(
            "MEDDEID_MAX_CONCURRENT_REQUESTS", DEFAULT_MAX_CONCURRENT_REQUESTS
        )
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

    class FlexibleModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    class PersonMetadata(FlexibleModel):
        given_name: str | None = None
        family_name: str | None = None

    class PatientMetadata(PersonMetadata):
        birth_date: str | None = None

    class KnownValueMetadata(FlexibleModel):
        value: str = Field(min_length=1)
        label: str = Field(min_length=1)

        @field_validator("label")
        @classmethod
        def canonical_label(cls, value: str) -> str:
            from meddeid_core.taxonomy import BERT_ENTITY_LABELS

            if value not in BERT_ENTITY_LABELS:
                raise ValueError("label must be one of the 14 canonical MedDeID labels")
            return value

    class RecordMetadata(FlexibleModel):
        lang: str | None = None
        patient: PatientMetadata | None = None
        caregivers: list[PersonMetadata] | None = None
        document_creation_date: str | None = None
        date_shift_days: int | None = None
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
            return value

    class DeidentifyRequest(BaseModel):
        text: str = Field(min_length=1, max_length=limit)
        metadata: RecordMetadata = Field(default_factory=RecordMetadata)

    class DeidentifyResponse(BaseModel):
        text: str
        deid_text: str
        spans: list[dict[str, Any]]

    class BatchDocumentRequest(DeidentifyRequest):
        document_id: str = Field(min_length=1, max_length=200)

    class BatchRequest(BaseModel):
        documents: list[BatchDocumentRequest] = Field(
            min_length=1, max_length=document_limit
        )

    class BatchDocumentResponse(DeidentifyResponse):
        document_id: str

    class BatchResponse(BaseModel):
        documents: list[BatchDocumentResponse]

    if engine is None:
        engine = Deidentifier.from_pretrained(
            model or os.environ.get("MEDDEID_MODEL", DEFAULT_MODEL),
            revision=os.environ.get("MEDDEID_REVISION") or None,
            local_files_only=os.environ.get("MEDDEID_OFFLINE", "").lower()
            in {"1", "true", "yes"},
            backend=backend or os.environ.get("MEDDEID_BACKEND", "torch"),
            device=device or os.environ.get("MEDDEID_DEVICE"),
            triton_url=triton_url or os.environ.get("MEDDEID_TRITON_URL"),
            triton_timeout_seconds=float(os.environ.get("MEDDEID_TRITON_TIMEOUT", "30")),
            max_windows_per_batch=(
                int(os.environ["MEDDEID_WINDOW_BATCH_SIZE"])
                if os.environ.get("MEDDEID_WINDOW_BATCH_SIZE")
                else None
            ),
            on_status=lambda message: print(f"[meddeid] {message}", flush=True),
        )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        engine.close()

    app = FastAPI(
        title="MedDeID",
        version="0.1.1",
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
                return add_security_headers(JSONResponse(
                    status_code=400,
                    content={
                        "detail": {
                            "code": "invalid_content_length",
                            "message": "Content-Length must be a non-negative integer",
                        }
                    },
                ), request_id)
            if supplied_bytes > request_byte_limit:
                return add_security_headers(JSONResponse(
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
                ), request_id)

        acquired = False
        if request.url.path in {"/deidentify", "/deidentify-batch"}:
            try:
                await asyncio.wait_for(
                    inference_slot.acquire(), timeout=queue_timeout
                )
                acquired = True
            except asyncio.TimeoutError:
                return add_security_headers(JSONResponse(
                    status_code=503,
                    content={
                        "detail": {
                            "code": "server_busy",
                            "message": "inference queue wait limit exceeded",
                        }
                    },
                    headers={"Retry-After": "1"},
                ), request_id)
        try:
            response = await call_next(request)
        finally:
            if acquired:
                inference_slot.release()
        return add_security_headers(response, request_id)

    @app.get("/health")
    def health():
        info = engine.model_info()
        ready = bool(info.get("runtime", {}).get("ready", False))
        payload = {"status": "ok" if ready else "unavailable", **info}
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
        dependencies=[Depends(authorize)],
    )
    def deidentify(payload: DeidentifyRequest):
        if not payload.text.strip():
            raise HTTPException(
                status_code=422,
                detail={"code": "empty_text", "message": "text must contain non-whitespace characters"},
            )
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
        return {
            "text": result.text,
            "deid_text": result.deid_text,
            "spans": result.spans,
        }

    @app.post(
        "/deidentify-batch",
        response_model=BatchResponse,
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
                    "text": result.text,
                    "deid_text": result.deid_text,
                    "spans": result.spans,
                }
                for document, result in zip(payload.documents, results, strict=True)
            ]
        }

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "meddeid.server:create_app",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=int(os.environ.get("MEDDEID_WORKERS", "1")),
        access_log=_enabled(os.environ.get("MEDDEID_ACCESS_LOG"), default=True),
        proxy_headers=_enabled(os.environ.get("MEDDEID_PROXY_HEADERS")),
        forwarded_allow_ips=os.environ.get("MEDDEID_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
