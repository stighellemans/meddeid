from contextlib import asynccontextmanager
import os
from typing import Any

from .api import Deidentifier


DEFAULT_MODEL = "stighellemans/meddeid-dutch-synth"
DEFAULT_MAX_INPUT_CHARS = 20_000
DEFAULT_MAX_BATCH_DOCUMENTS = 32
DEFAULT_MAX_BATCH_CHARS = 200_000


def create_app(
    model: str | None = None,
    *,
    backend: str | None = None,
    device: str | None = None,
    triton_url: str | None = None,
    max_input_chars: int | None = None,
    max_batch_documents: int | None = None,
    max_batch_chars: int | None = None,
    engine: Any | None = None,
):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
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

    app = FastAPI(title="MedDeID", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        info = engine.model_info()
        ready = bool(info.get("runtime", {}).get("ready", False))
        payload = {"status": "ok" if ready else "unavailable", **info}
        if not ready:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/deidentify", response_model=DeidentifyResponse)
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

    @app.post("/deidentify-batch", response_model=BatchResponse)
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
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
