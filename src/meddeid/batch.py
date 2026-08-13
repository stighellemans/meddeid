"""Canonical JSONL batch inference with resumable, checksummed output."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from meddeid_core.artifacts import OFFSET_UNIT, SCHEMA_VERSION, sha256_file, validate_document_set


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_number}: invalid JSON") from exc
        aliases = [key for key in ("doc_id", "plain_text", "annotations", "entities") if key in row]
        if aliases:
            raise ValueError(f"{path} line {line_number}: non-canonical field {aliases[0]!r}")
        rows.append(row)
    validate_document_set(rows)
    return rows


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run_batch(
    engine: Any,
    input_path: Path,
    output_path: Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    inputs = _read_rows(input_path)
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        if not resume and not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --resume or --overwrite")
        if resume:
            existing_rows = _read_rows(output_path)
            existing = {row["document_id"]: row for row in existing_rows}
            input_by_id = {row["document_id"]: row for row in inputs}
            extra = sorted(set(existing) - set(input_by_id))
            if extra:
                raise ValueError(f"resume output has document IDs absent from input: {extra[:5]}")
            for document_id, row in existing.items():
                if row["text"] != input_by_id[document_id]["text"]:
                    raise ValueError(f"resume text mismatch for {document_id}")

    started = perf_counter()
    processed = 0
    resumed = 0
    outputs: list[dict[str, Any]] = []
    total = len(inputs)
    for index, row in enumerate(inputs, start=1):
        document_id = row["document_id"]
        if document_id in existing:
            outputs.append(existing[document_id])
            resumed += 1
            reused = True
        else:
            result = engine(row["text"], metadata=row.get("metadata") or {})
            outputs.append(
                {
                    "document_id": document_id,
                    "text": row["text"],
                    "spans": result.spans,
                    "deid_text": result.deid_text,
                    "metadata": row.get("metadata") or {},
                }
            )
            processed += 1
            reused = False
            # A complete atomic snapshot after every new document makes interruption safe.
            _write_rows(output_path, outputs)
        if progress is not None:
            progress(
                {
                    "completed": index,
                    "total": total,
                    "document_id": document_id,
                    "resumed": reused,
                    "elapsed_seconds": perf_counter() - started,
                }
            )
    _write_rows(output_path, outputs)

    bundle = engine.bundle
    elapsed = perf_counter() - started
    model_info = engine.model_info() if hasattr(engine, "model_info") else {}
    runtime = model_info.get("runtime", {})
    model_details = model_info.get("model", {})
    manifest = {
        "manifest_version": "meddeid.inference-run.v1",
        "contracts": {
            "schema_version": SCHEMA_VERSION,
            "offset_unit": OFFSET_UNIT,
            "language_profile": bundle.postprocess.profile_id,
            "language_profile_version": bundle.postprocess.profile_version,
        },
        "model": {
            "name": bundle.name,
            "version": bundle.model_version,
            "bundle_sha256": bundle.contract_hash(),
            "source": model_details.get("source"),
            "requested_revision": model_details.get("requested_revision"),
            "resolved_revision": model_details.get("resolved_revision"),
        },
        "runtime": runtime,
        "environment": model_info.get("environment", {}),
        "files": {
            "input": {"filename": input_path.name, "sha256": sha256_file(input_path)},
            "predictions": {"filename": output_path.name, "sha256": sha256_file(output_path)},
        },
        "counts": {
            "documents": len(outputs),
            "processed": processed,
            "resumed": resumed,
            "failed": 0,
            "spans": sum(len(row["spans"]) for row in outputs),
        },
        "timing": {
            "elapsed_seconds": elapsed,
            "documents_per_second": processed / elapsed if elapsed > 0 else None,
        },
        "evaluation": "meddeid-eval score --gold GOLD.jsonl --predictions " + str(output_path),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
