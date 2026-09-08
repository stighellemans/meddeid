#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, *, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=180) as response:
            parsed = json.loads(response.read())
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"unexpected non-object response from {url}")
    return parsed


def read_documents(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("document_id", "text"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        documents.append(
            {
                "document_id": str(row["document_id"]),
                "text": str(row["text"]),
                "metadata": row.get("metadata") or {},
            }
        )
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return documents


def result_identity(payload: dict[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance") or {}
    return {
        "software": provenance.get("software"),
        "model": provenance.get("model"),
    }


def semantic_document(payload: dict[str, Any]) -> dict[str, Any]:
    spans = []
    for span in payload.get("spans") or []:
        spans.append(
            {
                key: value
                for key, value in span.items()
                if key not in {"score", "confidence", "bio_score", "label_score"}
            }
        )
    return {
        "document_id": payload.get("document_id"),
        "deid_text": payload.get("deid_text"),
        "spans": spans,
        "language_profile": (payload.get("provenance") or {}).get(
            "language_profile"
        ),
        "warnings": payload.get("warnings") or [],
        "processing": payload.get("processing"),
    }


def validate_outputs(outputs: Any, documents: list[dict[str, Any]], side: str) -> None:
    if not isinstance(outputs, list) or any(not isinstance(row, dict) for row in outputs):
        raise ValueError(f"{side} response documents must be a list of objects")
    expected = [row["document_id"] for row in documents]
    actual = [row.get("document_id") for row in outputs]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError(f"{side} response has missing, duplicate, or unexpected document IDs")
    for row in outputs:
        if not {"document_id", "deid_text", "spans", "provenance"} <= row.keys():
            raise ValueError(f"{side} response is missing required semantic fields")
        identity = result_identity(row)
        if not identity["software"] or not identity["model"]:
            raise ValueError(f"{side} response is missing software or model identity")


def selected_batches(documents, batch_size, document_id=None):
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    if document_id is None:
        return batches
    selected = [batch for batch in batches if any(row["document_id"] == document_id for row in batch)]
    if not selected:
        raise ValueError(f"document ID not found: {document_id}")
    return selected


def compare(documents, *, reference_url, candidate_url, api_key, batch_size, output,
            fail_fast=False, document_id=None, fixture_sha256=None):
    ids = [row["document_id"] for row in documents]
    if len(set(ids)) != len(ids):
        raise ValueError("fixture has duplicate document IDs")
    batches = selected_batches(documents, batch_size, document_id)
    total = sum(len(batch) for batch in batches)
    started = monotonic()
    report = {
        "schema": "meddeid.triton-parity.v1", "passed": False, "complete": False,
        "documents": total, "fixture_documents": len(documents), "checked_documents": 0,
        "fixture_sha256": fixture_sha256, "batch_size": batch_size,
        "selected_document_id": document_id,
        "reference_url": reference_url, "candidate_url": candidate_url,
        "model_identity_matches": True, "reference_identity": None, "candidate_identity": None,
        "semantic_differences": [], "errors": [],
        "source_commit": os.environ.get("GITHUB_SHA"),
    }

    def checkpoint():
        report["elapsed_seconds"] = round(monotonic() - started, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    checkpoint()
    try:
        for side, url in (("reference", reference_url), ("candidate", candidate_url)):
            health = request_json(f"{url.rstrip('/')}/health", api_key=api_key)
            report[f"{side}_health"] = health
            if health.get("status") != "ok" or health.get("ready") is not True:
                raise RuntimeError(f"{side} server must be healthy and ready")
        for batch_number, batch in enumerate(batches, 1):
            outputs = []
            for side, url in (("reference", reference_url), ("candidate", candidate_url)):
                response = request_json(f"{url.rstrip('/')}/deidentify-batch", api_key=api_key,
                                        payload={"documents": batch})
                rows = response.get("documents")
                validate_outputs(rows, batch, side)
                outputs.append({row["document_id"]: row for row in rows})
            for document in batch:
                key = document["document_id"]
                reference, candidate = outputs[0][key], outputs[1][key]
                ri, ci = result_identity(reference), result_identity(candidate)
                if report["reference_identity"] is None:
                    report["reference_identity"], report["candidate_identity"] = ri, ci
                if ri != ci or ri != report["reference_identity"] or ci != report["candidate_identity"]:
                    report["model_identity_matches"] = False
                expected, actual = semantic_document(reference), semantic_document(candidate)
                if expected != actual:
                    fields = [field for field in expected if expected[field] != actual[field]]
                    report["semantic_differences"].append({"document_id": key,
                        "reference": expected, "candidate": actual, "changed_fields": fields,
                        "batch_document_ids": [row["document_id"] for row in batch]})
                    print(json.dumps({"event": "semantic_difference", "document_id": key,
                                      "changed_fields": fields}), flush=True)
            report["checked_documents"] += len(batch)
            print(json.dumps({"event": "parity_progress", "batch": batch_number,
                              "checked": report["checked_documents"], "total": total,
                              "differences": len(report["semantic_differences"])}), flush=True)
            checkpoint()
            if fail_fast and (report["semantic_differences"] or not report["model_identity_matches"]):
                break
        report["complete"] = report["checked_documents"] == total
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["passed"] = bool(report["complete"] and report["model_identity_matches"]
                            and not report["semantic_differences"] and not report["errors"])
    checkpoint()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and Triton MedDeID HTTP semantics")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--reference-url", default="http://127.0.0.1:8001")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("MEDDEID_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("deploy/triton/parity-report.json"))
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failing batch")
    parser.add_argument("--document-id", help="Diagnostic replay of the original batch containing this ID; not a full release gate")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or MEDDEID_API_KEY")
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")

    documents = read_documents(args.fixture)
    report = compare(documents, reference_url=args.reference_url, candidate_url=args.candidate_url,
                     api_key=args.api_key, batch_size=args.batch_size, output=args.output,
                     fail_fast=args.fail_fast, document_id=args.document_id,
                     fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest())
    print(json.dumps({"passed": report["passed"], "documents": report["documents"],
                      "checked": report["checked_documents"],
                      "differences": len(report["semantic_differences"]), "errors": report["errors"]}), flush=True)
    if not report["passed"]:
        print(f"Parity failed; inspect {args.output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
