#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
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


def infer(base_url: str, api_key: str, documents: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for begin in range(0, len(documents), batch_size):
        response = request_json(
            f"{base_url.rstrip('/')}/deidentify-batch",
            api_key=api_key,
            payload={"documents": documents[begin : begin + batch_size]},
        )
        outputs.extend(response.get("documents") or [])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PyTorch and Triton MedDeID HTTP semantics")
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--reference-url", default="http://127.0.0.1:8001")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("MEDDEID_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("deploy/triton/parity-report.json"))
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or MEDDEID_API_KEY")
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")

    documents = read_documents(args.fixture)
    reference_health = request_json(f"{args.reference_url.rstrip('/')}/health", api_key=args.api_key)
    candidate_health = request_json(f"{args.candidate_url.rstrip('/')}/health", api_key=args.api_key)
    if reference_health.get("status") != "ok" or candidate_health.get("status") != "ok":
        raise RuntimeError("reference and candidate servers must both be healthy")

    reference_outputs = infer(args.reference_url, args.api_key, documents, args.batch_size)
    candidate_outputs = infer(args.candidate_url, args.api_key, documents, args.batch_size)
    reference_identity = result_identity(reference_outputs[0])
    candidate_identity = result_identity(candidate_outputs[0])
    identity_matches = reference_identity == candidate_identity
    candidate_by_id = {str(row.get("document_id")): row for row in candidate_outputs}
    differences: list[dict[str, Any]] = []
    for reference in reference_outputs:
        document_id = str(reference.get("document_id"))
        candidate = candidate_by_id.get(document_id)
        expected = semantic_document(reference)
        actual = semantic_document(candidate or {})
        if expected != actual:
            differences.append(
                {"document_id": document_id, "reference": expected, "candidate": actual}
            )

    passed = identity_matches and not differences and len(candidate_outputs) == len(documents)
    report = {
        "schema": "meddeid.triton-parity.v1",
        "passed": passed,
        "documents": len(documents),
        "reference_url": args.reference_url,
        "candidate_url": args.candidate_url,
        "model_identity_matches": identity_matches,
        "reference_identity": reference_identity,
        "candidate_identity": candidate_identity,
        "semantic_differences": differences,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": passed, "documents": len(documents), "differences": len(differences)}))
    if not passed:
        print(f"Parity failed; inspect {args.output}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
