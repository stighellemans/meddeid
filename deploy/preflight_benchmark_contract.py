#!/usr/bin/env python3
"""Exercise fixture requests against the real API without loading a model.

The injected engine only echoes validated inputs. This checks request contracts,
not inference correctness, model loading, or GPU readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from time import monotonic
from types import SimpleNamespace

DEPLOY = Path(__file__).resolve().parent
# Direct script execution must not expose deploy/triton as a namespace package
# named triton: PyTorch would mistake it for the optional compiler on CPU hosts.
if sys.path[0] == str(DEPLOY):
    sys.path[0] = str(DEPLOY.parent)

from fastapi.testclient import TestClient

from meddeid.server import create_app

read_documents = runpy.run_path(str(DEPLOY / "validate_triton_parity.py"))["read_documents"]
load_batches = runpy.run_path(str(DEPLOY / "benchmark_http.py"))["load_batches"]


class ContractEngine:
    def __init__(self, profiles):
        self.profiles = profiles

    def model_info(self):
        return {
            "model": {"name": "contract-preflight", "version": "test"},
            "contracts": {
                "language_profiles": [{"profile_id": p} for p in self.profiles],
                "default_language_profile": self.profiles[0],
            },
            "runtime": {"ready": True},
        }

    def __call__(self, text, *, metadata):
        return SimpleNamespace(
            deid_text=text, spans=[], warnings=[], processing={},
            provenance={
                "contract_version": "meddeid.inference-provenance.v1",
                "software": {"name": "meddeid", "version": "contract-preflight"},
                "model": {"name": "contract-preflight", "version": "test",
                          "resolved_revision": None, "bundle_sha256": "0" * 64},
                "language_profile": {"profile_id": metadata.get("lang", self.profiles[0])},
            },
        )

    def close(self):
        pass


def error_detail(response):
    detail = response.json().get("detail")
    if isinstance(detail, list):
        # Pydantic includes the input in errors. Retain locations, not note text.
        return [{key: item[key] for key in ("loc", "type", "msg") if key in item}
                for item in detail]
    return detail


def validate_fixture(path, *, profiles, batch_size=16, **app_options):
    started = monotonic()
    report = {
        "passed": False, "scope": "request-contract-only-no-inference",
        "source_commit": os.environ.get("GITHUB_SHA"),
        "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "documents": 0, "checked_requests": 0, "errors": [],
    }
    try:
        if batch_size < 1 or not profiles:
            raise ValueError("positive batch size and model language profiles are required")
        documents = read_documents(path)
        report["documents"] = len(documents)
        ids = [doc["document_id"] for doc in documents]
        if len(set(ids)) != len(ids):
            raise ValueError("fixture contains duplicate document IDs")
        parity_batches = [documents[i:i + batch_size]
                          for i in range(0, len(documents), batch_size)]
        benchmark_batches = load_batches(path, batch_size, repeat_fixture=1)
        benchmark_documents = [doc for batch in benchmark_batches for doc in batch]
        expected = [{**doc, "document_id": doc["document_id"] + "-r0"} for doc in documents]
        if benchmark_documents != expected:
            raise ValueError("parity and benchmark loaders disagree on normalized request inputs")
        app = create_app(
            engine=ContractEngine(profiles), language_profile=profiles[0],
            allowed_language_profiles=profiles, api_key="", require_api_key=False,
            ui_enabled=False, docs_enabled=False, **app_options,
        )
        with TestClient(app) as client:
            # Use the same JSON encoder as both real HTTP clients, including
            # ensure_ascii=True, so byte-limit checks use the actual wire size.
            for consumer, batches in (("parity", parity_batches), ("benchmark", benchmark_batches)):
                for index, batch in enumerate(batches, start=1):
                    response = client.post(
                        "/deidentify-batch", content=json.dumps({"documents": batch}),
                        headers={"Content-Type": "application/json"},
                    )
                    report["checked_requests"] += 1
                    if response.status_code != 200:
                        report["errors"].append({"consumer": consumer, "batch": index,
                                                 "status": response.status_code,
                                                 "detail": error_detail(response)})
            # The benchmark also exercises /deidentify with document_id omitted.
            for index, doc in enumerate(documents, start=1):
                response = client.post(
                    "/deidentify", content=json.dumps({"text": doc["text"], "metadata": doc["metadata"]}),
                    headers={"Content-Type": "application/json"},
                )
                report["checked_requests"] += 1
                if response.status_code != 200:
                    report["errors"].append({"consumer": "single", "document_index": index,
                                             "status": response.status_code,
                                             "detail": error_detail(response)})
        report["passed"] = not report["errors"]
    except (ValueError, TypeError, KeyError) as exc:
        report["errors"].append({"message": str(exc)})
    report["elapsed_seconds"] = round(monotonic() - started, 3)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--language-profiles", required=True, help="comma-separated catalog profiles")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_fixture(args.fixture, profiles=[p.strip() for p in args.language_profiles.split(",") if p.strip()],
                              batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
