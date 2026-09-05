#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_METADATA_KEYS = frozenset(
    {
        "lang",
        "patient",
        "caregivers",
        "document_creation_date",
        "date_shift_days",
        "known_values",
    }
)


def request_json(
    url: str, *, api_key: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(request, timeout=300) as response:
            parsed = json.loads(response.read())
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"unexpected non-object response from {url}")
    return parsed


def load_batches(
    path: Path,
    batch_size: int,
    repeat_fixture: int,
    *,
    min_chars: int = 0,
    max_chars: int | None = None,
) -> list[list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if "document_id" not in row or "text" not in row:
            raise ValueError(f"{path}:{line_number}: document_id and text are required")
        text = str(row["text"])
        if len(text) < min_chars or (max_chars is not None and len(text) > max_chars):
            continue
        metadata = row.get("metadata")
        if metadata is None and row.get("metadata_json") is not None:
            metadata = json.loads(row["metadata_json"])
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"{path}:{line_number}: metadata must be an object")
        # Public evaluation datasets may keep benchmark/provenance fields next
        # to trusted request metadata. Forward only the current suite's API
        # contract; strict request validation must remain enabled in production.
        api_metadata = {
            key: value
            for key, value in (metadata or {}).items()
            if key in API_METADATA_KEYS
        }
        for repetition in range(repeat_fixture):
            documents.append(
                {
                    "document_id": f"{row['document_id']}-r{repetition}",
                    "text": text,
                    "metadata": api_metadata,
                }
            )
    if not documents:
        raise ValueError(f"{path} contains no documents")
    return [
        documents[begin : begin + batch_size]
        for begin in range(0, len(documents), batch_size)
    ]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def run_request(
    url: str,
    api_key: str,
    batch: list[dict[str, Any]],
    *,
    endpoint_mode: str,
) -> tuple[float, int, int, dict[str, Any]]:
    started = perf_counter()
    if endpoint_mode == "single":
        document = batch[0]
        response = request_json(
            url,
            api_key=api_key,
            payload={"text": document["text"], "metadata": document["metadata"]},
        )
        returned = [response]
        actual_ids = [document["document_id"]]
    else:
        response = request_json(url, api_key=api_key, payload={"documents": batch})
        returned = response.get("documents") or []
        actual_ids = [item.get("document_id") for item in returned]
    latency = perf_counter() - started
    expected_ids = [item["document_id"] for item in batch]
    if actual_ids != expected_ids:
        raise RuntimeError(
            "benchmark response document IDs/order differ from the request"
        )
    provenance = returned[0].get("provenance") if returned else None
    if not isinstance(provenance, dict):
        raise TypeError("benchmark response is missing result provenance")
    return latency, len(batch), sum(len(item["text"]) for item in batch), provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a MedDeID batch HTTP endpoint without retaining text"
    )
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("MEDDEID_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--endpoint-mode",
        choices=("single", "batch"),
        default="batch",
        help="exercise /deidentify or /deidentify-batch",
    )
    parser.add_argument(
        "--repeat-fixture",
        type=int,
        default=1,
        help="repeat each synthetic fixture row with a unique document ID",
    )
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("provide --api-key or MEDDEID_API_KEY")
    if not 1 <= args.batch_size <= 32:
        parser.error("--batch-size must be between 1 and 32")
    if args.endpoint_mode == "single" and args.batch_size != 1:
        parser.error("--endpoint-mode=single requires --batch-size=1")
    if args.min_chars < 0 or (
        args.max_chars is not None and args.max_chars < args.min_chars
    ):
        parser.error("character bounds must be non-negative and ordered")
    if (
        args.warmup_requests < 0
        or args.requests < 1
        or args.concurrency < 1
        or args.repeat_fixture < 1
    ):
        parser.error(
            "warmup must be non-negative; requests, concurrency, and repeat-fixture "
            "must be positive"
        )

    fixture = args.fixture.resolve()
    batches = load_batches(
        fixture,
        args.batch_size,
        args.repeat_fixture,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    endpoint_path = (
        "deidentify" if args.endpoint_mode == "single" else "deidentify-batch"
    )
    endpoint = f"{args.base_url.rstrip('/')}/{endpoint_path}"
    health = request_json(f"{args.base_url.rstrip('/')}/health", api_key=args.api_key)

    cold_request = run_request(
        endpoint,
        args.api_key,
        batches[0],
        endpoint_mode=args.endpoint_mode,
    )

    for index in range(args.warmup_requests):
        run_request(
            endpoint,
            args.api_key,
            batches[index % len(batches)],
            endpoint_mode=args.endpoint_mode,
        )

    selected = [batches[index % len(batches)] for index in range(args.requests)]
    started = perf_counter()
    measurements: list[tuple[float, int, int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_request,
                endpoint,
                args.api_key,
                batch,
                endpoint_mode=args.endpoint_mode,
            )
            for batch in selected
        ]
        for future in as_completed(futures):
            measurements.append(future.result())
    wall_seconds = perf_counter() - started

    latencies = [item[0] for item in measurements]
    documents = sum(item[1] for item in measurements)
    characters = sum(item[2] for item in measurements)
    fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
    report = {
        "schema": "meddeid.http-benchmark.v1",
        "base_url": args.base_url,
        "fixture_sha256": fixture_sha256,
        "configuration": {
            "batch_size": args.batch_size,
            "endpoint_mode": args.endpoint_mode,
            "repeat_fixture": args.repeat_fixture,
            "documents_per_request": {
                "min": min(len(batch) for batch in selected),
                "max": max(len(batch) for batch in selected),
                "mean": sum(len(batch) for batch in selected) / len(selected),
            },
            "warmup_requests": args.warmup_requests,
            "measured_requests": args.requests,
            "concurrency": args.concurrency,
            "character_filter": {
                "minimum": args.min_chars,
                "maximum": args.max_chars,
            },
        },
        "health": health,
        "first_request": {
            "latency_seconds": cold_request[0],
            "documents": cold_request[1],
            "characters": cold_request[2],
        },
        "provenance": measurements[0][3],
        "results": {
            "wall_seconds": wall_seconds,
            "documents": documents,
            "characters": characters,
            "documents_per_second": documents / wall_seconds,
            "characters_per_second": characters / wall_seconds,
            "request_latency_seconds": {
                "min": min(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "max": max(latencies),
            },
        },
        "privacy": "No request or response text is retained in this report.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
