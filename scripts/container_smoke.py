#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    *,
    api_key: str | None = None,
    payload: dict | None = None,
    timeout: float = 10.0,
) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=data,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_until_ready(base_url: str, timeout_seconds: float) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            status, payload = request_json(base_url, "/health", timeout=3)
            if status == 200 and payload.get("status") == "ok":
                return payload
            last_error = f"health returned HTTP {status}: {payload}"
        except (
            URLError,
            TimeoutError,
            ConnectionError,
            http.client.RemoteDisconnected,
            json.JSONDecodeError,
        ) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"MedDeID did not become ready: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a running MedDeID API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=10.0,
        help="timeout for inference requests, including first-run compilation",
    )
    args = parser.parse_args()

    health = wait_until_ready(args.base_url, args.startup_timeout)
    assert health["ready"] is True

    sample = {
        "text": "Patiënt Jan Peeters belde 0470 12 34 56.",
        "metadata": {"patient": {"given_name": "Jan", "family_name": "Peeters"}},
    }
    if args.api_key:
        status, body = request_json(
            args.base_url,
            "/deidentify",
            payload=sample,
            timeout=args.request_timeout,
        )
        assert status == 401, body

    status, body = request_json(
        args.base_url,
        "/deidentify",
        api_key=args.api_key,
        payload=sample,
        timeout=args.request_timeout,
    )
    assert status == 200, body
    assert body["deid_text"] == ("Patiënt [Name:Patient] belde [Contactdetails]."), body
    assert list(body) == [
        "deid_text",
        "spans",
        "processing",
        "warnings",
        "provenance",
    ], body
    assert body["provenance"]["model"]["resolved_revision"], body
    assert body["provenance"]["language_profile"]["profile_id"], body

    status, body = request_json(
        args.base_url,
        "/deidentify-batch",
        api_key=args.api_key,
        payload={
            "documents": [
                {"document_id": "one", **sample},
                {"document_id": "two", "text": "Geen persoonsgegevens."},
            ]
        },
        timeout=args.request_timeout,
    )
    assert status == 200, body
    assert [item["document_id"] for item in body["documents"]] == ["one", "two"]
    provenance = body["documents"][0]["provenance"]
    print(
        json.dumps(
            {
                "status": "ok",
                "model": provenance["model"],
                "software": provenance["software"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
