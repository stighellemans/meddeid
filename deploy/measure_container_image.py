#!/usr/bin/env python3
"""Record comparable local image size and layer information."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any


class CountingWriter:
    def __init__(self) -> None:
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        self.bytes_written += len(data)
        return len(data)

    def flush(self) -> None:
        return None


def docker_json(*args: str):
    output = subprocess.check_output(["docker", *args], text=True)
    return json.loads(output)


def compressed_save_size(image: str) -> int:
    process = subprocess.Popen(
        ["docker", "image", "save", image], stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    counter = CountingWriter()
    with gzip.GzipFile(fileobj=counter, mode="wb", compresslevel=6, mtime=0) as archive:
        for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
            archive.write(chunk)
    if process.wait() != 0:
        raise RuntimeError(f"docker image save failed for {image}")
    return counter.bytes_written


def parse_docker_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kMGT]?B)", value)
    if match is None:
        raise ValueError(f"unexpected Docker size: {value}")
    multiplier = {
        "B": 1,
        "kB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "TB": 1_000_000_000_000,
    }[match.group(2)]
    return int(Decimal(match.group(1)) * multiplier)


def load_budget(path: Path, key: str) -> dict[str, int]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "meddeid.container-image-size-budgets.v1":
        raise ValueError(f"unsupported image-size budget schema: {path}")
    try:
        budget = payload["budgets"][key]
    except KeyError as exc:
        raise ValueError(f"unknown image-size budget {key!r}: {path}") from exc
    required = {"max_portable_save_gzip_bytes", "max_content_size_bytes"}
    if set(budget) != required or not all(
        isinstance(budget[field], int) and budget[field] > 0 for field in required
    ):
        raise ValueError(f"invalid image-size budget {key!r}: {path}")
    return budget


def size_budget_failures(report: dict[str, Any], budget: dict[str, int]) -> list[str]:
    failures = []
    checks = (
        ("portable_save_gzip_bytes", "max_portable_save_gzip_bytes"),
        ("content_size_bytes", "max_content_size_bytes"),
    )
    for actual_field, maximum_field in checks:
        actual = int(report[actual_field])
        maximum = budget[maximum_field]
        if actual > maximum:
            failures.append(
                f"{actual_field} {actual} exceeds {maximum_field} {maximum}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure a local container image")
    parser.add_argument("image")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--budget-key")
    args = parser.parse_args()
    if (args.budget_file is None) != (args.budget_key is None):
        parser.error("--budget-file and --budget-key must be supplied together")

    inspection = docker_json("image", "inspect", args.image)[0]
    history_lines = subprocess.check_output(
        [
            "docker",
            "image",
            "history",
            "--no-trunc",
            "--format",
            "{{json .}}",
            args.image,
        ],
        text=True,
    ).splitlines()
    layers = [json.loads(line) for line in history_lines if line]
    unpacked_history_size = sum(parse_docker_size(layer["Size"]) for layer in layers)
    report = {
        "schema": "meddeid.container-image-size.v1",
        "image": args.image,
        "image_id": inspection["Id"],
        "platform": {
            "architecture": inspection["Architecture"],
            "os": inspection["Os"],
        },
        "content_size_bytes": unpacked_history_size,
        "engine_reported_size_bytes": inspection["Size"],
        "content_size_note": (
            "Unpacked layer-size sum from docker image history. Newer containerd-backed "
            "Docker engines can report compressed content in image inspect Size."
        ),
        "portable_save_gzip_bytes": compressed_save_size(args.image),
        "portable_save_note": (
            "Gzip-compressed docker-save size is a consistent pull-size proxy; "
            "the registry transfer size can differ with its compression and shared layers."
        ),
        "labels": inspection.get("Config", {}).get("Labels") or {},
        "layers": layers,
    }
    failures: list[str] = []
    if args.budget_file is not None:
        budget = load_budget(args.budget_file, args.budget_key)
        report["budget"] = {"key": args.budget_key, **budget}
        failures = size_budget_failures(report, budget)
        report["budget"]["passed"] = not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "content_size_bytes": report["content_size_bytes"],
                "portable_save_gzip_bytes": report["portable_save_gzip_bytes"],
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit("image-size budget failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
