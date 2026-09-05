#!/usr/bin/env python3
"""Sample aggregate NVIDIA GPU utilization without retaining request data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
from threading import Event
from time import monotonic, sleep


FIELDS = (
    "index",
    "name",
    "compute_capability",
    "driver_version",
    "memory_used_mib",
    "utilization_gpu_percent",
    "power_draw_watts",
)
QUERY = (
    "index,name,compute_cap,driver_version,memory.used,utilization.gpu,power.draw"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record aggregate NVIDIA GPU metrics")
    parser.add_argument("--device", default="0")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")

    stopped = Event()
    signal.signal(signal.SIGINT, lambda *_args: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_args: stopped.set())
    samples: list[dict[str, object]] = []
    started = monotonic()
    while not stopped.is_set():
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={args.device}",
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        values = [item.strip() for item in output.split(",")]
        if len(values) != len(FIELDS):
            raise RuntimeError(f"unexpected nvidia-smi row: {output}")
        sample: dict[str, object] = dict(zip(FIELDS[:4], values[:4], strict=True))
        for name, value in zip(FIELDS[4:], values[4:], strict=True):
            sample[name] = None if value in {"N/A", "[Not Supported]"} else float(value)
        sample["elapsed_seconds"] = monotonic() - started
        samples.append(sample)
        sleep(args.interval)

    numeric = FIELDS[4:]
    report = {
        "schema": "meddeid.nvidia-monitor.v1",
        "device": args.device,
        "duration_seconds": monotonic() - started,
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "gpu": {name: samples[0][name] for name in FIELDS[:4]} if samples else {},
        "maximum": {
            name: max(
                (sample[name] for sample in samples if sample[name] is not None),
                default=None,
            )
            for name in numeric
        },
        "privacy": "Only aggregate device metrics are retained.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
