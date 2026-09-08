#!/usr/bin/env python3
"""Wait for MedDeID readiness and record startup timing."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from time import sleep, time_ns
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a MedDeID health endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--started-at-ns", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    parser.add_argument("--interval-seconds", type=float, default=0.25)
    parser.add_argument("--container", help="Fail immediately if this Docker container exits or restarts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.interval_seconds <= 0:
        parser.error("timeout and interval must be positive")
    deadline_ns = args.started_at_ns + int(args.timeout_seconds * 1_000_000_000)
    health_url = f"{args.base_url.rstrip('/')}/health"
    attempts = 0
    last_error = "not attempted"
    health = None
    while time_ns() < deadline_ns:
        attempts += 1
        if args.container:
            result = subprocess.run(["docker", "inspect", args.container], capture_output=True, text=True, timeout=10)
            if result.returncode:
                last_error = "Docker container inspection failed"
                break
            state = json.loads(result.stdout)[0]
            if state["State"]["Status"] in {"exited", "dead", "restarting"} or state.get("RestartCount", 0):
                last_error = f"container {args.container} stopped or restarted: {state['State']}"
                break
        try:
            with urlopen(health_url, timeout=5) as response:
                candidate = json.loads(response.read())
            if isinstance(candidate, dict) and candidate.get("ready") is True:
                health = candidate
                break
            last_error = f"health is not ready: {candidate!r}"
        except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        sleep(args.interval_seconds)
    finished_ns = time_ns()
    report = {
        "schema": "meddeid.container-startup.v1",
        "base_url": args.base_url,
        "ready_seconds": (finished_ns - args.started_at_ns) / 1_000_000_000,
        "attempts": attempts,
        "health": health,
        "passed": health is not None,
        "error": last_error if health is None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if health is None:
        raise RuntimeError(f"service did not become ready: {last_error}")
    print(json.dumps({"ready_seconds": report["ready_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
