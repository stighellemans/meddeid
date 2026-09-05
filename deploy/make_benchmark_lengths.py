#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any("text" not in row for row in rows):
        raise ValueError(f"{path} must contain JSONL rows with text")
    return rows


def repeat_to_length(text: str, length: int) -> str:
    if length < 1:
        raise ValueError("lengths must be positive")
    seed = text.strip()
    if not seed:
        raise ValueError("source text cannot be empty")
    repeated = (seed + " ") * ((length // (len(seed) + 1)) + 1)
    return repeated[:length]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic performance-only note lengths from public synthetic JSONL"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--length",
        type=int,
        action="append",
        dest="lengths",
        help="target character length; repeat for multiple rows",
    )
    args = parser.parse_args()
    lengths = args.lengths or [8000, 12000, 18000]
    rows = read_rows(args.source)
    generated = []
    for index, length in enumerate(lengths):
        source = rows[index % len(rows)]
        generated.append(
            {
                "document_id": f"performance-shape-{length}",
                "text": repeat_to_length(str(source["text"]), length),
                "metadata": source.get("metadata") or {},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in generated
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
