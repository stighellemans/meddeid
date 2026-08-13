from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .api import Deidentifier
from .batch import run_batch


DEFAULT_MODEL = "stighellemans/meddeid-dutch-synth"


def _add_model_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--model", default=DEFAULT_MODEL)
    command.add_argument("--revision", help="Hub branch, tag, or immutable commit SHA")
    command.add_argument("--cache-dir")
    command.add_argument(
        "--offline",
        action="store_true",
        help="Use only an already downloaded local model or Hub cache snapshot",
    )
    command.add_argument("--backend", choices=("torch", "triton"), default="torch")
    command.add_argument("--device", choices=("cpu", "mps", "cuda"))
    command.add_argument("--triton-url", help="Triton V2 HTTP URL for --backend triton")
    command.add_argument("--triton-timeout", type=float, default=30.0)
    command.add_argument(
        "--window-batch-size",
        type=int,
        help="Maximum model windows per inference call (default: 32 torch, 16 triton)",
    )
    command.add_argument("--quiet", action="store_true", help="Hide loading and progress status")


def _load_engine(args: argparse.Namespace) -> Deidentifier:
    status = None if args.quiet else lambda message: print(f"[meddeid] {message}", file=sys.stderr)
    return Deidentifier.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
        device=args.device,
        backend=args.backend,
        triton_url=args.triton_url,
        triton_timeout_seconds=args.triton_timeout,
        max_windows_per_batch=args.window_batch_size,
        on_status=status,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid")
    sub = parser.add_subparsers(dest="command", required=True)

    deidentify = sub.add_parser("deidentify", help="de-identify one UTF-8 text file locally")
    deidentify.add_argument("input")
    deidentify.add_argument("--output")
    deidentify.add_argument("--json", action="store_true")
    _add_model_arguments(deidentify)

    batch = sub.add_parser("batch", help="run canonical JSONL batch inference")
    batch.add_argument("input", type=Path)
    batch.add_argument("--output", type=Path, required=True)
    policy = batch.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    _add_model_arguments(batch)

    model_info = sub.add_parser(
        "model-info",
        help="load a model and report its bundle, runtime, revision, and environment",
    )
    _add_model_arguments(model_info)

    args = parser.parse_args(argv)
    try:
        engine = _load_engine(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"meddeid: error: {exc}\n")
    try:
        if args.command == "model-info":
            print(json.dumps(engine.model_info(), ensure_ascii=False, indent=2))
            return 0

        if args.command == "batch":
            progress = None
            if not args.quiet:
                def progress(event: dict) -> None:
                    completed = int(event["completed"])
                    total = int(event["total"])
                    elapsed = float(event["elapsed_seconds"])
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"\r[meddeid] {completed}/{total} documents ({rate:.2f}/s)",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

            manifest = run_batch(
                engine,
                args.input,
                args.output,
                resume=args.resume,
                overwrite=args.overwrite,
                progress=progress,
            )
            if progress is not None:
                print(file=sys.stderr)
            print(json.dumps(manifest["counts"], indent=2))
            return 0

        result = engine(Path(args.input).read_text(encoding="utf-8"))
    finally:
        engine.close()

    payload = (
        json.dumps(
            {"text": result.text, "deid_text": result.deid_text, "spans": result.spans},
            ensure_ascii=False,
            indent=2,
        )
        if args.json
        else result.deid_text
    )
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0
