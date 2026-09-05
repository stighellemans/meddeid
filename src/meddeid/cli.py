from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING

from .workflow import WorkflowError
from .workflow_cli import add_parsers as add_workflow_parsers
from .workflow_cli import dispatch as dispatch_workflow

if TYPE_CHECKING:
    from .api import Deidentifier


PUBLIC_MODELS = (
    {
        "model": "stighellemans/meddeid-dutch-synth",
        "language": "Dutch",
        "profiles": ("nl-BE",),
        "scope": "public synthetic-data baseline; validate on your institution",
    },
    {
        "model": "stighellemans/meddeid-english-synth",
        "language": "English",
        "profiles": ("en-GB", "en-US"),
        "scope": "public synthetic-data baseline; validate on your institution",
    },
)
PUBLIC_MODELS_BY_ID = {item["model"]: item for item in PUBLIC_MODELS}


def _add_model_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--model",
        help=(
            "Hub model ID or local bundle directory; required so MedDeID never "
            "silently chooses a language or validation context"
        ),
    )
    command.add_argument("--revision", help="Hub branch, tag, or immutable commit SHA")
    command.add_argument(
        "--language-profile",
        help="Default locale for a multi-profile model, for example en-GB or en-US",
    )
    command.add_argument(
        "--age-granularity-config",
        type=Path,
        help="Suite-wide declarative age-granularity JSON policy",
    )
    command.add_argument(
        "--min-recommended-date-shift-days",
        type=int,
        default=366,
        help="Warn when the absolute nonzero date shift is below this value (default: 366)",
    )
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


def _add_input_arguments(
    command: argparse.ArgumentParser,
    *,
    description: str,
) -> None:
    command.add_argument(
        "input_positional",
        nargs="?",
        metavar="input",
        type=Path,
        help=f"{description} (positional form)",
    )
    command.add_argument(
        "--input",
        dest="input_option",
        metavar="PATH",
        type=Path,
        help=f"{description} (explicit option form)",
    )


def _resolve_and_validate_input(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if args.command not in {"deidentify", "batch"}:
        return

    positional = args.input_positional
    explicit = args.input_option
    if positional is not None and explicit is not None:
        parser.exit(
            2,
            "meddeid: error: pass the input file either positionally or with "
            "`--input`, not both.\n",
        )
    input_path = explicit if explicit is not None else positional
    if input_path is None:
        parser.exit(
            2,
            "meddeid: error: no input file provided; pass it positionally or with "
            "`--input <path>`.\n",
        )
    if not input_path.exists():
        parser.exit(2, f"meddeid: error: input file not found: {input_path}\n")
    if not input_path.is_file():
        parser.exit(2, f"meddeid: error: input path is not a file: {input_path}\n")
    try:
        with input_path.open("rb"):
            pass
    except OSError as exc:
        parser.exit(2, f"meddeid: error: cannot read input file {input_path}: {exc}\n")

    args.input = input_path


def _print_model_guide(*, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"public_models": list(PUBLIC_MODELS)}, indent=2))
        return
    print("Public models (trained only on synthetic notes):")
    for item in PUBLIC_MODELS:
        profiles = ", ".join(item["profiles"])
        print(f"  {item['model']}")
        print(f"    {item['language']}; profiles: {profiles}")
        print(f"    {item['scope']}")
    print()
    print(
        "Next: use one of the model IDs above as the --model value in "
        "meddeid deidentify, batch, or model-info."
    )
    print()
    print(
        "For an institution-specific model, pass its Hub ID or local bundle directory."
    )
    print(
        "A model is not safe for operational use until it is validated on "
        "representative local data."
    )


def _require_model_selection(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.model:
        return
    parser.exit(
        2,
        "meddeid: error: no model selected.\n"
        "MedDeID will not silently assume a language or validation context. "
        "Run `meddeid models` to review the public baselines, then pass "
        "`--model <Hub-ID-or-local-directory>`.\n",
    )


def _validate_known_public_profile(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    model = PUBLIC_MODELS_BY_ID.get(args.model)
    if model is None:
        return
    profile = args.language_profile
    supported = tuple(model["profiles"])
    if profile is not None and profile.replace("_", "-") not in supported:
        parser.exit(
            2,
            f"meddeid: error: model {args.model!r} does not declare language profile "
            f"{profile!r}; choose one of: {', '.join(supported)}.\n",
        )
    if args.command == "deidentify" and len(supported) > 1 and profile is None:
        parser.exit(
            2,
            f"meddeid: error: model {args.model!r} supports multiple regional "
            f"profiles. Pass `--language-profile <{'|'.join(supported)}>`; "
            "MedDeID will not guess the region.\n",
        )


def _model_status_reporter(args: argparse.Namespace):
    started = time.monotonic()

    def report(message: str) -> None:
        elapsed = time.monotonic() - started
        print(f"[meddeid +{elapsed:5.1f}s] {message}", file=sys.stderr, flush=True)

    if args.quiet:
        return None
    status = report
    if status is not None:
        status(f"selected model: {args.model}")
        if args.language_profile:
            status(f"selected language profile: {args.language_profile}")
        public_model = PUBLIC_MODELS_BY_ID.get(args.model)
        if public_model is not None:
            status(
                "safety: this public model is trained on synthetic notes; "
                "validate it on representative local data"
            )
    return status


def _load_engine(args: argparse.Namespace) -> "Deidentifier":
    from .api import Deidentifier

    status = _model_status_reporter(args)
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
        language_profile=args.language_profile,
        age_granularity_config=args.age_granularity_config,
        min_recommended_date_shift_days=args.min_recommended_date_shift_days,
    )


def _run_batch(*args, **kwargs):
    from .batch import run_batch

    return run_batch(*args, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid")
    sub = parser.add_subparsers(dest="command", required=True)

    deidentify = sub.add_parser("deidentify", help="de-identify one UTF-8 text file locally")
    _add_input_arguments(deidentify, description="UTF-8 text input file")
    deidentify.add_argument("--output")
    deidentify.add_argument("--json", action="store_true")
    _add_model_arguments(deidentify)

    batch = sub.add_parser("batch", help="run canonical JSONL batch inference")
    _add_input_arguments(batch, description="canonical JSONL input file")
    batch.add_argument("--output", type=Path, required=True)
    policy = batch.add_mutually_exclusive_group()
    policy.add_argument("--resume", action="store_true")
    policy.add_argument("--overwrite", action="store_true")
    _add_model_arguments(batch)

    model_info = sub.add_parser(
        "model-info",
        help="inspect a model bundle, revision, profiles, files, and environment",
    )
    _add_model_arguments(model_info)
    model_info.add_argument(
        "--verify-runtime",
        action="store_true",
        help="also load the weights and verify the configured backend and device",
    )

    models = sub.add_parser(
        "models",
        help="show public model choices, regional profiles, and validation scope",
    )
    models.add_argument("--json", action="store_true")

    add_workflow_parsers(sub)

    args = parser.parse_args(argv)
    if args.command == "models":
        _print_model_guide(as_json=args.json)
        return 0
    if args.command in {"guide", "start", "status", "next", "doctor", "workflow"}:
        try:
            return dispatch_workflow(args)
        except WorkflowError as exc:
            print(f"meddeid: error: {exc}", file=sys.stderr)
            return exc.code
    _resolve_and_validate_input(parser, args)
    _require_model_selection(parser, args)
    _validate_known_public_profile(parser, args)
    if args.command == "model-info" and not args.verify_runtime:
        from .model_inspection import inspect_model

        try:
            info = inspect_model(
                args.model,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_files_only=args.offline,
                on_status=_model_status_reporter(args),
                language_profile=args.language_profile,
                age_granularity_config=args.age_granularity_config,
                min_recommended_date_shift_days=(
                    args.min_recommended_date_shift_days
                ),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            parser.exit(2, f"meddeid: error: {exc}\n")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

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

            manifest = _run_batch(
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

        result = engine(args.input.read_text(encoding="utf-8"))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"meddeid: error: {exc}\n")
    finally:
        engine.close()

    payload = (
        json.dumps(
            result.to_contract(),
            ensure_ascii=False,
            indent=2,
        )
        if args.json
        else result.deid_text
    )
    if not args.quiet:
        for warning in getattr(result, "warnings", []):
            print(
                f"[meddeid] warning {warning['code']}: {warning['message']}",
                file=sys.stderr,
            )
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0
