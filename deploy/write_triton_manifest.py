#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from triton_targets import (
    BUILD_MANIFEST_SCHEMA,
    CATALOG_PATH,
    get_target,
    load_catalog,
    target_spec_sha256,
    verify_host,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an immutable Triton build manifest"
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--triton-stack", required=True)
    parser.add_argument("--triton-server-version", required=True)
    parser.add_argument("--tensorrt-version", required=True)
    parser.add_argument("--builder-image", required=True)
    parser.add_argument("--target-catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--gpu-target", required=True)
    parser.add_argument("--gpu-name", required=True)
    parser.add_argument("--compute-capability", required=True)
    parser.add_argument("--driver-version", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--output-precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--min-shape", required=True)
    parser.add_argument("--opt-shape", required=True)
    parser.add_argument("--max-shape", required=True)
    parser.add_argument(
        "--throughput-dynamic-batching",
        choices=("true", "false"),
        required=True,
    )
    parser.add_argument("--throughput-queue-delay-microseconds", type=int)
    args = parser.parse_args()
    throughput_dynamic_batching = args.throughput_dynamic_batching == "true"
    if throughput_dynamic_batching and (
        args.throughput_queue_delay_microseconds is None
        or args.throughput_queue_delay_microseconds < 0
    ):
        parser.error("dynamic batching requires a non-negative throughput queue delay")

    try:
        target_spec = get_target(load_catalog(args.target_catalog), args.gpu_target)
        verify_host(
            target_spec,
            gpu_name=args.gpu_name,
            compute_capability=args.compute_capability,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    repository = args.repository.resolve()
    model_root = repository / args.model_name
    plan = model_root / args.model_version / "model.plan"
    config = model_root / "config.pbtxt"
    if not plan.is_file() or not config.is_file():
        raise SystemExit(
            "model.plan and config.pbtxt must exist before writing the manifest"
        )
    profile_configs = {
        path.stem: path for path in sorted((model_root / "configs").glob("*.pbtxt"))
    }
    if set(profile_configs) != {"latency", "throughput"}:
        raise SystemExit("latency and throughput Triton profile configs are required")

    payload = {
        "schema": BUILD_MANIFEST_SCHEMA,
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "bundle_sha256": args.bundle_sha256,
            "triton_name": args.model_name,
            "triton_version": args.model_version,
        },
        "runtime": {
            "triton_stack": args.triton_stack,
            "triton_server_version": args.triton_server_version,
            "tensorrt_version": args.tensorrt_version,
            "builder_image": args.builder_image,
        },
        "target": {
            "id": args.gpu_target,
            "display_name": target_spec["display_name"],
            "catalog_spec_sha256": target_spec_sha256(target_spec),
            "release_status": target_spec["release_status"],
            "image_repository": target_spec["image_repository"],
            "gpu_name": args.gpu_name,
            "compute_capability": args.compute_capability,
            "driver_version": args.driver_version,
            "precision": args.precision,
            "output_precision": args.output_precision,
        },
        "optimization_profile": {
            "min": args.min_shape,
            "opt": args.opt_shape,
            "max": args.max_shape,
        },
        "scheduler_profiles": {
            "latency": {"dynamic_batching": False},
            "throughput": {
                "dynamic_batching": throughput_dynamic_batching,
                **(
                    {
                        "max_queue_delay_microseconds": (
                            args.throughput_queue_delay_microseconds
                        )
                    }
                    if throughput_dynamic_batching
                    else {}
                ),
            },
        },
        "artifacts": {
            "config": {
                "path": str(config.relative_to(repository)),
                "sha256": sha256(config),
                "bytes": config.stat().st_size,
            },
            "config_profiles": {
                name: {
                    "path": str(path.relative_to(repository)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in profile_configs.items()
            },
            "plan": {
                "path": str(plan.relative_to(repository)),
                "sha256": sha256(plan),
                "bytes": plan.stat().st_size,
            },
        },
    }
    destination = repository / "build-manifest.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
