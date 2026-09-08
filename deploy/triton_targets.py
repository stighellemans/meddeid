#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent / "triton" / "targets.json"
CATALOG_SCHEMA = "meddeid.triton-target-catalog.v2"
BUILD_MANIFEST_SCHEMA = "meddeid.triton-build.v2"
RELEASE_STATUSES = frozenset({"ready", "on-request"})


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"{path} must use schema {CATALOG_SCHEMA}")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path} must contain a non-empty targets list")

    required = {
        "id",
        "hardware",
        "display_name",
        "compute_capability",
        "gpu_names",
        "runner_label",
        "artifact_repository",
        "release_status",
    }
    seen: set[str] = set()
    hardware_names: set[str] = set()
    ready = 0
    known_gpu_names: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"targets[{index}] must be an object")
        missing = required - set(target)
        if missing:
            raise ValueError(f"targets[{index}] is missing {sorted(missing)}")
        target_id = str(target["id"])
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", target_id):
            raise ValueError(f"invalid target id: {target_id!r}")
        if target_id in seen:
            raise ValueError(f"duplicate target id: {target_id}")
        seen.add(target_id)
        hardware = str(target["hardware"])
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", hardware):
            raise ValueError(f"invalid hardware name: {hardware!r}")
        if hardware in hardware_names:
            raise ValueError(f"duplicate hardware name: {hardware}")
        hardware_names.add(hardware)
        if not re.fullmatch(r"[0-9]+\.[0-9]+", str(target["compute_capability"])):
            raise ValueError(f"invalid compute capability for {target_id}")
        gpu_names = target["gpu_names"]
        if (
            not isinstance(gpu_names, list)
            or not gpu_names
            or not all(isinstance(name, str) and name.strip() for name in gpu_names)
        ):
            raise ValueError(f"invalid GPU names for {target_id}")
        normalized_names = {_normalize_gpu_name(name) for name in gpu_names}
        overlap = known_gpu_names & normalized_names
        if overlap:
            raise ValueError(f"duplicate GPU name: {sorted(overlap)[0]}")
        known_gpu_names.update(normalized_names)
        if target["runner_label"] != target_id:
            raise ValueError(f"runner_label for {target_id} must equal its target id")
        if not str(target["artifact_repository"]).endswith(f"-{target_id}"):
            raise ValueError(
                f"artifact_repository for {target_id} must end with -{target_id}"
            )
        if target["release_status"] not in RELEASE_STATUSES:
            raise ValueError(f"invalid release_status for {target_id}")
        ready += target["release_status"] == "ready"

    default_target = payload.get("default_target")
    if default_target not in seen:
        raise ValueError("default_target must name a catalog target")
    if get_target(payload, str(default_target))["release_status"] != "ready":
        raise ValueError("default_target must be ready")
    if ready < 1:
        raise ValueError("the catalog must contain at least one ready target")
    return payload


def get_target(catalog: dict[str, Any], target_id: str) -> dict[str, Any]:
    for target in catalog["targets"]:
        if target["id"] == target_id:
            return target
    available = ", ".join(target["id"] for target in catalog["targets"])
    raise ValueError(f"unknown TensorRT target {target_id!r}; available: {available}")


def get_hardware_target(catalog: dict[str, Any], hardware: str) -> dict[str, Any]:
    for target in catalog["targets"]:
        if target["hardware"] == hardware:
            return target
    available = ", ".join(target["hardware"] for target in catalog["targets"])
    raise ValueError(f"unknown TensorRT hardware {hardware!r}; available: {available}")


def target_spec_sha256(target: dict[str, Any]) -> str:
    return _canonical_sha256(target)


def _normalize_gpu_name(value: str) -> str:
    return gpu_key(value)


def gpu_key(name: str) -> str:
    # Same normalization as meddeid.gpu_identity; this catalog tool also runs
    # in release CI before the MedDeID package/dependencies are installed.
    key = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not key:
        raise ValueError("NVIDIA GPU name is empty or invalid")
    return key


def local_target(gpu_name: str, compute_capability: str) -> dict[str, Any]:
    slug = gpu_key(gpu_name)
    if not slug or not re.fullmatch(r"[0-9]+\.[0-9]+", compute_capability):
        raise ValueError("cannot derive a local target from the detected GPU")
    target_id = slug
    return {
        "id": target_id,
        "hardware": target_id,
        "display_name": gpu_name.strip(),
        "compute_capability": compute_capability,
        "gpu_names": [gpu_name.strip()],
        "runner_label": target_id,
        "artifact_repository": "local",
        "release_status": "local",
    }


def detect_target(
    catalog: dict[str, Any],
    *,
    gpu_name: str,
    compute_capability: str,
    allow_local: bool = False,
) -> dict[str, Any]:
    # Local builds need no pre-registered target. Old release IDs remain valid
    # for published artifacts and CI runners, independently of name recognition.
    if allow_local:
        return local_target(gpu_name, compute_capability)
    normalized_name = _normalize_gpu_name(gpu_name)
    name_matches = [
        target
        for target in catalog["targets"]
        if normalized_name
        in {_normalize_gpu_name(name) for name in target["gpu_names"]}
    ]
    if len(name_matches) > 1:
        raise ValueError(f"detected GPU name {gpu_name!r} matches multiple targets")
    if name_matches:
        target = name_matches[0]
        expected_capability = str(target["compute_capability"])
        if compute_capability != expected_capability:
            raise ValueError(
                f"detected GPU {gpu_name!r} reports compute capability "
                f"{compute_capability}; expected {expected_capability}"
            )
        return target
    if allow_local:
        return local_target(gpu_name, compute_capability)
    raise ValueError(
        f"detected GPU {gpu_name!r} (compute capability {compute_capability}) "
        "is not a declared MedDeID target"
    )


def verify_host(
    target: dict[str, Any], *, gpu_name: str, compute_capability: str
) -> None:
    expected_capability = str(target["compute_capability"])
    if compute_capability != expected_capability:
        raise ValueError(
            f"target {target['id']} requires compute capability "
            f"{expected_capability}; GPU {gpu_name!r} reports {compute_capability}"
        )
    normalized_names = {_normalize_gpu_name(name) for name in target["gpu_names"]}
    if _normalize_gpu_name(gpu_name) not in normalized_names:
        raise ValueError(
            f"target {target['id']} does not match detected GPU name {gpu_name!r}"
        )


def artifact_tag(
    target: dict[str, Any],
    *,
    version: str,
    triton_stack: str,
    precision: str,
    model_key: str | None = None,
) -> str:
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", version):
        raise ValueError(f"invalid MedDeID version for image tag: {version!r}")
    if not re.fullmatch(r"[0-9]+\.[0-9]+", triton_stack):
        raise ValueError(f"invalid Triton stack: {triton_stack!r}")
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"invalid TensorRT precision: {precision!r}")
    if model_key is not None and not re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", model_key
    ):
        raise ValueError(f"invalid model key for artifact tag: {model_key!r}")
    model_suffix = f"-{model_key}" if model_key else ""
    return (
        f"{target['artifact_repository']}:"
        f"{version}-trt{triton_stack}-{precision}{model_suffix}"
    )


def verify_manifest(target: dict[str, Any], manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != BUILD_MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path} must use schema {BUILD_MANIFEST_SCHEMA}")
    recorded = payload.get("target")
    if not isinstance(recorded, dict):
        raise TypeError(f"{manifest_path} target must be an object")
    checks = {
        "id": target["id"],
        "display_name": target["display_name"],
        "catalog_spec_sha256": target_spec_sha256(target),
        "release_status": target["release_status"],
        "artifact_repository": target["artifact_repository"],
        "compute_capability": target["compute_capability"],
    }
    for field, expected in checks.items():
        actual = recorded.get(field)
        if actual != expected:
            raise ValueError(
                f"manifest target.{field} is {actual!r}; expected {expected!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and validate MedDeID TensorRT publication targets"
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)

    list_command = commands.add_parser("list")
    list_command.add_argument("--status", choices=sorted(RELEASE_STATUSES))
    list_command.add_argument("--json", action="store_true")

    show_command = commands.add_parser("show")
    show_command.add_argument("target")
    show_command.add_argument("--field")

    verify_command = commands.add_parser("verify-host")
    verify_command.add_argument("target")
    verify_command.add_argument("--gpu-name", required=True)
    verify_command.add_argument("--compute-capability", required=True)

    hardware_command = commands.add_parser("resolve-hardware")
    hardware_command.add_argument("hardware")
    hardware_command.add_argument("--field", default="id")

    detect_command = commands.add_parser("detect-host")
    detect_command.add_argument("--gpu-name", required=True)
    detect_command.add_argument("--compute-capability", required=True)
    detect_command.add_argument("--allow-local", action="store_true")
    detect_command.add_argument("--field", default="id")

    tag_command = commands.add_parser("artifact-tag")
    tag_command.add_argument("target")
    tag_command.add_argument("--version", required=True)
    tag_command.add_argument("--triton-stack", required=True)
    tag_command.add_argument("--precision", required=True)
    tag_command.add_argument("--model-key")

    manifest_command = commands.add_parser("verify-manifest")
    manifest_command.add_argument("target")
    manifest_command.add_argument("manifest", type=Path)

    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        if args.command == "list":
            targets = [
                target
                for target in catalog["targets"]
                if args.status is None or target["release_status"] == args.status
            ]
            if args.json:
                print(json.dumps(targets, indent=2, sort_keys=True))
            else:
                for target in targets:
                    print(
                        f"{target['id']}\t{target['release_status']}\t"
                        f"{target['display_name']}\tSM {target['compute_capability']}"
                    )
            return

        if args.command == "resolve-hardware":
            target = get_hardware_target(catalog, args.hardware)
            if args.field not in target:
                raise ValueError(
                    f"target for {args.hardware!r} has no field {args.field!r}"
                )
            print(target[args.field])
            return

        if args.command == "detect-host":
            target = detect_target(
                catalog,
                gpu_name=args.gpu_name,
                compute_capability=args.compute_capability,
                allow_local=args.allow_local,
            )
            if args.field not in target:
                raise ValueError(f"detected target has no field {args.field!r}")
            print(target[args.field])
            return

        target = get_target(catalog, args.target)
        if args.command == "show":
            if args.field:
                if args.field not in target:
                    raise ValueError(
                        f"target {args.target!r} has no field {args.field!r}"
                    )
                print(target[args.field])
            else:
                print(json.dumps(target, indent=2, sort_keys=True))
        elif args.command == "verify-host":
            verify_host(
                target,
                gpu_name=args.gpu_name,
                compute_capability=args.compute_capability,
            )
            print(
                json.dumps(
                    {
                        "target": target["id"],
                        "release_status": target["release_status"],
                        "gpu_name": args.gpu_name,
                        "compute_capability": args.compute_capability,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "artifact-tag":
            print(
                artifact_tag(
                    target,
                    version=args.version,
                    triton_stack=args.triton_stack,
                    precision=args.precision,
                    model_key=args.model_key,
                )
            )
        elif args.command == "verify-manifest":
            verify_manifest(target, args.manifest)
            print(f"Verified target catalog binding: {target['id']}")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"triton-targets: error: {exc}\n")


if __name__ == "__main__":
    main()
