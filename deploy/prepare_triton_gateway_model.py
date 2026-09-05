#!/usr/bin/env python3
"""Create the metadata/tokenizer projection used by the TensorRT gateway."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from meddeid.bundle import load_model_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def project_gateway_model(
    source: Path,
    output: Path,
    *,
    model_id: str,
    revision: str,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.absolute()
    if output.exists():
        raise ValueError(f"gateway model output already exists: {output}")

    bundle = load_model_bundle(source / "bundle.json", validate_package=True)
    contract_hash = bundle.contract_hash()
    if contract_hash != expected_bundle_sha256:
        raise ValueError(
            f"bundle hash mismatch: expected {expected_bundle_sha256}, found {contract_hash}"
        )
    checkpoint = bundle.checkpoint_path
    checkpoint_relative = checkpoint.relative_to(source)
    excluded_weights = {
        "path": checkpoint_relative.as_posix(),
        "bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
    }

    def ignore(directory: str, names: list[str]) -> set[str]:
        current = Path(directory)
        ignored = {".cache"} & set(names)
        for name in names:
            candidate = current / name
            if candidate == checkpoint or candidate.suffix.lower() in {
                ".safetensors",
                ".pt",
                ".onnx",
            }:
                ignored.add(name)
        return ignored

    shutil.copytree(source, output, symlinks=False, ignore=ignore)
    projected = load_model_bundle(
        output / "bundle.json",
        validate_package=True,
        require_weights=False,
    )
    if projected.contract_hash() != contract_hash:
        raise RuntimeError("gateway projection changed the model contract")
    if projected.checkpoint_path.exists():
        raise RuntimeError("gateway projection unexpectedly contains model weights")
    if any(
        path.suffix.lower() in {".safetensors", ".pt", ".onnx"}
        for path in output.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError("gateway projection contains an undeclared weights artifact")

    manifest = {
        "schema": "meddeid.triton-gateway-model.v1",
        "source": {
            "model_id": model_id,
            "revision": revision,
            "bundle_sha256": contract_hash,
        },
        "excluded_weights": excluded_weights,
        "included_files": _inventory(output),
    }
    manifest_path = output / "gateway-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip local inference weights from a validated Triton gateway bundle"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    manifest = project_gateway_model(
        args.source,
        args.output,
        model_id=args.model_id,
        revision=args.revision,
        expected_bundle_sha256=args.bundle_sha256,
    )
    print(json.dumps(manifest["source"], sort_keys=True))


if __name__ == "__main__":
    main()
