#!/usr/bin/env python3
"""Verify retained TensorRT candidate bytes before registry promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from meddeid.triton_artifact import load_catalog, select_plan, verify_repository


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_candidate(
    *,
    catalog_path: Path,
    repository: Path,
    evidence: Path,
    target: str,
    model_key: str,
    source_commit: str,
    minimum_documents: int,
) -> dict:
    catalog = load_catalog(catalog_path)
    plans = [
        item
        for item in catalog["plans"]
        if item.get("target") == target and item.get("model_key") == model_key
    ]
    if len(plans) != 1:
        raise ValueError(
            f"release catalog must contain one {target}/{model_key} plan"
        )
    plan = plans[0]
    selection = select_plan(
        catalog,
        hardware=plan["hardware"],
        model=plan["model"],
        revision=plan["revision"],
        language_profile=plan["language_profiles"][0],
    )
    if selection.target != target or selection.artifact != plan["artifact"]:
        raise ValueError("release selection does not resolve the requested plan")

    manifest = verify_repository(repository, selection)
    manifest_target = manifest["target"]
    artifact_repository = plan["artifact"].rsplit(":", 1)[0]
    if manifest_target.get("release_status") != "ready":
        raise ValueError("candidate manifest is not approved for release")
    if manifest_target.get("artifact_repository") != artifact_repository:
        raise ValueError("candidate manifest has the wrong artifact repository")

    parity = read_json(evidence / "parity-report.json")
    counts = {
        int(parity.get("documents", -1)),
        int(parity.get("fixture_documents", -1)),
        int(parity.get("checked_documents", -1)),
    }
    if counts != {minimum_documents}:
        raise ValueError(
            f"candidate parity did not cover exactly {minimum_documents} documents"
        )
    if not parity.get("passed") or not parity.get("complete"):
        raise ValueError("candidate parity did not complete under its acceptance policy")
    if parity.get("errors") != []:
        raise ValueError("candidate parity contains technical errors")
    if parity.get("model_identity_matches") is not True:
        raise ValueError("candidate parity model identity does not match")
    if parity.get("source_commit") != source_commit:
        raise ValueError("candidate parity source commit does not match")

    preflight = read_json(evidence / "fixture-contract-preflight.json")
    if preflight.get("passed") is not True or preflight.get("errors") != []:
        raise ValueError("candidate fixture contract preflight failed")
    if preflight.get("source_commit") != source_commit:
        raise ValueError("candidate preflight source commit does not match")
    if int(preflight.get("documents", -1)) != minimum_documents:
        raise ValueError("candidate preflight document count does not match")
    if not parity.get("fixture_sha256") or (
        parity.get("fixture_sha256") != preflight.get("fixture_sha256")
    ):
        raise ValueError("candidate parity and preflight fixtures do not match")

    return {
        "target": target,
        "model_key": model_key,
        "source_commit": source_commit,
        "documents": minimum_documents,
        "strict_passed": parity.get("strict_passed"),
        "semantic_differences": len(parity.get("semantic_differences") or []),
        "artifact": plan["artifact"],
        "plan_sha256": manifest["artifacts"]["plan"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--minimum-documents", type=int, default=300)
    args = parser.parse_args()
    try:
        result = verify_candidate(
            catalog_path=args.catalog,
            repository=args.repository,
            evidence=args.evidence,
            target=args.target,
            model_key=args.model_key,
            source_commit=args.source_commit,
            minimum_documents=args.minimum_documents,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"meddeid-triton-candidate: error: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
