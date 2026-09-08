"""Resolve, download, and verify separately published Triton model plans."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .gpu_identity import gpu_key
from .triton_hardware import FAMILY_MODES, family_for_gpu, plan_supports_gpu, validate_family

CATALOG_SCHEMA = "meddeid.triton-release.v2"
BUILD_MANIFEST_SCHEMA = "meddeid.triton-build.v2"
ARTIFACT_TYPE = "application/vnd.meddeid.triton-model-repository.v1"
LAYER_MEDIA_TYPE = "application/vnd.meddeid.triton-model-repository.v1.tar+gzip"
OCI_MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json"
)
SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
LOCAL_BUILD_URL = (
    "https://stighellemans.github.io/meddeid/workflows/"
    "production-deployment/#advanced-build-a-plan-locally"
)


@dataclass(frozen=True)
class PlanSelection:
    suite_version: str
    meddeid_version: str
    hardware: str
    target: str
    model: str
    revision: str
    bundle_sha256: str
    language_profiles: tuple[str, ...]
    artifact: str
    family: str | None = None


@dataclass(frozen=True)
class NvidiaGpu:
    name: str
    compute_capability: str


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != CATALOG_SCHEMA:
        raise ValueError(f"{path} must use schema {CATALOG_SCHEMA}")
    if not isinstance(payload.get("plans"), list):
        raise TypeError(f"{path} must contain a plans list")
    for field in ("suite_version", "meddeid_version"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"{path} must define {field}")
    hardware_targets = payload.get("hardware_targets")
    if not isinstance(hardware_targets, dict) or not hardware_targets:
        raise ValueError(f"{path} must contain hardware_targets")
    known_gpu_names: set[str] = set()
    for hardware, target in hardware_targets.items():
        if not isinstance(hardware, str) or not hardware:
            raise ValueError(f"{path} contains an invalid hardware name")
        if not isinstance(target, dict):
            raise TypeError(f"{path} hardware target {hardware!r} must be an object")
        if hardware in FAMILY_MODES:
            validate_family({**target, "id": target.get("target"), "family": hardware})
            continue
        for field in ("target", "compute_capability", "gpu_names"):
            if field not in target:
                raise ValueError(
                    f"{path} hardware target {hardware!r} must define {field}"
                )
        if not re.fullmatch(r"[0-9]+\.[0-9]+", str(target["compute_capability"])):
            raise ValueError(
                f"{path} hardware target {hardware!r} has an invalid "
                "compute capability"
            )
        gpu_names = target["gpu_names"]
        if (
            not isinstance(gpu_names, list)
            or not gpu_names
            or not all(isinstance(name, str) and name.strip() for name in gpu_names)
        ):
            raise ValueError(
                f"{path} hardware target {hardware!r} must define GPU names"
            )
        normalized_names = {_normalize_gpu_name(name) for name in gpu_names}
        overlap = known_gpu_names & normalized_names
        if overlap:
            raise ValueError(f"{path} repeats GPU name {sorted(overlap)[0]!r}")
        known_gpu_names.update(normalized_names)
    seen: set[tuple[str, str, str]] = set()
    for index, plan in enumerate(payload["plans"]):
        if not isinstance(plan, dict):
            raise TypeError(f"{path} plans[{index}] must be an object")
        for field in (
            "hardware",
            "target",
            "model",
            "revision",
            "bundle_sha256",
            "artifact",
        ):
            if not isinstance(plan.get(field), str) or not plan[field]:
                raise ValueError(f"{path} plans[{index}] must define {field}")
        hardware_target = hardware_targets.get(plan["hardware"])
        if not isinstance(hardware_target, dict) or (
            hardware_target.get("target") != plan["target"]
        ):
            raise ValueError(
                f"{path} plans[{index}] does not match its hardware target"
            )
        key = (plan["hardware"], plan["model"], plan["revision"])
        if key in seen:
            raise ValueError(f"{path} contains a duplicate plan selection")
        seen.add(key)
    return payload


def _normalize_gpu_name(value: str) -> str:
    return gpu_key(value) if value.strip() else ""


def detect_nvidia_gpu() -> NvidiaGpu:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "cannot inspect the NVIDIA GPU because nvidia-smi is unavailable"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "unknown error"
        raise RuntimeError(f"nvidia-smi failed: {detail}") from exc

    rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError(
            "MedDeID must see exactly one NVIDIA GPU; select one with "
            "MEDDEID_GPU_DEVICE_ID"
        )
    gpu_name, compute_capability = (value.strip() for value in rows[0])
    if not gpu_name or not re.fullmatch(r"[0-9]+\.[0-9]+", compute_capability):
        raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout.strip()!r}")
    return NvidiaGpu(gpu_name, compute_capability)


def detect_hardware(catalog: Mapping[str, Any], gpu: NvidiaGpu) -> str:
    normalized_name = _normalize_gpu_name(gpu.name)
    name_matches = [
        (hardware, target)
        for hardware, target in catalog["hardware_targets"].items()
        if normalized_name
        in {_normalize_gpu_name(name) for name in target["gpu_names"]}
    ]
    if len(name_matches) > 1:
        raise ValueError(f"detected GPU name {gpu.name!r} matches multiple targets")
    if not name_matches:
        return gpu_key(gpu.name)
    hardware, target = name_matches[0]
    expected_capability = str(target["compute_capability"])
    if gpu.compute_capability != expected_capability:
        raise ValueError(
            f"detected {gpu.name} with compute capability "
            f"{gpu.compute_capability}; this release expects {expected_capability}"
        )
    return gpu_key(gpu.name)


def _catalog_hardware(catalog: Mapping[str, Any], hardware: str) -> str:
    """Read historical release names without making them a detection requirement."""
    if hardware in catalog["hardware_targets"]:
        return hardware
    matches = [
        key for key, target in catalog["hardware_targets"].items()
        if hardware in {gpu_key(name) for name in target["gpu_names"]}
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous catalog GPU name {hardware!r}")
    return matches[0] if matches else hardware


def select_plan(
    catalog: Mapping[str, Any],
    *,
    hardware: str,
    model: str,
    revision: str,
    language_profile: str,
    gpu: NvidiaGpu | None = None,
) -> PlanSelection:
    catalog_hardware = _catalog_hardware(catalog, hardware)
    family = None
    cc = gpu.compute_capability if gpu else (
        catalog["hardware_targets"].get(catalog_hardware, {}).get("compute_capability")
    )
    if cc:
        try:
            family = family_for_gpu(cc)
        except ValueError:
            pass
    family_matches = [
        item for item in catalog["plans"]
        if item.get("hardware") == family and item.get("model") == model
        and item.get("revision") == revision
    ]
    matches = family_matches or [
        item
        for item in catalog["plans"]
        if item.get("hardware") == catalog_hardware
        and item.get("model") == model
        and item.get("revision") == revision
    ]
    if not matches:
        raise ValueError(
            f"suite {catalog['suite_version']} has no published TensorRT plan for "
            f"{model}@{revision} on {hardware}. Build a local plan on this server: "
            f"{LOCAL_BUILD_URL}. The build stays on your device; "
            "nothing is published to the internet."
        )
    if len(matches) != 1:
        raise ValueError("the Triton release catalog contains duplicate plan entries")
    item = matches[0]
    profiles = item.get("language_profiles")
    if (
        not isinstance(profiles, list)
        or not profiles
        or not all(isinstance(profile, str) and profile for profile in profiles)
    ):
        raise ValueError("the selected plan must declare language_profiles")
    if language_profile not in profiles:
        raise ValueError(
            f"the selected {hardware} plan supports {', '.join(profiles)}, not "
            f"{language_profile}"
        )
    artifact = item.get("artifact")
    bundle_sha256 = item.get("bundle_sha256")
    target = item.get("target")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("the selected plan does not have a published artifact")
    if not isinstance(target, str) or not target:
        raise ValueError("the selected plan does not declare its GPU target")
    if not isinstance(bundle_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", bundle_sha256
    ):
        raise ValueError("the selected plan does not declare a valid bundle checksum")
    return PlanSelection(
        suite_version=str(catalog["suite_version"]),
        meddeid_version=str(catalog["meddeid_version"]),
        hardware=hardware,
        target=target,
        model=model,
        revision=revision,
        bundle_sha256=bundle_sha256,
        language_profiles=tuple(profiles),
        artifact=artifact,
        family=item["hardware"] if item["hardware"] in FAMILY_MODES else None,
    )


def request_selection(
    catalog: Mapping[str, Any],
    *,
    hardware: str,
    model: str,
    revision: str,
    language_profile: str,
) -> PlanSelection:
    """Describe a local plan request even when no official artifact exists."""
    targets = {
        str(item.get("target"))
        for item in catalog["plans"]
        if item.get("hardware") == hardware and item.get("target")
    }
    hardware_targets = catalog.get("hardware_targets") or {}
    hardware_target = hardware_targets.get(hardware)
    if isinstance(hardware_target, Mapping) and hardware_target.get("target"):
        targets.add(str(hardware_target["target"]))
    if len(targets) != 1:
        raise ValueError(
            f"suite catalog does not map hardware {hardware!r} to one target"
        )
    return PlanSelection(
        suite_version=str(catalog["suite_version"]),
        meddeid_version=str(catalog["meddeid_version"]),
        hardware=hardware,
        target=targets.pop(),
        model=model,
        revision=revision,
        bundle_sha256="",
        language_profiles=(language_profile,),
        artifact="",
    )


def _require_relative_file(repository: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact manifest paths must be non-empty strings")
    candidate = (repository / relative).resolve()
    try:
        candidate.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes the repository: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"artifact file is missing: {relative}")
    return candidate


def verify_repository(
    repository: Path, selection: PlanSelection | None = None
) -> dict[str, Any]:
    repository = repository.resolve()
    manifest_path = repository / "build-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{repository} does not contain build-manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != BUILD_MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path} must use schema {BUILD_MANIFEST_SCHEMA}")

    validate_family(payload.get("target") or {})
    if selection is not None:
        if selection.family and (payload.get("target") or {}).get("family") != selection.family:
            raise ValueError("Published family plan is missing its compatibility contract")
        expected = {
            ("release", "suite_version"): selection.suite_version,
            ("release", "meddeid_version"): selection.meddeid_version,
            ("model", "id"): selection.model,
            ("model", "revision"): selection.revision,
            ("target", "id"): selection.target,
        }
        for (section, field), value in expected.items():
            actual = (payload.get(section) or {}).get(field)
            if actual != value:
                raise ValueError(
                    f"plan manifest {section}.{field} is {actual!r}; expected {value!r}"
                )
        if selection.bundle_sha256:
            actual_bundle_sha256 = (payload.get("model") or {}).get("bundle_sha256")
            if actual_bundle_sha256 != selection.bundle_sha256:
                raise ValueError(
                    "plan manifest model.bundle_sha256 is "
                    f"{actual_bundle_sha256!r}; expected {selection.bundle_sha256!r}"
                )
        declared_profiles = set(
            (payload.get("model") or {}).get("language_profiles") or []
        )
        if not set(selection.language_profiles).issubset(declared_profiles):
            raise ValueError(
                "plan manifest does not contain the catalog language profiles"
            )

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("plan manifest must contain artifacts")
    descriptions: list[Mapping[str, Any]] = []
    for key in ("config", "plan"):
        value = artifacts.get(key)
        if not isinstance(value, dict):
            raise TypeError(f"plan manifest artifacts.{key} is missing")
        descriptions.append(value)
    profiles = artifacts.get("config_profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"latency", "throughput"}:
        raise ValueError("plan manifest must contain latency and throughput configs")
    descriptions.extend(profiles.values())

    for description in descriptions:
        path = _require_relative_file(repository, description.get("path"))
        expected_digest = description.get("sha256")
        if isinstance(expected_digest, str) and not expected_digest.startswith(
            "sha256:"
        ):
            expected_digest = f"sha256:{expected_digest}"
        if not isinstance(expected_digest, str) or not SHA256.fullmatch(
            expected_digest
        ):
            raise ValueError(f"invalid SHA-256 for {path.name}")
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"checksum mismatch for {path.name}: expected {expected_digest}, "
                f"found {actual_digest}"
            )
        if description.get("bytes") != path.stat().st_size:
            raise ValueError(f"size mismatch for {path.name}")
    return payload


def pack_repository(repository: Path, output: Path) -> str:
    verify_repository(repository)
    repository = repository.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(repository.rglob("*")):
            if path.is_symlink():
                raise ValueError(
                    f"symbolic links are not allowed in plan artifacts: {path}"
                )
            if not path.is_file():
                continue
            info = archive.gettarinfo(str(path), str(path.relative_to(repository)))
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    return _sha256_file(output)


def _parse_authenticate(value: str) -> tuple[str, dict[str, str]]:
    scheme, _, raw = value.partition(" ")
    if scheme.lower() != "bearer" or not raw:
        raise ValueError("registry did not offer Bearer authentication")
    fields = {
        match.group(1): match.group(2) for match in re.finditer(r'(\w+)="([^"]*)"', raw)
    }
    if "realm" not in fields:
        raise ValueError("registry authentication challenge has no realm")
    return scheme, fields


class RegistryClient:
    def __init__(self, registry: str) -> None:
        self.registry = registry
        self.token: str | None = None
        self.username = os.environ.get("MEDDEID_REGISTRY_USERNAME")
        self.password = os.environ.get("MEDDEID_REGISTRY_TOKEN") or os.environ.get(
            "GHCR_TOKEN"
        )

    def _open(self, url: str, *, accept: str | None = None) -> tuple[bytes, Any]:
        headers = {"User-Agent": "meddeid-triton-plan/1"}
        if accept:
            headers["Accept"] = accept
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as exc:
            if exc.code != 401 or self.token:
                raise RuntimeError(
                    f"registry request failed: {url}: HTTP {exc.code}"
                ) from exc
            _, challenge = _parse_authenticate(exc.headers.get("WWW-Authenticate", ""))
            query = {
                key: challenge[key]
                for key in ("service", "scope")
                if challenge.get(key)
            }
            token_url = f"{challenge['realm']}?{urllib.parse.urlencode(query)}"
            token_headers = {"User-Agent": "meddeid-triton-plan/1"}
            if self.username and self.password:
                encoded = base64.b64encode(
                    f"{self.username}:{self.password}".encode()
                ).decode("ascii")
                token_headers["Authorization"] = f"Basic {encoded}"
            token_request = urllib.request.Request(token_url, headers=token_headers)
            try:
                with urllib.request.urlopen(token_request, timeout=30) as response:
                    token_payload = json.loads(response.read())
            except urllib.error.HTTPError as token_exc:
                raise RuntimeError(
                    f"registry authentication failed: HTTP {token_exc.code}"
                ) from token_exc
            self.token = token_payload.get("token") or token_payload.get("access_token")
            if not self.token:
                raise RuntimeError("registry authentication returned no token")
            return self._open(url, accept=accept)

    def pull(self, reference: str) -> tuple[bytes, str]:
        registry, repository, identifier = parse_oci_reference(reference)
        if registry != self.registry:
            raise ValueError("registry client/reference mismatch")
        manifest_url = f"https://{registry}/v2/{repository}/manifests/{identifier}"
        manifest_bytes, _ = self._open(manifest_url, accept=OCI_MANIFEST_TYPES)
        manifest_digest = _sha256_bytes(manifest_bytes)
        if identifier.startswith("sha256:") and manifest_digest != identifier:
            raise ValueError(
                f"artifact manifest digest mismatch: expected {identifier}, "
                f"found {manifest_digest}"
            )
        manifest = json.loads(manifest_bytes)
        artifact_type = manifest.get("artifactType")
        if artifact_type != ARTIFACT_TYPE:
            raise ValueError(f"unexpected artifact type: {artifact_type}")
        layers = [
            layer
            for layer in manifest.get("layers", [])
            if layer.get("mediaType") == LAYER_MEDIA_TYPE
        ]
        if len(layers) != 1:
            raise ValueError(
                "Triton plan artifact must contain exactly one repository layer"
            )
        layer = layers[0]
        digest = layer.get("digest")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise ValueError("Triton plan layer has an invalid digest")
        blob_url = f"https://{registry}/v2/{repository}/blobs/{digest}"
        blob, _ = self._open(blob_url, accept=LAYER_MEDIA_TYPE)
        if _sha256_bytes(blob) != digest:
            raise ValueError(
                "downloaded Triton plan layer failed checksum verification"
            )
        return blob, manifest_digest


def parse_oci_reference(reference: str) -> tuple[str, str, str]:
    if "://" in reference or reference.startswith("/"):
        raise ValueError("artifact must be an OCI reference without a URL scheme")
    registry, separator, remainder = reference.partition("/")
    if not separator or not registry or not remainder:
        raise ValueError(f"invalid OCI artifact reference: {reference}")
    if "@" in remainder:
        repository, identifier = remainder.rsplit("@", 1)
    else:
        slash = remainder.rfind("/")
        colon = remainder.rfind(":")
        if colon <= slash:
            raise ValueError("OCI artifact reference must include a tag or digest")
        repository, identifier = remainder[:colon], remainder[colon + 1 :]
    if not repository or not identifier:
        raise ValueError(f"invalid OCI artifact reference: {reference}")
    return registry, repository, identifier


def _safe_extract(blob: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError("plan archives may not contain links or device files")
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise ValueError("plan archive contains an unsafe path") from exc
        archive.extractall(destination)


def fetch_plan(
    selection: PlanSelection, output: Path, artifact: str | None = None
) -> str:
    output = output.resolve()
    if output == Path(output.anchor) or (
        len(output.parts) < 3 and output != Path("/models")
    ):
        raise ValueError(f"refusing unsafe plan output directory: {output}")
    if output.exists() and any(output.iterdir()):
        try:
            verify_repository(output, selection)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            if not (output / ".meddeid-managed").is_file():
                raise ValueError(
                    f"{output} contains a different or invalid local repository; "
                    "move it or select matching model settings"
                )
        else:
            return "cached"

    reference = artifact or selection.artifact
    registry, _, _ = parse_oci_reference(reference)
    blob, manifest_digest = RegistryClient(registry).pull(reference)
    output.mkdir(parents=True, exist_ok=True)
    previous_entries = list(output.iterdir())
    # /models is a writable bind mount in an otherwise read-only container.
    # Stage inside it and preserve the mount point during installation.
    with tempfile.TemporaryDirectory(prefix=".meddeid-plan-", dir=output) as raw:
        candidate = Path(raw) / "repository"
        candidate.mkdir()
        _safe_extract(blob, candidate)
        verify_repository(candidate, selection)
        (candidate / ".meddeid-managed").write_text(
            json.dumps(
                {"artifact": reference, "manifest_digest": manifest_digest},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        for entry in previous_entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        shutil.copytree(candidate, output, dirs_exist_ok=True)
    return manifest_digest


def prepare_plan(
    catalog: Mapping[str, Any],
    *,
    hardware: str | None,
    model: str,
    revision: str,
    language_profile: str,
    output: Path,
    artifact: str | None = None,
    gpu: NvidiaGpu | None = None,
) -> str:
    if output.is_dir() and any(output.iterdir()):
        try:
            manifest = verify_repository(output)
            for field in ("suite_version", "meddeid_version"):
                if (manifest.get("release") or {}).get(field) != catalog[field]:
                    raise ValueError(f"plan {field} does not match this release")
            for field, expected in (catalog.get("runtime") or {}).items():
                if (manifest.get("runtime") or {}).get(field) != expected:
                    raise ValueError(f"plan {field} does not match the runtime stack")
            manifest_model = manifest.get("model") or {}
            expected_model = {
                "id": model,
                "revision": revision,
            }
            for field, expected in expected_model.items():
                if manifest_model.get(field) != expected:
                    raise ValueError(
                        f"local plan model.{field} is {manifest_model.get(field)!r}; "
                        f"expected {expected!r}"
                    )
            if language_profile not in set(
                manifest_model.get("language_profiles") or []
            ):
                raise ValueError(
                    f"local plan does not support language profile {language_profile}"
                )
            manifest_target = manifest.get("target") or {}
            if gpu is not None:
                if (
                    not plan_supports_gpu(manifest_target, gpu.name, gpu.compute_capability)
                ):
                    raise ValueError(
                        f"local plan was built for "
                        f"{manifest_target.get('gpu_name')!r} "
                        f"({manifest_target.get('compute_capability')}); detected "
                        f"{gpu.name!r} ({gpu.compute_capability})"
                    )
            elif hardware is not None:
                requested = request_selection(
                    catalog,
                    hardware=hardware,
                    model=model,
                    revision=revision,
                    language_profile=language_profile,
                )
                if manifest_target.get("id") != requested.target:
                    raise ValueError(
                        f"local plan target is {manifest_target.get('id')!r}; "
                        f"expected {requested.target!r}"
                    )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if not (output / ".meddeid-managed").is_file():
                raise ValueError(
                    f"Local plan cannot be used: {exc}. "
                    f"Build a matching plan on this server: {LOCAL_BUILD_URL}"
                ) from exc
        else:
            return "local"
    if hardware is None:
        if gpu is None:
            gpu = detect_nvidia_gpu()
        hardware = detect_hardware(catalog, gpu)
    try:
        published = select_plan(
            catalog,
            hardware=hardware,
            model=model,
            revision=revision,
            language_profile=language_profile,
            gpu=gpu,
        )
    except ValueError as exc:
        if gpu is not None and "has no published TensorRT plan" in str(exc):
            raise ValueError(
                f"detected {gpu.name} (compute capability "
                f"{gpu.compute_capability}), but {exc}"
            ) from exc
        raise
    result = fetch_plan(published, output, artifact)
    manifest = verify_repository(output, published)
    for field, expected in (catalog.get("runtime") or {}).items():
        if (manifest.get("runtime") or {}).get(field) != expected:
            raise ValueError(f"Downloaded plan {field} does not match the runtime stack")
    if gpu and not plan_supports_gpu(manifest["target"], gpu.name, gpu.compute_capability):
        raise ValueError("Downloaded plan does not support the detected GPU")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage MedDeID Triton plan artifacts")
    commands = parser.add_subparsers(dest="command", required=True)

    for name in ("resolve", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("--catalog", type=Path, required=True)
        command.add_argument("--hardware")
        command.add_argument("--model", required=True)
        command.add_argument("--revision", default="")
        command.add_argument("--language-profile", required=True)
        if name == "fetch":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--artifact")

    revision_check = commands.add_parser("check-revision")
    revision_check.add_argument("--model", required=True)
    revision_check.add_argument("--revision", default="")
    revision_check.add_argument("--local-model", action="store_true")

    verify = commands.add_parser("verify")
    verify.add_argument("repository", type=Path)
    package = commands.add_parser("pack")
    package.add_argument("repository", type=Path)
    package.add_argument("output", type=Path)

    args = parser.parse_args()
    try:
        from .triton_revision import resolve_revision

        if args.command == "check-revision":
            print(resolve_revision(args.model, args.revision, local_model=args.local_model))
            return
        if args.command in {"resolve", "fetch"}:
            catalog = load_catalog(args.catalog)
            args.revision = resolve_revision(
                args.model, args.revision,
                local_model=os.environ.get("MEDDEID_LOCAL_BUNDLE", "") == "true",
            )
            if args.command == "resolve":
                hardware = args.hardware
                gpu = None
                if hardware is None:
                    gpu = detect_nvidia_gpu()
                    hardware = detect_hardware(catalog, gpu)
                selection = select_plan(
                    catalog,
                    hardware=hardware,
                    model=args.model,
                    revision=args.revision,
                    language_profile=args.language_profile,
                    gpu=gpu,
                )
                print(selection.artifact)
            else:
                gpu = None if args.hardware else detect_nvidia_gpu()
                result = prepare_plan(
                    catalog,
                    hardware=args.hardware,
                    model=args.model,
                    revision=args.revision,
                    language_profile=args.language_profile,
                    output=args.output,
                    artifact=args.artifact,
                    gpu=gpu,
                )
                print(f"Triton plan ready at {args.output.resolve()} ({result})")
        elif args.command == "verify":
            verify_repository(args.repository)
            print(f"Verified Triton plan repository: {args.repository.resolve()}")
        else:
            digest = pack_repository(args.repository, args.output)
            print(f"{args.output.resolve()} {digest}")
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"meddeid-triton-plan: error: {exc}\n")


if __name__ == "__main__":
    main()
