"""Lightweight administrator inspection of a model bundle."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
from typing import Any, Callable, Mapping

from meddeid_core.age_policy import AgeGranularityPolicy, load_age_granularity_policy

from .bundle import ModelBundle, load_model_bundle
from .language import installed_language_profile_packages, resolve_language_profile
from .model_source import ResolvedModelSource, resolve_model_source


def package_versions() -> dict[str, str | None]:
    packages: dict[str, str | None] = {}
    for package in ("meddeid", "meddeid-core", "torch", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    packages.update(installed_language_profile_packages())
    return packages


def requested_revision_label(
    requested_revision: str | None,
    *,
    source_is_local: bool,
) -> str:
    if requested_revision is not None:
        return requested_revision
    return "local" if source_is_local else "latest"


def model_file_inventory(
    bundle: ModelBundle,
    *,
    source_is_local: bool,
    offline: bool,
) -> dict[str, Any]:
    root = bundle.manifest_path.parent.absolute()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "path": str(path.absolute()),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "loaded_from": "local-directory" if source_is_local else "huggingface-cache",
        "offline_enforced": offline,
        "root_path": str(root),
        "manifest_path": str(bundle.manifest_path.absolute()),
        "weights_path": str(bundle.checkpoint_path.absolute()),
        "weights_present": bundle.checkpoint_path.is_file(),
        "files": files,
    }


def inspect_model(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    on_status: Callable[[str], None] | None = None,
    language_profile: str | None = None,
    age_granularity_config: (
        AgeGranularityPolicy | str | Path | Mapping[str, Any] | None
    ) = None,
    min_recommended_date_shift_days: int = 366,
) -> dict[str, Any]:
    """Inspect bundle metadata and files without constructing an inference runtime."""

    if (
        isinstance(min_recommended_date_shift_days, bool)
        or not isinstance(min_recommended_date_shift_days, int)
        or min_recommended_date_shift_days <= 0
    ):
        raise ValueError("min_recommended_date_shift_days must be a positive integer")

    policy = load_age_granularity_policy(age_granularity_config)
    resolved: ResolvedModelSource = resolve_model_source(
        model,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        on_status=on_status,
    )
    if on_status is not None:
        on_status("validating model bundle")
    bundle = load_model_bundle(resolved.root / "bundle.json", validate_package=True)
    profiles = tuple(
        resolve_language_profile(item.profile_id) for item in bundle.postprocess.profiles
    )
    selected_profile = None
    if language_profile is not None:
        requested = language_profile.strip().replace("_", "-")
        matches = [profile for profile in profiles if profile.accepts_language(requested)]
        if len(matches) != 1:
            supported = ", ".join(profile.profile_id for profile in profiles)
            raise ValueError(
                f"language profile {requested!r} is not uniquely supported by this "
                f"model; expected one of: {supported}"
            )
        selected_profile = matches[0]
    elif len(profiles) == 1:
        selected_profile = profiles[0]

    if on_status is not None:
        on_status("inspection ready; inference runtime not checked")
    return {
        "model": {
            "source": resolved.source,
            "name": bundle.name,
            "version": bundle.model_version,
            "requested_revision": resolved.requested_revision_label,
            "resolved_revision": resolved.resolved_revision,
            "bundle_sha256": bundle.contract_hash(),
            "weights_format": bundle.weights_format,
        },
        "model_files": model_file_inventory(
            bundle,
            source_is_local=resolved.source_is_local,
            offline=resolved.offline,
        ),
        "contracts": {
            "language_profiles": [
                {"profile_id": profile.profile_id} for profile in profiles
            ],
            "default_language_profile": (
                selected_profile.profile_id if selected_profile is not None else None
            ),
            "max_length": bundle.inference.max_length,
            "overlap": bundle.inference.overlap,
            "entity_labels": len(bundle.entity_labels),
            "age_granularity_policy": policy.identity.to_dict(),
            "minimum_recommended_abs_date_shift_days": (
                min_recommended_date_shift_days
            ),
        },
        "runtime": {"checked": False},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
