"""Resolve local and Hugging Face model bundles without importing a runtime."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Callable


StatusCallback = Callable[[str], None]
MODEL_ALLOW_PATTERNS = (
    "bundle.json",
    "config.json",
    "model.safetensors",
    "model.pt",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "README.md",
    "LICENSE*",
    "NOTICE*",
)
MODEL_METADATA_ALLOW_PATTERNS = tuple(
    pattern
    for pattern in MODEL_ALLOW_PATTERNS
    if pattern not in {"model.safetensors", "model.pt"}
)


def _human_file_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class ResolvedModelSource:
    root: Path
    source: str
    source_is_local: bool
    requested_revision: str | None
    resolved_revision: str | None
    offline: bool

    @property
    def requested_revision_label(self) -> str:
        if self.requested_revision is not None:
            return self.requested_revision
        return "local" if self.source_is_local else "latest"


def resolve_model_source(
    model: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    include_weights: bool = True,
    on_status: StatusCallback | None = None,
) -> ResolvedModelSource:
    """Resolve a model directory while preserving the existing cache semantics."""

    def emit(message: str) -> None:
        if on_status is not None:
            on_status(message)

    emit(f"resolving model source: {model!s}")
    root = Path(model).expanduser()
    source_is_local = root.exists()
    if not source_is_local:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "the meddeid installation is missing huggingface-hub"
            ) from exc
        if local_files_only:
            emit("loading model files from the local Hugging Face cache (--offline)")
        else:
            emit(
                "checking the Hugging Face cache and Hub; only missing or changed "
                "files will be downloaded"
            )
        try:
            snapshot_kwargs = {
                "repo_id": str(model),
                "revision": revision,
                "cache_dir": str(cache_dir) if cache_dir is not None else None,
                "token": token,
                "local_files_only": local_files_only,
                "allow_patterns": list(
                    MODEL_ALLOW_PATTERNS
                    if include_weights
                    else MODEL_METADATA_ALLOW_PATTERNS
                ),
            }
            download_progress = None
            silent_progress = None
            snapshot_parameters = signature(snapshot_download).parameters
            if "tqdm_class" in snapshot_parameters:
                from huggingface_hub.utils import tqdm as hub_tqdm

                class ModelDownloadTqdm(hub_tqdm):
                    # tqdm's global monitor otherwise survives model loading and
                    # can redraw stale Hub progress during interpreter exit.
                    monitor_interval = 0

                class SilentTqdm(ModelDownloadTqdm):
                    def __init__(self, *args, **kwargs):
                        kwargs["disable"] = True
                        super().__init__(*args, **kwargs)

                silent_progress = SilentTqdm
                download_progress = SilentTqdm if local_files_only else ModelDownloadTqdm
            if not local_files_only and "dry_run" in snapshot_parameters:
                plan = snapshot_download(
                    **snapshot_kwargs,
                    dry_run=True,
                    **(
                        {"tqdm_class": silent_progress}
                        if silent_progress is not None
                        else {}
                    ),
                )
                pending = [item for item in plan if item.will_download]
                if pending:
                    pending_size = sum(item.file_size for item in pending)
                    emit(
                        f"downloading {len(pending)} model file(s) "
                        f"({_human_file_size(pending_size)}); progress follows"
                    )
                else:
                    total_size = sum(item.file_size for item in plan)
                    emit(
                        f"using the cached model bundle "
                        f"({_human_file_size(total_size)}); Hub revision checked"
                    )
                    download_progress = silent_progress
            root = Path(
                snapshot_download(
                    **snapshot_kwargs,
                    **(
                        {"tqdm_class": download_progress}
                        if download_progress is not None
                        else {}
                    ),
                )
            )
        except Exception as exc:
            if local_files_only:
                recovery = (
                    f"model {model!s} is not available in the local cache; "
                    f"download it first with `hf download {model!s} "
                    "--local-dir ./meddeid-model`"
                )
            else:
                recovery = (
                    f"model {model!s} could not be downloaded; verify the repository "
                    "name and visibility, or authenticate with `hf auth login` if it "
                    "is private"
                )
            raise RuntimeError(recovery) from exc
        emit("model files ready in the Hugging Face cache")
    else:
        emit(f"using local model bundle: {root}")

    resolved_revision = (
        root.name if root.parent.name == "snapshots" and len(root.name) >= 7 else revision
    )
    return ResolvedModelSource(
        root=root,
        source=str(model),
        source_is_local=source_is_local,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        offline=local_files_only,
    )
