"""Resolve the default Hub revision once for a matching plan and gateway."""

import re


def resolve_revision(
    model: str,
    revision: str,
    *,
    local_model: bool = False,
) -> str:
    revision = revision.strip()
    if local_model or re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(
            model, revision=revision or "main", expand=["sha"], timeout=10
        )
        if not re.fullmatch(r"[0-9a-f]{40}", info.sha or ""):
            raise ValueError("Hub did not return a commit")
        return info.sha
    except Exception:
        # Avoid echoing URLs, credentials, or internal network details from an
        # authentication/network exception into operational logs.
        raise ValueError(
            "Could not resolve the Hub revision. Check connectivity and model "
            "read access, or set MEDDEID_REVISION to an approved commit. "
            "Mounted local model directories need no Hub connection."
        ) from None
