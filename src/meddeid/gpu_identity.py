"""GPU names for display, and deterministic names for files and artifacts."""

import re


def gpu_key(name: str) -> str:
    """Keep the NVIDIA name's words and numbers; normalize separators only."""
    key = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    if not key:
        raise ValueError("NVIDIA GPU name is empty or invalid")
    return key
