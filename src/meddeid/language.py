"""Discover installed language-profile packages without hard-wiring inference."""

from __future__ import annotations

from importlib.metadata import entry_points

from meddeid_core.language import LanguageProfile

ENTRY_POINT_GROUP = "meddeid.language_profiles"


def _providers():
    discovered = entry_points()
    if hasattr(discovered, "select"):
        yield from discovered.select(group=ENTRY_POINT_GROUP)
    else:  # Python 3.10 compatibility
        yield from discovered.get(ENTRY_POINT_GROUP, ())


def resolve_language_profile(profile_id: str, *, version: str) -> LanguageProfile:
    attempted: list[str] = []
    for entry_point in _providers():
        attempted.append(entry_point.name)
        provider = entry_point.load()
        try:
            profile = provider(profile_id, version=version)
        except ValueError:
            continue
        if not isinstance(profile, LanguageProfile):
            raise TypeError(
                f"language-profile plugin {entry_point.name!r} returned {type(profile).__name__}, "
                "expected meddeid_core.LanguageProfile"
            )
        return profile

    # Source-tree development does not install entry-point metadata. Keep this
    # fallback out of the normal installed path while retaining local ergonomics.
    try:
        from meddeid_language_nl import get_profile

        return get_profile(profile_id, version=version)
    except (ImportError, ValueError) as exc:
        installed = ", ".join(attempted) or "none"
        raise ValueError(
            f"no installed language-profile plugin provides {profile_id!r}@{version}; "
            f"discovered plugins: {installed}. Install the matching meddeid-language-* package."
        ) from exc

