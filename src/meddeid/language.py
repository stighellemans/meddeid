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


def installed_language_profile_packages() -> dict[str, str | None]:
    """Report distributions that provide installed language profiles.

    Entry-point ownership is used instead of a hard-coded package list so
    health and provenance output automatically includes new language packs.
    Lightweight test entry points may not expose distribution metadata; those
    providers are intentionally omitted rather than guessed from their name.
    """

    packages: dict[str, str | None] = {}
    for entry_point in _providers():
        distribution = getattr(entry_point, "dist", None)
        if distribution is None:
            continue
        metadata = getattr(distribution, "metadata", {})
        name = metadata.get("Name") if hasattr(metadata, "get") else None
        if not name:
            continue
        packages[str(name)] = getattr(distribution, "version", None)
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def resolve_language_profile(profile_id: str) -> LanguageProfile:
    attempted: list[str] = []
    for entry_point in _providers():
        attempted.append(entry_point.name)
        provider = entry_point.load()
        try:
            profile = provider(profile_id)
        except ValueError:
            continue
        if not isinstance(profile, LanguageProfile):
            raise TypeError(
                f"language-profile plugin {entry_point.name!r} returned {type(profile).__name__}, "
                "expected meddeid_core.LanguageProfile"
            )
        if not profile.accepts_language(profile_id):
            raise ValueError(
                f"language-profile plugin {entry_point.name!r} returned "
                f"{profile.profile_id}, which does not satisfy {profile_id!r}"
            )
        return profile

    installed = ", ".join(attempted) or "none"
    raise ValueError(
        f"no installed language-profile plugin provides {profile_id!r}; "
        f"discovered plugins: {installed}. Install the matching meddeid-language-* package."
    )
