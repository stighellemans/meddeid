import pytest

from meddeid_core.language import LanguageProfile
from meddeid import language
from meddeid.language import installed_language_profile_packages, resolve_language_profile


class _EntryPoint:
    def __init__(self, name, provider, *, dist=None):
        self.name = name
        self._provider = provider
        self.dist = dist

    def load(self):
        return self._provider


def _installed_profiles(monkeypatch):
    from meddeid_language_en import get_profile as english
    from meddeid_language_nl import get_profile as dutch

    monkeypatch.setattr(
        language,
        "_providers",
        lambda: [_EntryPoint("nl", dutch), _EntryPoint("en", english)],
    )


@pytest.mark.parametrize(
    "requested, expected",
    [("nl-BE", "nl-BE"), ("en-GB", "en-GB"), ("en_GB", "en-GB"), ("en-US", "en-US"), ("en_US", "en-US")],
)
def test_installed_entry_points_are_authoritative(monkeypatch, requested, expected):
    _installed_profiles(monkeypatch)
    profile = resolve_language_profile(requested)
    assert isinstance(profile, LanguageProfile)
    assert profile.profile_id == expected


def test_bare_english_is_rejected_as_ambiguous(monkeypatch):
    _installed_profiles(monkeypatch)
    with pytest.raises(ValueError, match="no installed language-profile plugin"):
        resolve_language_profile("en")


def test_no_source_tree_fallback_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(language, "_providers", lambda: [])
    with pytest.raises(ValueError, match="discovered plugins: none"):
        resolve_language_profile("nl-BE")


def test_installed_package_reporting_comes_from_entry_point_owners(monkeypatch):
    class Distribution:
        metadata = {"Name": "meddeid-language-en"}
        version = "1.2.3"

    monkeypatch.setattr(
        language,
        "_providers",
        lambda: [_EntryPoint("en", lambda *_args, **_kwargs: None, dist=Distribution())],
    )

    assert installed_language_profile_packages() == {
        "meddeid-language-en": "1.2.3"
    }
