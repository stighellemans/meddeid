from meddeid_core.language import LanguageProfile
from meddeid.language import resolve_language_profile


def test_source_tree_resolves_dutch_profile_without_inference_changes():
    profile = resolve_language_profile("nl-BE", version="1")
    assert isinstance(profile, LanguageProfile)
    assert profile.profile_id == "nl-BE"
