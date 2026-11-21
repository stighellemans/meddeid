import pytest

from meddeid.api import DeidentificationResult
from meddeid.rendering import PseudonymizationScope, derive_date_offset, pseudonymize_name


@pytest.mark.parametrize(("locale", "alias"), [
    ("nl-BE", "[Jonathon Silva]"), ("nl-NL", "[Raymundus Neervoort]"),
    ("en-GB", "[Nabiel ROBSON]"), ("en-US", "[Denzel Memije]"),
])
def test_predictions_and_structured_values_share_mapping(locale, alias):
    scope = PseudonymizationScope(patient_id="P-0007", secret=bytes(range(32)))
    spans = [{"begin": 0, "end": 11, "label": "Name:Patient", "text": "Alex Peters"}]
    result = DeidentificationResult("Alex Peters", spans, "[Name:Patient]",
                                    {"language_profile": {"profile_id": locale}})
    assert result.render() == result.deid_text
    assert result.render(pseudonymization_scope=scope, name_replacement=True) == alias
    assert pseudonymize_name(result.text, pseudonymization_scope=scope, profile_id=locale) == alias
    assert derive_date_offset(scope) == -1368
