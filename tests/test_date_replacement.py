from __future__ import annotations

from meddeid.api import Deidentifier, redact
from meddeid_core.age_policy import load_age_granularity_policy
from meddeid_language_nl import get_profile
from meddeid_language_en import get_profile as get_english_profile


def engine(*, minimum: int = 366, policy=None) -> Deidentifier:
    instance = object.__new__(Deidentifier)
    instance.min_recommended_date_shift_days = minimum
    instance.age_granularity_policy = load_age_granularity_policy(policy)
    return instance


def span(text: str, label: str = "Date") -> dict:
    return {"begin": 0, "end": len(text), "text": text, "label": label}


def warning_codes(warnings: list[dict[str, str]]) -> set[str]:
    return {warning["code"] for warning in warnings}


def test_missing_and_zero_shifts_use_placeholders_but_only_zero_warns() -> None:
    profile = get_profile("nl-BE")
    instance = engine()
    for metadata, expected_warnings in (
        ({}, set()),
        ({"date_shift_days": 0}, {"zero_date_shift_placeholder"}),
    ):
        spans, warnings, processing = instance._apply_replacements(
            "15/01/2025", [span("15/01/2025")], metadata, profile
        )
        assert spans[0]["replacement"] == "[Date]"
        assert redact("15/01/2025", spans) == "[Date]"
        assert warning_codes(warnings) == expected_warnings
        assert processing["date_replacement"]["placeholder_spans"] == 1


def test_nonzero_shift_is_applied_and_weak_threshold_is_deployment_wide() -> None:
    profile = get_profile("nl-BE")
    instance = engine(minimum=30)
    shifted, warnings, processing = instance._apply_replacements(
        "15/01/2025",
        [span("15/01/2025")],
        {"date_shift_days": 17},
        profile,
    )
    assert shifted[0]["replacement"] == "[01/02/2025]"
    assert warning_codes(warnings) == {"date_shift_below_recommended_minimum"}
    assert processing["date_replacement"]["shifted_spans"] == 1

    _, strong_warnings, _ = instance._apply_replacements(
        "15/01/2025",
        [span("15/01/2025")],
        {"date_shift_days": -30},
        profile,
    )
    assert strong_warnings == []


def test_parse_failure_and_birthdate_year_fallback_are_explicit() -> None:
    profile = get_profile("nl-BE")
    instance = engine()
    failed, warnings, _ = instance._apply_replacements(
        "onbekend", [span("onbekend")], {"date_shift_days": 400}, profile
    )
    assert failed[0]["replacement"] == "[Date]"
    assert warning_codes(warnings) == {"date_parse_fallback"}

    fallback, warnings, processing = instance._apply_replacements(
        "14/05/2013",
        [span("14/05/2013", "Age_Birthdate")],
        {"date_shift_days": 400},
        profile,
    )
    assert fallback[0]["replacement"] == "[2014]"
    assert warning_codes(warnings) == {"birthdate_year_fallback"}
    assert processing["date_replacement"]["year_fallback_spans"] == 1


def test_custom_policy_is_used_by_the_language_profile() -> None:
    policy = {
        "schema_version": "meddeid.age-granularity.v1",
        "policy_id": "weeks",
        "policy_version": "1",
        "bands": [{"output": ["week"]}],
    }
    profile = get_profile("nl-BE")
    replaced, warnings, processing = engine(policy=policy)._apply_replacements(
        "18 weken",
        [span("18 weken", "Age_Birthdate")],
        {"date_shift_days": 400},
        profile,
    )
    assert replaced[0]["replacement"] == "[18 weken]"
    assert warnings == []
    assert processing["age_granularity_policy"]["policy_id"] == "weeks"


def test_dutch_and_english_share_semantic_age_granularity() -> None:
    instance = engine()
    cases = (
        (get_profile("nl-BE"), "14/09/2022", "14/05/2026", "[3 jaar, 7 maanden oud]"),
        (get_english_profile("en-GB"), "14/09/2022", "14/05/2026", "[3 years, 7 months old]"),
    )
    for profile, birthdate, document_date, expected in cases:
        replaced, warnings, processing = instance._apply_replacements(
            birthdate,
            [span(birthdate, "Age_Birthdate")],
            {"date_shift_days": 400, "document_creation_date": document_date},
            profile,
        )
        assert replaced[0]["replacement"] == expected
        assert warnings == []
        assert processing["date_replacement"]["age_generalized_spans"] == 1


def test_patient_birth_date_expands_to_trusted_locale_variants() -> None:
    instance = engine()
    profile = get_profile("nl-BE")
    metadata = instance._with_patient_birth_date_variants(
        {"patient": {"birth_date": "2013-05-14"}}, profile
    )
    assert {
        (item["value"], item["label"]) for item in metadata["known_values"]
    } >= {("14 mei 2013", "Age_Birthdate")}
    spans = profile.post_process_spans([], "Geboren op 14 mei 2013.", metadata)
    assert any(item["text"] == "14 mei 2013" for item in spans)


def test_invalid_birth_date_and_shift_types_are_rejected() -> None:
    instance = engine()
    profile = get_profile("nl-BE")
    try:
        instance._with_patient_birth_date_variants(
            {"patient": {"birth_date": "not-a-date"}}, profile
        )
    except ValueError as exc:
        assert "patient.birth_date" in str(exc)
    else:
        raise AssertionError("invalid patient birth date was accepted")

    try:
        instance._apply_replacements(
            "15/01/2025", [span("15/01/2025")], {"date_shift_days": True}, profile
        )
    except ValueError as exc:
        assert "date_shift_days" in str(exc)
    else:
        raise AssertionError("boolean date shift was accepted")


def test_out_of_range_shift_becomes_a_validation_error() -> None:
    instance = engine()
    profile = get_profile("nl-BE")
    try:
        instance._apply_replacements(
            "15/01/2025",
            [span("15/01/2025")],
            {"date_shift_days": 10**10},
            profile,
        )
    except ValueError as exc:
        assert "outside the supported range" in str(exc)
    else:
        raise AssertionError("out-of-range date shift was accepted")
