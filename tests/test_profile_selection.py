from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from threading import Lock

import pytest

from meddeid.api import Deidentifier, choose_language_profile
from meddeid_core.language import LanguageProfile


@dataclass(frozen=True)
class _Profile:
    profile_id: str
    version: str = "1"

    def accepts_language(self, value: str) -> bool:
        normalized = value.strip().replace("_", "-")
        return normalized == self.profile_id


GB = _Profile("en-GB")
US = _Profile("en-US")


def test_document_metadata_wins_over_explicit_default() -> None:
    selected = choose_language_profile(
        (GB, US), document_language="en_US", explicit_default=GB
    )
    assert selected is US


def test_explicit_default_avoids_repeating_locale_on_every_document() -> None:
    selected = choose_language_profile(
        (GB, US), document_language=None, explicit_default=US
    )
    assert selected is US


def test_single_profile_bundle_is_unambiguous_without_metadata() -> None:
    assert choose_language_profile(
        (GB,), document_language=None, explicit_default=None
    ) is GB


@pytest.mark.parametrize("language", [None, "en"])
def test_multi_profile_bundle_never_guesses(language: str | None) -> None:
    with pytest.raises(ValueError, match="required|unambiguous"):
        choose_language_profile(
            (GB, US), document_language=language, explicit_default=None
        )


def test_postprocessing_and_provenance_are_selected_per_document() -> None:
    calls = []

    def profile(profile_id: str) -> LanguageProfile:
        def postprocess(spans, text, metadata):
            calls.append((profile_id, text, metadata.get("lang")))
            return [
                {
                    "begin": 0,
                    "end": len(text),
                    "text": text,
                    "label": f"profile:{profile_id}",
                }
            ]

        return LanguageProfile(
            profile_id=profile_id,
            language_tags=(profile_id,),
            post_process_spans=postprocess,
        )

    gb = profile("en-GB")
    us = profile("en-US")

    class Pipeline:
        def prepare_document(self, *, record_id, text, metadata):
            return SimpleNamespace(record_id=record_id, text=text, metadata=metadata)

        def prepare_state(self, doc_index, prepared):
            return SimpleNamespace(windows=[SimpleNamespace(doc_index=doc_index)])

        def apply_predictions(self, state, predictions):
            assert len(predictions) == 1

        def decode_document(self, state):
            return SimpleNamespace(spans=[])

    class Runtime:
        def infer_windows(self, windows):
            return [object() for _ in windows]

    engine = object.__new__(Deidentifier)
    engine._inference_lock = Lock()
    engine.pipeline = Pipeline()
    engine.runtime = Runtime()
    engine.language_profiles = (gb, us)
    engine.language_profile = gb
    engine.inference_provenance = lambda profile_id: {
        "contract_version": "meddeid.inference-provenance.v1",
        "language_profile": {"profile_id": profile_id},
    }

    results = engine.deidentify_many(
        [("British", None), ("American", {"lang": "en-US"})]
    )

    assert calls == [
        ("en-GB", "British", None),
        ("en-US", "American", "en-US"),
    ]
    assert [
        result.provenance["language_profile"]["profile_id"] for result in results
    ] == ["en-GB", "en-US"]
    assert results[0].spans[0]["label"] == "profile:en-GB"
    assert results[1].spans[0]["label"] == "profile:en-US"
