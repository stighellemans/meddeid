from __future__ import annotations

from meddeid.api import redact
from meddeid.pipeline.core import windows_full_tail
from meddeid.pipeline.decoder import bio_tags_to_char_entities, split_label_fields
from meddeid.pipeline.tokenizer import build_prepared_window


def test_windows_full_tail_covers_tail_without_gaps() -> None:
    assert list(windows_full_tail(600, usable=254, overlap=64)) == [
        (0, 254),
        (190, 444),
        (346, 600),
    ]


def test_decoder_uses_first_label_for_entire_i_run() -> None:
    text = "Jan Peeters"
    spans = bio_tags_to_char_entities(
        bio_tags=["B", "I"],
        offsets=[(0, 3), (4, 11)],
        token_scores=[0.95, 0.92],
        token_entity_labels=["Name:Patient", "Name:Caregiver"],
        text=text,
        min_entity_score=0.0,
    )
    assert spans[0]["label"] == "Name:Patient"
    assert spans[0]["text"] == "Jan Peeters"


def test_split_label_fields_handles_plain_and_typed_labels() -> None:
    assert split_label_fields("Name:Patient") == ("Name", "Patient")
    assert split_label_fields("Date") == ("Date", None)


def test_redact_replaces_spans_from_right_to_left() -> None:
    text = "Jan bezocht Gent."
    spans = [
        {"begin": 0, "end": 3, "label": "Name:Patient"},
        {"begin": 12, "end": 16, "label": "Address_Location:Other"},
    ]
    assert redact(text, spans) == "[Name:Patient] bezocht [Address_Location:Other]."


def test_window_builder_supports_transformers_5_tokenizer_api() -> None:
    class TokenizerWithoutLegacyPackingMethods:
        cls_token_id = 0
        sep_token_id = 2

        def get_special_tokens_mask(self, ids, *, already_has_special_tokens):
            assert already_has_special_tokens is True
            return [1 if token_id in {0, 2} else 0 for token_id in ids]

    assert build_prepared_window(
        TokenizerWithoutLegacyPackingMethods(), [10, 11]
    ) == {
        "input_ids": [0, 10, 11, 2],
        "attention_mask": [1, 1, 1, 1],
        "special_tokens_mask": [1, 0, 0, 1],
    }
