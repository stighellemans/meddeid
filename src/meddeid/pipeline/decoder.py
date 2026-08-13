from __future__ import annotations

from typing import Any

import numpy as np


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def split_bio_tag(tag: str) -> str:
    raw = str(tag).strip()
    if not raw or raw == "O":
        return "O"
    prefix = raw.split("-", 1)[0]
    if prefix in {"B", "I"}:
        return prefix
    return "O"


def split_label_fields(label: str) -> tuple[str, str | None]:
    category, _, subtype = label.partition(":")
    return category, subtype or None


def bio_tags_to_char_entities(
    *,
    bio_tags: list[str],
    offsets: list[tuple[int, int]],
    token_scores: list[float],
    token_entity_labels: list[str],
    text: str,
    min_entity_score: float,
) -> list[dict[str, Any]]:
    if (
        len(bio_tags) != len(offsets)
        or len(token_scores) != len(bio_tags)
        or len(token_entity_labels) != len(bio_tags)
    ):
        raise ValueError("BIO tags, offsets, token scores, and token labels must align")

    entities: list[dict[str, Any]] = []
    cur_start_tok: int | None = None
    last_valid_tok: int | None = None

    def flush(end_tok_inclusive: int | None) -> None:
        nonlocal cur_start_tok
        if cur_start_tok is None or end_tok_inclusive is None:
            cur_start_tok = None
            return

        char_start = int(offsets[cur_start_tok][0])
        char_end = int(offsets[end_tok_inclusive][1])
        if char_end <= char_start:
            cur_start_tok = None
            return

        span_scores = token_scores[cur_start_tok : end_tok_inclusive + 1]
        score = float(sum(span_scores) / max(1, len(span_scores)))
        if score < min_entity_score:
            cur_start_tok = None
            return

        label = str(token_entity_labels[cur_start_tok])
        category, subtype = split_label_fields(label)
        span = {
            "begin": char_start,
            "end": char_end,
            "label": label,
            "text": text[char_start:char_end],
            "category": category,
            "subtype": subtype,
            "score": score,
        }
        entities.append(span)
        cur_start_tok = None

    for tok_idx, tag in enumerate(bio_tags):
        char_begin, char_end = offsets[tok_idx]
        if int(char_end) <= int(char_begin):
            continue

        prefix = split_bio_tag(tag)
        if prefix == "O":
            flush(last_valid_tok)
            last_valid_tok = tok_idx
            continue

        if prefix == "B":
            flush(last_valid_tok)
            cur_start_tok = tok_idx
            last_valid_tok = tok_idx
            continue

        if prefix == "I":
            if cur_start_tok is not None:
                last_valid_tok = tok_idx
                continue
            flush(last_valid_tok)
            cur_start_tok = tok_idx
            last_valid_tok = tok_idx

    if bio_tags:
        flush(last_valid_tok)

    return entities
