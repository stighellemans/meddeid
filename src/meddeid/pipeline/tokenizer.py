from __future__ import annotations

import os
from typing import Any

from transformers import AutoTokenizer


def load_tokenizer(model_name: str, *, local_files_only: bool = False):
    def _from_pretrained(use_fast: bool):
        return AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            add_prefix_space=True,
            local_files_only=local_files_only,
        )

    try:
        tokenizer = _from_pretrained(use_fast=True)
    except ImportError as exc:
        if "protobuf" not in str(exc).lower():
            raise
        tokenizer = _from_pretrained(use_fast=False)

    stable_offsets_enabled = str(
        os.environ.get("DEID_TOKENIZER_STABLE_OFFSETS", "1")
    ).strip().lower() not in {"0", "false", "no", "off"}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    pre_tokenizer = getattr(backend, "pre_tokenizer", None)
    if (
        stable_offsets_enabled
        and getattr(tokenizer, "is_fast", False)
        and backend is not None
        and "ByteLevel" in str(pre_tokenizer)
    ):
        try:
            from tokenizers.pre_tokenizers import ByteLevel

            backend.pre_tokenizer = ByteLevel(
                add_prefix_space=True,
                use_regex=True,
            )
        except Exception:
            pass

    return tokenizer


def build_prepared_window(tokenizer, ids_slice: list[int]) -> dict[str, Any]:
    try:
        prepared = tokenizer.prepare_for_model(
            ids_slice,
            add_special_tokens=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
    except (AssertionError, NotImplementedError, TypeError, ValueError):
        build_inputs = getattr(tokenizer, "build_inputs_with_special_tokens", None)
        if callable(build_inputs):
            packed_ids = build_inputs(ids_slice)
        else:
            cls_id = getattr(tokenizer, "cls_token_id", None)
            sep_id = getattr(tokenizer, "sep_token_id", None)
            if sep_id is None:
                sep_id = getattr(tokenizer, "eos_token_id", None)
            if cls_id is None or sep_id is None:
                raise AttributeError(f"{tokenizer.__class__.__name__} cannot add special tokens")
            packed_ids = [cls_id, *ids_slice, sep_id]

        get_special_mask = getattr(tokenizer, "get_special_tokens_mask", None)
        if callable(get_special_mask):
            try:
                special_mask = get_special_mask(packed_ids, already_has_special_tokens=True)
            except (AssertionError, NotImplementedError, TypeError, ValueError):
                special_mask = [1] + [0] * len(ids_slice) + [1]
        else:
            special_mask = [1] + [0] * len(ids_slice) + [1]

        prepared = {
            "input_ids": packed_ids,
            "attention_mask": [1] * len(packed_ids),
            "special_tokens_mask": special_mask,
        }

    return prepared
