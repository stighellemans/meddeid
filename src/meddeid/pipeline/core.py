from __future__ import annotations

from typing import Any

import numpy as np

from meddeid.bundle import ModelBundle

from .decoder import bio_tags_to_char_entities, softmax_np
from .tokenizer import build_prepared_window, load_tokenizer
from .types import (
    DetectionResult,
    DocumentPredictionState,
    PreparedDocument,
    PreparedWindow,
    WindowPrediction,
)


def windows_full_tail(n_tokens: int, usable: int, overlap: int):
    if n_tokens <= 0:
        return
    if n_tokens <= usable:
        yield 0, n_tokens
        return

    step = usable - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than usable sequence length")

    last_begin = n_tokens - usable
    begin = 0
    while begin < last_begin:
        end = begin + usable
        yield begin, end
        begin += step
    yield last_begin, n_tokens


class DeidentificationPipeline:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        tokenizer_source = str(bundle.tokenizer_path) if bundle.tokenizer_path else bundle.base_encoder
        self.tokenizer = load_tokenizer(
            tokenizer_source,
            local_files_only=bundle.tokenizer_path is not None,
        )
        self.usable = bundle.inference.max_length - self.tokenizer.num_special_tokens_to_add(
            pair=False
        )
        if self.usable <= 8:
            raise ValueError("max_length too small after accounting for special tokens")
        self.bio_id2label = dict(enumerate(bundle.bio_labels))
        self.entity_id2label = dict(enumerate(bundle.entity_labels))
        self.bio_o_id = next(
            (idx for idx, value in self.bio_id2label.items() if value == "O"),
            0,
        )

    def prepare_document(
        self, *, record_id: str, text: str, metadata: dict[str, Any]
    ) -> PreparedDocument:
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            verbose=False,
        )
        token_ids = list(enc["input_ids"])
        offsets = [tuple(item) for item in enc["offset_mapping"]]
        window_spans = list(
            windows_full_tail(
                len(token_ids),
                usable=self.usable,
                overlap=self.bundle.inference.overlap,
            )
        )
        return PreparedDocument(
            record_id=record_id,
            text=text,
            metadata=dict(metadata),
            token_ids=token_ids,
            offsets=offsets,
            window_spans=window_spans,
        )

    def prepare_state(self, doc_index: int, prepared: PreparedDocument) -> DocumentPredictionState:
        num_tokens = prepared.token_count
        state = DocumentPredictionState(
            prepared=prepared,
            bio_logits_sum=np.zeros((num_tokens, len(self.bundle.bio_labels)), dtype=np.float64),
            entity_logits_sum=np.zeros(
                (num_tokens, len(self.bundle.entity_labels)), dtype=np.float64
            ),
            counts=np.zeros(num_tokens, dtype=np.int64),
        )
        for begin, end in prepared.window_spans:
            built = build_prepared_window(self.tokenizer, prepared.token_ids[begin:end])
            state.windows.append(
                PreparedWindow(
                    doc_index=doc_index,
                    begin=begin,
                    end=end,
                    input_ids=list(built["input_ids"]),
                    attention_mask=list(built["attention_mask"]),
                    special_tokens_mask=list(built["special_tokens_mask"]),
                )
            )
        return state

    def apply_predictions(
        self, state: DocumentPredictionState, predictions: list[WindowPrediction]
    ) -> None:
        if len(state.windows) != len(predictions):
            raise ValueError("Window predictions do not align with prepared windows")

        for window, prediction in zip(state.windows, predictions, strict=True):
            special_mask = np.array(window.special_tokens_mask, dtype=bool)
            token_bio_logits = prediction.bio_logits[~special_mask]
            token_label_logits = prediction.label_logits[~special_mask]
            if token_bio_logits.shape[0] != (window.end - window.begin):
                raise RuntimeError("Mismatch between inference logits and token window length")
            state.bio_logits_sum[window.begin : window.end] += token_bio_logits
            state.entity_logits_sum[window.begin : window.end] += token_label_logits
            state.counts[window.begin : window.end] += 1

    def decode_document(self, state: DocumentPredictionState) -> DetectionResult:
        prepared = state.prepared
        if prepared.token_count == 0:
            return DetectionResult(
                spans=[],
                token_rows=[],
                stats={
                    "input_chars": prepared.input_chars,
                    "token_count": 0,
                    "window_count": 0,
                    "truncated": False,
                },
            )

        if np.any(state.counts == 0):
            missing = int(np.sum(state.counts == 0))
            raise RuntimeError(
                f"{missing} tokens received no predictions in record_id={prepared.record_id}"
            )

        avg_bio_logits = state.bio_logits_sum / state.counts[:, None]
        avg_entity_logits = state.entity_logits_sum / state.counts[:, None]
        pred_bio_ids = avg_bio_logits.argmax(axis=1).tolist()
        pred_entity_ids = avg_entity_logits.argmax(axis=1).tolist()

        bio_probs = softmax_np(avg_bio_logits)
        entity_probs = softmax_np(avg_entity_logits)
        bio_tags: list[str] = []
        token_entity_labels: list[str] = []
        token_scores: list[float] = []
        token_rows: list[dict[str, Any]] = []

        for tok_idx, (pb, pe) in enumerate(zip(pred_bio_ids, pred_entity_ids, strict=True)):
            pb_i = int(pb)
            pe_i = int(pe)
            bio_tag = str(self.bio_id2label.get(pb_i, "O"))
            ent_name = str(self.entity_id2label.get(pe_i, f"ENTITY_{pe_i}"))
            begin, end = prepared.offsets[tok_idx]
            token_text = prepared.text[begin:end]
            o_prob = (
                float(bio_probs[tok_idx, self.bio_o_id])
                if 0 <= self.bio_o_id < len(self.bundle.bio_labels)
                else float(np.max(bio_probs[tok_idx]))
            )
            bio_prob = float(bio_probs[tok_idx, pb_i])
            entity_prob = float(entity_probs[tok_idx, pe_i])
            token_rows.append(
                {
                    "idx": tok_idx,
                    "begin": int(begin),
                    "end": int(end),
                    "offset_valid": bool(int(end) > int(begin)),
                    "text": token_text,
                    "bio_id": pb_i,
                    "bio_tag": bio_tag,
                    "bio_score": bio_prob,
                    "entity_id": pe_i,
                    "entity_label": ent_name,
                    "entity_score": entity_prob,
                }
            )

            if pb_i == self.bio_o_id or bio_tag == "O":
                bio_tags.append("O")
                token_entity_labels.append(ent_name)
                token_scores.append(o_prob)
                continue

            if bio_tag not in {"B", "I"}:
                bio_tags.append("O")
                token_entity_labels.append(ent_name)
                token_scores.append(o_prob)
                continue

            bio_tags.append(bio_tag)
            token_entity_labels.append(ent_name)
            token_scores.append(float(bio_probs[tok_idx, pb_i]))

        spans = bio_tags_to_char_entities(
            bio_tags=bio_tags,
            offsets=prepared.offsets,
            token_scores=token_scores,
            token_entity_labels=token_entity_labels,
            text=prepared.text,
            min_entity_score=self.bundle.inference.min_entity_score,
        )
        return DetectionResult(
            spans=spans,
            token_rows=token_rows,
            stats={
                "input_chars": prepared.input_chars,
                "token_count": prepared.token_count,
                "window_count": len(prepared.window_spans),
                "truncated": False,
            },
        )
