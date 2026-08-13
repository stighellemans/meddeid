from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreparedWindow:
    doc_index: int
    begin: int
    end: int
    input_ids: list[int]
    attention_mask: list[int]
    special_tokens_mask: list[int]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class PreparedDocument:
    record_id: str
    text: str
    metadata: dict[str, Any]
    token_ids: list[int]
    offsets: list[tuple[int, int]]
    window_spans: list[tuple[int, int]]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def input_chars(self) -> int:
        return len(self.text)


@dataclass
class DocumentPredictionState:
    prepared: PreparedDocument
    bio_logits_sum: np.ndarray
    entity_logits_sum: np.ndarray
    counts: np.ndarray
    windows: list[PreparedWindow] = field(default_factory=list)


@dataclass(frozen=True)
class WindowPrediction:
    bio_logits: np.ndarray
    label_logits: np.ndarray


@dataclass(frozen=True)
class DetectionResult:
    spans: list[dict[str, Any]]
    token_rows: list[dict[str, Any]]
    stats: dict[str, Any]
