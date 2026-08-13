from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel


@dataclass
class DualHeadOutput:
    """Raw token-level logits emitted by the dual-head classifier."""

    bio_logits: torch.Tensor
    label_logits: torch.Tensor


class DualHeadTokenClassifier(torch.nn.Module):
    """Transformer encoder with separate BIO and entity-label classification heads.

    The checkpoint does not contain enough information to build this module by
    itself. Callers must provide the base encoder id and exact label counts from
    `model/bundle.json` so the classifier heads match the saved tensors.
    """

    def __init__(
        self,
        model_name: str,
        num_bio_labels: int,
        num_entity_labels: int,
        attn_implementation: str | None = None,
        initialize_from_pretrained: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        encoder_kwargs: dict[str, Any] = {"config": config}
        if attn_implementation is not None:
            encoder_kwargs["attn_implementation"] = attn_implementation
        if initialize_from_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, **encoder_kwargs)
        else:
            encoder_kwargs.pop("config")
            self.encoder = AutoModel.from_config(config, **encoder_kwargs)
        hidden_size = int(config.hidden_size)
        dropout_p = float(getattr(config, "hidden_dropout_prob", 0.1))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.bio_classifier = torch.nn.Linear(hidden_size, num_bio_labels)
        self.label_classifier = torch.nn.Linear(hidden_size, num_entity_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> DualHeadOutput:
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = self.dropout(encoder_out.last_hidden_state)
        return DualHeadOutput(
            bio_logits=self.bio_classifier(hidden),
            label_logits=self.label_classifier(hidden),
        )


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load either a training checkpoint payload or a plain state dict.

    Training checkpoints may include fields such as `epoch`, `args`, or metrics
    alongside `model_state_dict`; older/exported checkpoints may be just the
    state dict. The serving metadata still comes from `model/bundle.json`.
    """

    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("the meddeid installation is missing safetensors") from exc
        return {"args": {}}, load_file(path, device="cpu")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload, payload["model_state_dict"]
    if isinstance(payload, dict):
        return {"args": {}}, payload
    raise ValueError("Unsupported checkpoint format")
