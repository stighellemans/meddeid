#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from meddeid.bundle import load_model_bundle
from meddeid.model import DualHeadTokenClassifier, load_checkpoint


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: DualHeadTokenClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.bio_logits, output.label_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MedDeID bundle to temporary ONNX")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    bundle = load_model_bundle(model_dir / "bundle.json", validate_package=True)
    _, state = load_checkpoint(bundle.checkpoint_path)
    model = DualHeadTokenClassifier(
        str(bundle.encoder_config_path.parent),
        num_bio_labels=len(bundle.bio_labels),
        num_entity_labels=len(bundle.entity_labels),
        attn_implementation="eager",
        initialize_from_pretrained=False,
        local_files_only=True,
    )
    model.load_state_dict(state, strict=True)
    wrapper = ExportWrapper(model).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy_ids = torch.ones((2, bundle.inference.max_length), dtype=torch.int32)
    dummy_mask = torch.ones((2, bundle.inference.max_length), dtype=torch.int32)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(args.output),
        input_names=["input_ids", "attention_mask"],
        output_names=["bio_logits", "label_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "bio_logits": {0: "batch", 1: "sequence"},
            "label_logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )


if __name__ == "__main__":
    main()
