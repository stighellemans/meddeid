#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from meddeid.bundle import load_model_bundle
from meddeid.model import DualHeadTokenClassifier, load_checkpoint


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: DualHeadTokenClassifier, *, output_precision: str) -> None:
        super().__init__()
        self.model = model
        self.output_precision = output_precision

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.output_precision == "fp32":
            return output.bio_logits.float(), output.label_logits.float()
        return output.bio_logits, output.label_logits


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MedDeID bundle to temporary ONNX")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--output-precision", choices=("fp16", "fp32"), required=True)
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
    if args.precision == "fp16":
        model.half()
    wrapper = ExportWrapper(model, output_precision=args.output_precision).eval()

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
    exported = onnx.load(args.output)
    onnx.checker.check_model(exported)
    expected_initializer_type = (
        onnx.TensorProto.FLOAT16 if args.precision == "fp16" else onnx.TensorProto.FLOAT
    )
    expected_output_type = (
        onnx.TensorProto.FLOAT16
        if args.output_precision == "fp16"
        else onnx.TensorProto.FLOAT
    )
    floating_types = {
        onnx.TensorProto.FLOAT,
        onnx.TensorProto.FLOAT16,
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BFLOAT16,
    }
    wrong_initializers = [
        value.name
        for value in exported.graph.initializer
        if value.data_type in floating_types and value.data_type != expected_initializer_type
    ]
    wrong_outputs = [
        value.name
        for value in exported.graph.output
        if value.type.tensor_type.elem_type != expected_output_type
    ]
    if wrong_initializers or wrong_outputs:
        raise RuntimeError(
            f"ONNX precision validation failed for compute={args.precision}, "
            f"output={args.output_precision}: "
            f"initializers={wrong_initializers}, outputs={wrong_outputs}"
        )
    for key, value in (
        ("io.meddeid.precision", args.precision),
        ("io.meddeid.output-precision", args.output_precision),
    ):
        metadata = exported.metadata_props.add()
        metadata.key = key
        metadata.value = value
    onnx.save(exported, args.output)


if __name__ == "__main__":
    main()
