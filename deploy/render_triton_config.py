#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from meddeid.bundle import load_model_bundle


def render_config(
    *,
    model_name: str,
    model_version: str,
    max_batch_size: int,
    instance_count: int,
    bio_labels: int,
    entity_labels: int,
    output_type: str,
    queue_delay_microseconds: int | None,
) -> str:
    scheduler = ""
    if queue_delay_microseconds is not None:
        scheduler = f"""dynamic_batching {{
  max_queue_delay_microseconds: {queue_delay_microseconds}
}}
"""
    return f'''name: "{model_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch_size}
input [
  {{ name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] }},
  {{ name: "attention_mask" data_type: TYPE_INT32 dims: [ -1 ] }}
]
output [
  {{ name: "bio_logits" data_type: {output_type} dims: [ -1, {bio_labels} ] }},
  {{ name: "label_logits" data_type: {output_type} dims: [ -1, {entity_labels} ] }}
]
{scheduler}instance_group [ {{ count: {instance_count} kind: KIND_GPU }} ]
version_policy {{ specific {{ versions: [ {int(model_version)} ] }} }}
'''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a TensorRT Triton repository config"
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-precision", choices=("fp16", "fp32"), required=True)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument(
        "--throughput-queue-delay-microseconds",
        type=int,
        default=None,
        help="enable Triton dynamic batching and set its maximum queue delay",
    )
    args = parser.parse_args()
    if (
        args.throughput_queue_delay_microseconds is not None
        and args.throughput_queue_delay_microseconds < 0
    ):
        parser.error("--throughput-queue-delay-microseconds cannot be negative")

    bundle = load_model_bundle(args.model_dir / "bundle.json", validate_package=True)
    model_root = args.repository / bundle.name
    (model_root / bundle.model_version).mkdir(parents=True, exist_ok=True)
    output_type = "TYPE_FP16" if args.output_precision == "fp16" else "TYPE_FP32"
    common = {
        "model_name": bundle.name,
        "model_version": bundle.model_version,
        "max_batch_size": args.max_batch_size,
        "instance_count": args.instance_count,
        "bio_labels": len(bundle.bio_labels),
        "entity_labels": len(bundle.entity_labels),
        "output_type": output_type,
    }
    latency_config = render_config(**common, queue_delay_microseconds=None)
    throughput_config = render_config(
        **common,
        queue_delay_microseconds=args.throughput_queue_delay_microseconds,
    )
    (model_root / "config.pbtxt").write_text(latency_config, encoding="utf-8")
    config_dir = model_root / "configs"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "latency.pbtxt").write_text(latency_config, encoding="utf-8")
    (config_dir / "throughput.pbtxt").write_text(
        throughput_config,
        encoding="utf-8",
    )
    print(f"{bundle.name}\n{bundle.model_version}")


if __name__ == "__main__":
    main()
