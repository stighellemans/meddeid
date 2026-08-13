#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from meddeid.bundle import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a TensorRT Triton repository config")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--instance-count", type=int, default=1)
    args = parser.parse_args()

    bundle = load_model_bundle(args.model_dir / "bundle.json", validate_package=True)
    model_root = args.repository / bundle.name
    (model_root / bundle.model_version).mkdir(parents=True, exist_ok=True)
    config = f'''name: "{bundle.name}"
platform: "tensorrt_plan"
max_batch_size: {args.max_batch_size}
input [
  {{ name: "input_ids" data_type: TYPE_INT32 dims: [ -1 ] }},
  {{ name: "attention_mask" data_type: TYPE_INT32 dims: [ -1 ] }}
]
output [
  {{ name: "bio_logits" data_type: TYPE_FP32 dims: [ -1, {len(bundle.bio_labels)} ] }},
  {{ name: "label_logits" data_type: TYPE_FP32 dims: [ -1, {len(bundle.entity_labels)} ] }}
]
dynamic_batching {{
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000
}}
instance_group [ {{ count: {args.instance_count} kind: KIND_GPU }} ]
version_policy {{ specific {{ versions: [ {int(bundle.model_version)} ] }} }}
'''
    (model_root / "config.pbtxt").write_text(config, encoding="utf-8")
    print(f"{bundle.name}\n{bundle.model_version}")


if __name__ == "__main__":
    main()
