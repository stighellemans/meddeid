#!/usr/bin/env bash
set -euo pipefail

model_dir="${1:?usage: build_triton_repository.sh MODEL_DIR [OUTPUT_REPOSITORY]}"
output_repository="${2:-deploy/triton/model_repository}"
builder_image="${MEDDEID_TENSORRT_BUILDER_IMAGE:-nvcr.io/nvidia/tensorrt:24.02-py3}"
min_batch="${MEDDEID_TRT_MIN_BATCH:-1}"
opt_batch="${MEDDEID_TRT_OPT_BATCH:-16}"
max_batch="${MEDDEID_TRT_MAX_BATCH:-64}"
min_sequence="${MEDDEID_TRT_MIN_SEQUENCE:-8}"
opt_sequence="${MEDDEID_TRT_OPT_SEQUENCE:-256}"
max_sequence="${MEDDEID_TRT_MAX_SEQUENCE:-512}"

model_dir="$(cd "$model_dir" && pwd -P)"
mkdir -p "$output_repository"
output_repository="$(cd "$output_repository" && pwd -P)"
temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/meddeid-triton-XXXXXX")"
trap 'rm -rf "$temporary_dir"' EXIT

python deploy/export_onnx.py --model-dir "$model_dir" --output "$temporary_dir/model.onnx"
model_contract="$(python deploy/render_triton_config.py \
  --model-dir "$model_dir" \
  --repository "$output_repository" \
  --max-batch-size "$max_batch")"
model_name="$(printf '%s\n' "$model_contract" | sed -n '1p')"
model_version="$(printf '%s\n' "$model_contract" | sed -n '2p')"

docker run --rm --gpus all \
  -v "$temporary_dir:/onnx:ro" \
  -v "$output_repository:/output" \
  "$builder_image" \
  bash -lc "/usr/src/tensorrt/bin/trtexec \
    --onnx=/onnx/model.onnx \
    --saveEngine=/output/$model_name/$model_version/model.plan \
    --fp16 \
    --builderOptimizationLevel=5 \
    --minShapes=input_ids:${min_batch}x${min_sequence},attention_mask:${min_batch}x${min_sequence} \
    --optShapes=input_ids:${opt_batch}x${opt_sequence},attention_mask:${opt_batch}x${opt_sequence} \
    --maxShapes=input_ids:${max_batch}x${max_sequence},attention_mask:${max_batch}x${max_sequence}"

printf 'TensorRT repository created at %s\n' "$output_repository"
