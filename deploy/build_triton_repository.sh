#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"
cd "${project_dir}"

model_dir="${1:?usage: build_triton_repository.sh MODEL_DIR OUTPUT_REPOSITORY TARGET [GPU_DEVICE_ID]}"
output_repository="${2:?usage: build_triton_repository.sh MODEL_DIR OUTPUT_REPOSITORY TARGET [GPU_DEVICE_ID]}"
gpu_target="${3:?usage: build_triton_repository.sh MODEL_DIR OUTPUT_REPOSITORY TARGET [GPU_DEVICE_ID]}"
gpu_device_id="${4:-0}"

"${script_dir}/preflight_triton_host.sh" "${gpu_target}" "${gpu_device_id}"
model_dir="$(cd -- "${model_dir}" && pwd -P)"
output_parent="$(dirname -- "${output_repository}")"
output_name="$(basename -- "${output_repository}")"
mkdir -p -- "${output_parent}"
output_parent="$(cd -- "${output_parent}" && pwd -P)"
output_repository="${output_parent}/${output_name}"

triton_scheduler_args=()
case "${MEDDEID_TRITON_THROUGHPUT_DYNAMIC_BATCHING}" in
  true)
    triton_scheduler_args=(
      --throughput-queue-delay-microseconds
      "${MEDDEID_TRITON_THROUGHPUT_QUEUE_DELAY_US}"
    )
    ;;
  false) ;;
  *)
    printf 'MEDDEID_TRITON_THROUGHPUT_DYNAMIC_BATCHING must be true or false.\n' >&2
    exit 1
    ;;
esac

if [[ -e "${output_repository}" ]]; then
  if [[ -n "$(find "${output_repository}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'Refusing to replace non-empty repository: %s\n' "${output_repository}" >&2
    exit 1
  fi
  rmdir -- "${output_repository}"
fi

build_root="$(mktemp -d "${output_parent}/.${output_name}.build-XXXXXX")"
onnx_root="$(mktemp -d "${TMPDIR:-/tmp}/meddeid-onnx-XXXXXX")"
cleanup() {
  rm -rf -- "${build_root}" "${onnx_root}"
}
trap cleanup EXIT

python -m deploy.export_onnx \
  --model-dir "${model_dir}" \
  --output "${onnx_root}/model.onnx" \
  --precision "${MEDDEID_TRT_PRECISION}" \
  --output-precision "${MEDDEID_TRT_OUTPUT_PRECISION}"
model_contract="$(python "${script_dir}/render_triton_config.py" \
  --model-dir "${model_dir}" \
  --repository "${build_root}" \
  --output-precision "${MEDDEID_TRT_OUTPUT_PRECISION}" \
  --max-batch-size "${MEDDEID_TRT_MAX_BATCH}" \
  "${triton_scheduler_args[@]}")"
model_name="$(printf '%s\n' "${model_contract}" | sed -n '1p')"
model_version="$(printf '%s\n' "${model_contract}" | sed -n '2p')"

case "${MEDDEID_TRT_PRECISION}" in
  fp16 | fp32) ;;
  *) printf 'MEDDEID_TRT_PRECISION must be fp16 or fp32.\n' >&2; exit 1 ;;
esac

docker run --rm --gpus "device=${gpu_device_id}" \
  -v "${onnx_root}:/onnx:ro" \
  -v "${build_root}:/output" \
  "${MEDDEID_TENSORRT_BUILDER_IMAGE}" \
  trtexec \
    --onnx=/onnx/model.onnx \
    --saveEngine="/output/${model_name}/${model_version}/model.plan" \
    --builderOptimizationLevel=5 \
    --skipInference \
    --minShapes="input_ids:${MEDDEID_TRT_MIN_BATCH}x${MEDDEID_TRT_MIN_SEQUENCE},attention_mask:${MEDDEID_TRT_MIN_BATCH}x${MEDDEID_TRT_MIN_SEQUENCE}" \
    --optShapes="input_ids:${MEDDEID_TRT_OPT_BATCH}x${MEDDEID_TRT_OPT_SEQUENCE},attention_mask:${MEDDEID_TRT_OPT_BATCH}x${MEDDEID_TRT_OPT_SEQUENCE}" \
    --maxShapes="input_ids:${MEDDEID_TRT_MAX_BATCH}x${MEDDEID_TRT_MAX_SEQUENCE},attention_mask:${MEDDEID_TRT_MAX_BATCH}x${MEDDEID_TRT_MAX_SEQUENCE}"

gpu_name="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
compute_capability="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=compute_cap --format=csv,noheader | head -n 1 | xargs)"
driver_version="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=driver_version --format=csv,noheader | head -n 1 | xargs)"

python "${script_dir}/write_triton_manifest.py" \
  --repository "${build_root}" \
  --model-name "${model_name}" \
  --model-version "${model_version}" \
  --model-id "${MEDDEID_MODEL_ID}" \
  --model-revision "${MEDDEID_MODEL_REVISION}" \
  --bundle-sha256 "${MEDDEID_BUNDLE_SHA256}" \
  --triton-stack "${MEDDEID_TRITON_STACK}" \
  --triton-server-version "${MEDDEID_TRITON_SERVER_VERSION}" \
  --tensorrt-version "${MEDDEID_TENSORRT_VERSION}" \
  --builder-image "${MEDDEID_TENSORRT_BUILDER_IMAGE}" \
  --gpu-target "${gpu_target}" \
  --gpu-name "${gpu_name}" \
  --compute-capability "${compute_capability}" \
  --driver-version "${driver_version}" \
  --precision "${MEDDEID_TRT_PRECISION}" \
  --output-precision "${MEDDEID_TRT_OUTPUT_PRECISION}" \
  --min-shape "${MEDDEID_TRT_MIN_BATCH}x${MEDDEID_TRT_MIN_SEQUENCE}" \
  --opt-shape "${MEDDEID_TRT_OPT_BATCH}x${MEDDEID_TRT_OPT_SEQUENCE}" \
  --max-shape "${MEDDEID_TRT_MAX_BATCH}x${MEDDEID_TRT_MAX_SEQUENCE}" \
  --throughput-dynamic-batching \
    "${MEDDEID_TRITON_THROUGHPUT_DYNAMIC_BATCHING}" \
  "${triton_scheduler_args[@]}"

mv -- "${build_root}" "${output_repository}"
printf 'TensorRT repository created at %s\n' "${output_repository}"
printf 'Review its immutable build manifest before packaging: %s/build-manifest.json\n' "${output_repository}"
