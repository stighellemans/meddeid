#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"

target="${1:?usage: preflight_triton_host.sh TARGET [GPU_DEVICE_ID]}"
gpu_device_id="${2:-0}"

if [[ "$(uname -s)" != Linux ]]; then
  printf 'TensorRT plan creation requires a Linux NVIDIA host; found %s.\n' "$(uname -s)" >&2
  exit 1
fi
for command_name in nvidia-smi python hf; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 1
  fi
done

gpu_name="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
compute_capability="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=compute_cap --format=csv,noheader | head -n 1 | xargs)"
driver_version="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=driver_version --format=csv,noheader | head -n 1 | xargs)"

if python "${script_dir}/triton_targets.py" show "${target}" >/dev/null 2>&1; then
  python "${script_dir}/triton_targets.py" verify-host "${target}" \
    --gpu-name "${gpu_name}" \
    --compute-capability "${compute_capability}"
else
  detected_local_target="$(python "${script_dir}/triton_targets.py" detect-host \
    --gpu-name "${gpu_name}" \
    --compute-capability "${compute_capability}" \
    --allow-local)"
  if [[ "${target}" != "${detected_local_target}" ]]; then
    printf 'Target %s does not match detected local target %s.\n' \
      "${target}" "${detected_local_target}" >&2
    exit 1
  fi
fi

if command -v trtexec >/dev/null 2>&1; then
  trtexec --help >/dev/null
elif command -v docker >/dev/null 2>&1; then
  docker info >/dev/null
  docker run --rm --gpus "device=${gpu_device_id}" \
    "${MEDDEID_TENSORRT_BUILDER_IMAGE}" \
    bash -lc 'command -v trtexec && trtexec --help >/dev/null && nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader'
else
  printf 'Missing required command: trtexec or docker\n' >&2
  exit 1
fi

cat <<EOF
MedDeID Triton host preflight passed.
  project: ${project_dir}
  target: ${target}
  GPU device: ${gpu_device_id}
  GPU: ${gpu_name}
  compute capability: ${compute_capability}
  NVIDIA driver: ${driver_version}
  Triton stack: ${MEDDEID_TRITON_STACK}
  TensorRT: ${MEDDEID_TENSORRT_VERSION}
EOF
