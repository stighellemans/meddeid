#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=pytorch-cuda/versions.env
source "${script_dir}/pytorch-cuda/versions.env"

image="${1:?usage: validate_pytorch_cuda_image.sh IMAGE [GPU_DEVICE_ID] [REPORT]}"
gpu_device_id="${2:-0}"
report="${3:-${script_dir}/pytorch-cuda/validation-report.json}"
run_id="$$"
network_name="meddeid-cuda-validation-${run_id}"
container_name="meddeid-cuda-validation-${run_id}"

if [[ "$(uname -s)" != Linux ]]; then
  printf 'PyTorch CUDA validation requires a Linux NVIDIA host; found %s.\n' "$(uname -s)" >&2
  exit 1
fi
for command_name in docker nvidia-smi; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 1
  fi
done
docker info >/dev/null
nvidia-smi --id="${gpu_device_id}" >/dev/null

accelerator_label="$(docker image inspect "${image}" --format '{{index .Config.Labels "io.meddeid.accelerator"}}')"
torch_label="$(docker image inspect "${image}" --format '{{index .Config.Labels "io.meddeid.torch-version"}}')"
cuda_label="$(docker image inspect "${image}" --format '{{index .Config.Labels "io.meddeid.cuda-version"}}')"
if [[ "${accelerator_label}" != cuda ]]; then
  printf 'Image accelerator label is %q, expected cuda.\n' "${accelerator_label}" >&2
  exit 1
fi
if [[ "${torch_label}" != "${MEDDEID_PYTORCH_VERSION}" ]]; then
  printf 'Image PyTorch label is %q, expected %s.\n' "${torch_label}" "${MEDDEID_PYTORCH_VERSION}" >&2
  exit 1
fi
if [[ "${cuda_label}" != "${MEDDEID_PYTORCH_CUDA_VERSION}" ]]; then
  printf 'Image CUDA label is %q, expected %s.\n' "${cuda_label}" "${MEDDEID_PYTORCH_CUDA_VERSION}" >&2
  exit 1
fi

mkdir -p -- "$(dirname -- "${report}")"
docker run --rm \
  --gpus "device=${gpu_device_id}" \
  --volume "${project_dir}/deploy/check_pytorch_cuda.py:/check.py:ro" \
  --entrypoint python \
  "${image}" \
  /check.py --expected-cuda "${MEDDEID_PYTORCH_CUDA_VERSION}" >"${report}"

cleanup() {
  docker container rm --force "${container_name}" >/dev/null 2>&1 || true
  docker network rm "${network_name}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker network create --internal "${network_name}" >/dev/null
docker run --detach --name "${container_name}" \
  --network "${network_name}" \
  --gpus "device=${gpu_device_id}" \
  --read-only \
  --tmpfs /tmp:size=64m,noexec,nosuid,nodev \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --env MEDDEID_API_KEY=cuda-validation-only-secret \
  --env MEDDEID_REQUIRE_API_KEY=true \
  "${image}" >/dev/null

docker run --rm \
  --network "${network_name}" \
  --volume "${project_dir}/scripts/container_smoke.py:/smoke.py:ro" \
  --entrypoint python \
  "${image}" \
  /smoke.py \
  --base-url "http://${container_name}:8000" \
  --api-key cuda-validation-only-secret \
  --startup-timeout 180 \
  --request-timeout 180

test "$(docker inspect --format '{{.Config.User}}' "${container_name}")" = "10001:10001"
test "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${container_name}")" = true
printf 'Validated PyTorch CUDA image %s on GPU %s. Evidence: %s\n' \
  "${image}" "${gpu_device_id}" "${report}"
