#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"

if ! python3 -c 'import distro, requests' >/dev/null 2>&1; then
  printf '%s\n' \
    "The pinned NVIDIA Triton composer requires the Python 'distro' and 'requests' packages." \
    "Install them in the active environment with: python -m pip install distro requests" >&2
  exit 1
fi

output_image="${1:-meddeid-triton-runtime:${MEDDEID_TRITON_STACK}-tensorrt-only}"
composed_image="meddeid-triton-runtime:${MEDDEID_TRITON_STACK}-official-compose"
checkout_dir="$(mktemp -d)"
compose_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "${checkout_dir}" "${compose_dir}"
}
trap cleanup EXIT

git -C "${checkout_dir}" init --quiet
git -C "${checkout_dir}" remote add origin \
  https://github.com/triton-inference-server/server.git
git -C "${checkout_dir}" fetch --quiet --depth=1 origin \
  "${MEDDEID_TRITON_COMPOSE_REVISION}"
git -C "${checkout_dir}" checkout --quiet --detach FETCH_HEAD
test "$(git -C "${checkout_dir}" rev-parse HEAD)" = \
  "${MEDDEID_TRITON_COMPOSE_REVISION}"

(
  cd "${checkout_dir}"
  python3 compose.py \
    --backend tensorrt \
    --image "min,${MEDDEID_TRITON_MIN_IMAGE}" \
    --image "full,${MEDDEID_TRITON_BASE_IMAGE}" \
    --work-dir "${compose_dir}" \
    --output-name "${composed_image}"
)

docker build \
  --file "${script_dir}/triton-runtime.Dockerfile" \
  --build-arg "TRITON_COMPOSED_IMAGE=${composed_image}" \
  --build-arg "CUDA_BASE_IMAGE=${MEDDEID_CUDA_BASE_IMAGE}" \
  --build-arg "TRITON_STACK=${MEDDEID_TRITON_STACK}" \
  --build-arg "TRITON_SERVER_VERSION=${MEDDEID_TRITON_SERVER_VERSION}" \
  --build-arg "TENSORRT_VERSION=${MEDDEID_TENSORRT_VERSION}" \
  --build-arg "TRITON_FULL_IMAGE=${MEDDEID_TRITON_BASE_IMAGE}" \
  --build-arg "TRITON_MIN_IMAGE=${MEDDEID_TRITON_MIN_IMAGE}" \
  --build-arg "TRITON_COMPOSE_REVISION=${MEDDEID_TRITON_COMPOSE_REVISION}" \
  --tag "${output_image}" \
  "${script_dir}/.."

docker run --rm --entrypoint /bin/bash "${output_image}" -ceu '
  test -x /opt/tritonserver/bin/tritonserver
  test -d /opt/tritonserver/backends/tensorrt
  test "$(find /opt/tritonserver/backends -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1
  ! find /usr/lib/x86_64-linux-gnu -name "libnvinfer_builder_resource*" -print -quit | grep -q .
  ! command -v nvcc
'
docker image inspect "${output_image}" --format '{{.Id}} {{.Size}} bytes'
printf 'Built TensorRT-only Triton runtime: %s\n' "${output_image}"
