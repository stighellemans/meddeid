#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=pytorch-cuda/versions.env
source "${script_dir}/pytorch-cuda/versions.env"
cd "${project_dir}"

image="${1:?usage: build_pytorch_cuda_image.sh IMAGE}"
compile_mode="${MEDDEID_TORCH_COMPILE_MODE:-off}"
package_version="$(python deploy/read_project_version.py)"
vcs_ref="${VCS_REF:-$(git rev-parse HEAD 2>/dev/null || printf unknown)}"
build_date="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

docker buildx build \
  --file Dockerfile \
  --platform "${MEDDEID_PYTORCH_PLATFORM}" \
  --load \
  --build-arg "TORCH_VERSION=${MEDDEID_PYTORCH_VERSION}" \
  --build-arg "TORCH_INDEX_URL=${MEDDEID_PYTORCH_INDEX_URL}" \
  --build-arg MEDDEID_ACCELERATOR=cuda \
  --build-arg "MEDDEID_CUDA_VERSION=${MEDDEID_PYTORCH_CUDA_VERSION}" \
  --build-arg MEDDEID_DEVICE=cuda \
  --build-arg MEDDEID_TORCH_PRECISION=fp16 \
  --build-arg "MEDDEID_TORCH_COMPILE_MODE=${compile_mode}" \
  --build-arg "MEDDEID_VERSION=${package_version}" \
  --build-arg "VCS_REF=${vcs_ref}" \
  --build-arg "BUILD_DATE=${build_date}" \
  --tag "${image}" \
  .

docker image inspect "${image}" \
  --format '{{.Id}} accelerator={{index .Config.Labels "io.meddeid.accelerator"}} torch={{index .Config.Labels "io.meddeid.torch-version"}} cuda={{index .Config.Labels "io.meddeid.cuda-version"}}'
