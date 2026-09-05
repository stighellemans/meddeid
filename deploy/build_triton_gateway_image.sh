#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"
cd "${project_dir}"

image="${1:?usage: build_triton_gateway_image.sh IMAGE}"
package_version="$(python deploy/read_project_version.py)"
vcs_ref="${VCS_REF:-$(git rev-parse HEAD 2>/dev/null || printf unknown)}"
build_date="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

docker build \
  --file deploy/triton-gateway.Dockerfile \
  --build-arg "MEDDEID_MODEL_ID=${MEDDEID_MODEL_ID}" \
  --build-arg "MEDDEID_MODEL_REVISION=${MEDDEID_MODEL_REVISION}" \
  --build-arg "MEDDEID_BUNDLE_SHA256=${MEDDEID_BUNDLE_SHA256}" \
  --build-arg "TRITON_CLIENT_VERSION=${MEDDEID_TRITON_SERVER_VERSION}" \
  --build-arg "MEDDEID_VERSION=${package_version}" \
  --build-arg "VCS_REF=${vcs_ref}" \
  --build-arg "BUILD_DATE=${build_date}" \
  --tag "${image}" \
  .

docker image inspect "${image}" \
  --format '{{.Id}} {{.Size}} bytes role={{index .Config.Labels "io.meddeid.runtime-role"}} torch={{index .Config.Labels "io.meddeid.torch-included"}}'
