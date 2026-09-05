#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"
cd "${project_dir}"

repository="${1:?usage: build_triton_image.sh MODEL_REPOSITORY IMAGE TARGET}"
image="${2:?usage: build_triton_image.sh MODEL_REPOSITORY IMAGE TARGET}"
gpu_target="${3:?usage: build_triton_image.sh MODEL_REPOSITORY IMAGE TARGET}"
package_version="$(python deploy/read_project_version.py)"
repository="$(cd -- "${repository}" && pwd -P)"
manifest="${repository}/build-manifest.json"

if [[ ! -s "${manifest}" ]]; then
  printf 'Missing build manifest: %s\n' "${manifest}" >&2
  exit 1
fi

python "${script_dir}/triton_targets.py" verify-manifest \
  "${gpu_target}" "${manifest}"

python - "${manifest}" "${gpu_target}" "${MEDDEID_MODEL_REVISION}" "${MEDDEID_BUNDLE_SHA256}" "${MEDDEID_TRITON_STACK}" "${MEDDEID_TENSORRT_VERSION}" "${MEDDEID_TENSORRT_BUILDER_IMAGE}" "${MEDDEID_TRT_PRECISION}" "${MEDDEID_TRT_OUTPUT_PRECISION}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
repository = manifest_path.parent
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "target.id": (payload["target"]["id"], sys.argv[2]),
    "model.revision": (payload["model"]["revision"], sys.argv[3]),
    "model.bundle_sha256": (payload["model"]["bundle_sha256"], sys.argv[4]),
    "runtime.triton_stack": (payload["runtime"]["triton_stack"], sys.argv[5]),
    "runtime.tensorrt_version": (payload["runtime"]["tensorrt_version"], sys.argv[6]),
    "runtime.builder_image": (payload["runtime"]["builder_image"], sys.argv[7]),
    "target.precision": (payload["target"]["precision"], sys.argv[8]),
    "target.output_precision": (payload["target"]["output_precision"], sys.argv[9]),
}
for field, (actual, wanted) in expected.items():
    if actual != wanted:
        raise SystemExit(f"{field} mismatch: manifest has {actual!r}, expected {wanted!r}")
def iter_artifacts(node, prefix="artifacts"):
    if not isinstance(node, dict):
        raise SystemExit(f"{prefix} must be an object")
    if {"path", "sha256", "bytes"}.issubset(node):
        yield prefix, node
        return
    if not node:
        raise SystemExit(f"{prefix} cannot be empty")
    for name, value in node.items():
        yield from iter_artifacts(value, f"{prefix}.{name}")


for name, artifact in iter_artifacts(payload["artifacts"]):
    path = (repository / artifact["path"]).resolve()
    if repository not in path.parents or not path.is_file():
        raise SystemExit(f"{name} artifact path is invalid: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != artifact["sha256"] or path.stat().st_size != artifact["bytes"]:
        raise SystemExit(f"{name} artifact no longer matches the build manifest")
PY

manifest_sha256="$(python - "${manifest}" <<'PY'
import hashlib
from pathlib import Path
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"
vcs_ref="${VCS_REF:-$(git rev-parse HEAD 2>/dev/null || printf unknown)}"
build_date="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
runtime_image="${MEDDEID_TRITON_RUNTIME_IMAGE:-meddeid-triton-runtime:${MEDDEID_TRITON_STACK}-tensorrt-only}"
if ! docker image inspect "${runtime_image}" >/dev/null 2>&1; then
  "${script_dir}/build_slim_triton_runtime.sh" "${runtime_image}"
fi
docker build \
  --file deploy/triton-model.Dockerfile \
  --build-context "model_repository=${repository}" \
  --build-arg "TRITON_BASE_IMAGE=${runtime_image}" \
  --build-arg "MODEL_REVISION=${MEDDEID_MODEL_REVISION}" \
  --build-arg "BUNDLE_SHA256=${MEDDEID_BUNDLE_SHA256}" \
  --build-arg "GPU_TARGET=${gpu_target}" \
  --build-arg "TRITON_STACK=${MEDDEID_TRITON_STACK}" \
  --build-arg "TENSORRT_VERSION=${MEDDEID_TENSORRT_VERSION}" \
  --build-arg "TRITON_FULL_IMAGE=${MEDDEID_TRITON_BASE_IMAGE}" \
  --build-arg "TRITON_MIN_IMAGE=${MEDDEID_TRITON_MIN_IMAGE}" \
  --build-arg "TRITON_COMPOSE_REVISION=${MEDDEID_TRITON_COMPOSE_REVISION}" \
  --build-arg "TRT_PRECISION=${MEDDEID_TRT_PRECISION}" \
  --build-arg "TRT_OUTPUT_PRECISION=${MEDDEID_TRT_OUTPUT_PRECISION}" \
  --build-arg "BUILD_MANIFEST_SHA256=${manifest_sha256}" \
  --build-arg "MEDDEID_VERSION=${package_version}" \
  --build-arg "VCS_REF=${vcs_ref}" \
  --build-arg "BUILD_DATE=${build_date}" \
  --tag "${image}" \
  .

docker image inspect "${image}" --format '{{json .RepoDigests}} {{json .Config.Labels}}'
printf 'Built target-specific Triton image: %s\n' "${image}"
