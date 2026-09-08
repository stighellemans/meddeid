#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_dir="$(cd -- "${script_dir}/.." && pwd -P)"

# Preserve the public deployment choices before loading release defaults.
: "${MEDDEID_MODEL:?set MEDDEID_MODEL to a Hugging Face model ID}"
: "${MEDDEID_REVISION:=}"
: "${MEDDEID_LANGUAGE_PROFILE:?set MEDDEID_LANGUAGE_PROFILE}"
revision_args=()
if [[ -n "${MEDDEID_LOCAL_MODEL:-}" ]]; then
  revision_args=(--local-model)
fi
MEDDEID_REVISION="$(python -m meddeid.triton_artifact check-revision \
  --model "${MEDDEID_MODEL}" --revision "${MEDDEID_REVISION}" \
  "${revision_args[@]}")"
export MEDDEID_MODEL_ID="${MEDDEID_MODEL}"
export MEDDEID_MODEL_REVISION="${MEDDEID_REVISION}"
export MEDDEID_LANGUAGE_PROFILES="${MEDDEID_LANGUAGE_PROFILE}"
# Normal local builds choose one of two families automatically. The low-level
# release builder retains native mode for previously published exact-GPU plans.
export MEDDEID_TRT_HARDWARE_COMPATIBILITY=auto
# A local build accepts any compatible MedDeID bundle. The published model hash
# in versions.env is only a release default, so do not inherit it here.
: "${MEDDEID_BUNDLE_SHA256:=}"

# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"

gpu_device_id="${MEDDEID_GPU_DEVICE_ID:-0}"
gpu_name="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
compute_capability="$(nvidia-smi --id="${gpu_device_id}" --query-gpu=compute_cap --format=csv,noheader | head -n 1 | xargs)"
target="${MEDDEID_GPU_TARGET:-}"
if [[ -z "${target}" ]]; then
  target="$(python "${script_dir}/triton_targets.py" detect-host \
    --gpu-name "${gpu_name}" \
    --compute-capability "${compute_capability}" \
    --allow-local)"
fi
output_repository="${MEDDEID_TRITON_BUILD_OUTPUT:-/output/model_repository}"
temporary_source="$(mktemp -d /tmp/meddeid-model-source-XXXXXX)"
cleanup() {
  rm -rf -- "${temporary_source}"
}
trap cleanup EXIT

if [[ -n "${MEDDEID_LOCAL_MODEL:-}" ]]; then
  model_source="${MEDDEID_LOCAL_MODEL}"
  if [[ ! -f "${model_source}/bundle.json" ]]; then
    printf 'Local model does not contain bundle.json: %s\n' "${model_source}" >&2
    exit 1
  fi
else
  model_source="${temporary_source}"
  "${script_dir}/stage_triton_model.sh" "${model_source}"
fi

bundle_values="$(python - "${model_source}" <<'PY'
from pathlib import Path
import sys

from meddeid.bundle import load_model_bundle

bundle = load_model_bundle(Path(sys.argv[1]) / "bundle.json", validate_package=True)
print(bundle.contract_hash())
print(",".join(profile.profile_id for profile in bundle.postprocess.profiles))
PY
)"
actual_bundle_sha256="$(printf '%s\n' "${bundle_values}" | sed -n '1p')"
bundle_profiles="$(printf '%s\n' "${bundle_values}" | sed -n '2p')"
if [[ -n "${MEDDEID_BUNDLE_SHA256}" ]] && \
   [[ "${MEDDEID_BUNDLE_SHA256}" != "${actual_bundle_sha256}" ]]; then
  printf 'Bundle contract mismatch: expected %s, got %s.\n' \
    "${MEDDEID_BUNDLE_SHA256}" "${actual_bundle_sha256}" >&2
  exit 1
fi
if [[ ",${bundle_profiles}," != *",${MEDDEID_LANGUAGE_PROFILE},"* ]]; then
  printf 'Model does not support language profile %s; available: %s\n' \
    "${MEDDEID_LANGUAGE_PROFILE}" "${bundle_profiles}" >&2
  exit 1
fi
export MEDDEID_BUNDLE_SHA256="${actual_bundle_sha256}"
export MEDDEID_LANGUAGE_PROFILES="${bundle_profiles}"

"${script_dir}/build_triton_repository.sh" \
  "${model_source}" \
  "${output_repository}" \
  "${target}" \
  "${gpu_device_id}"

python -m meddeid.triton_artifact verify "${output_repository}"
printf 'Local Triton plan is ready for %s@%s on %s (%s).\n' \
  "${MEDDEID_MODEL}" "${MEDDEID_REVISION}" "${gpu_name}" "${target}"
printf 'The plan stays on this device. Nothing has been published to the internet.\n'
