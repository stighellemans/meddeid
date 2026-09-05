#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=triton/versions.env
source "${script_dir}/triton/versions.env"

model_dir="${1:-${script_dir}/triton/model_source}"

for command_name in hf python; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${command_name}" >&2
    exit 1
  fi
done
if [[ -e "${model_dir}" ]] && [[ -n "$(find "${model_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing to replace non-empty model directory: %s\n' "${model_dir}" >&2
  exit 1
fi

hf download "${MEDDEID_MODEL_ID}" \
  --revision "${MEDDEID_MODEL_REVISION}" \
  --local-dir "${model_dir}"
hf cache verify "${MEDDEID_MODEL_ID}" \
  --revision "${MEDDEID_MODEL_REVISION}" \
  --local-dir "${model_dir}" \
  --fail-on-missing-files

actual_bundle_sha256="$(python - "${model_dir}" <<'PY'
from pathlib import Path
import sys

from meddeid.bundle import load_model_bundle

bundle = load_model_bundle(Path(sys.argv[1]) / "bundle.json", validate_package=True)
print(bundle.contract_hash())
PY
)"
if [[ "${actual_bundle_sha256}" != "${MEDDEID_BUNDLE_SHA256}" ]]; then
  printf 'Bundle contract mismatch: expected %s, got %s.\n' \
    "${MEDDEID_BUNDLE_SHA256}" "${actual_bundle_sha256}" >&2
  exit 1
fi

printf 'Staged verified model %s@%s at %s\n' \
  "${MEDDEID_MODEL_ID}" "${MEDDEID_MODEL_REVISION}" "$(cd -- "${model_dir}" && pwd -P)"
