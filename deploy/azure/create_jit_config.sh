#!/usr/bin/env bash
set -euo pipefail

# Run on the trusted maintainer machine. The output is a short-lived secret.
output_path="${1:?usage: create_jit_config.sh OUTPUT_PATH [RUNNER_NAME]}"
runner_name="${2:-meddeid-t4-release}"
if [[ -e "${output_path}" ]]; then
  printf 'Refusing to overwrite %s\n' "${output_path}" >&2
  exit 1
fi
command -v gh >/dev/null 2>&1 || {
  printf 'Missing GitHub CLI.\n' >&2
  exit 1
}

umask 077
gh api --method POST \
  repos/stighellemans/meddeid/actions/runners/generate-jitconfig \
  --raw-field "name=${runner_name}" \
  --field runner_group_id=1 \
  --raw-field 'labels[]=self-hosted' \
  --raw-field 'labels[]=linux' \
  --raw-field 'labels[]=x64' \
  --raw-field 'labels[]=nvidia' \
  --raw-field 'labels[]=t4-sm75' \
  --raw-field 'work_folder=_work' \
  --jq .encoded_jit_config >"${output_path}"
chmod 600 "${output_path}"
test -s "${output_path}"
printf 'Wrote one-job JIT config to %s; transfer it securely and delete it after use.\n' \
  "${output_path}"
