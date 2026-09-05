#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=triton/benchmark.env
source "${script_dir}/triton/benchmark.env"

output_dir="${1:?usage: stage_benchmark_fixture.sh OUTPUT_DIRECTORY}"
if [[ -e "${output_dir}" ]] && \
   [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  printf 'Refusing to replace non-empty benchmark directory: %s\n' "${output_dir}" >&2
  exit 1
fi
mkdir -p -- "${output_dir}"

hf download "${MEDDEID_BENCHMARK_DATASET_ID}" \
  --repo-type dataset \
  --revision "${MEDDEID_BENCHMARK_DATASET_REVISION}" \
  --local-dir "${output_dir}"
# `hf download --local-dir` writes transfer metadata beneath the destination.
# It is not part of the dataset revision and must not weaken the exact-file gate.
find "${output_dir}/.cache" -depth -delete
hf cache verify "${MEDDEID_BENCHMARK_DATASET_ID}" \
  --repo-type dataset \
  --revision "${MEDDEID_BENCHMARK_DATASET_REVISION}" \
  --local-dir "${output_dir}" \
  --fail-on-missing-files \
  --fail-on-extra-files
test -s "${output_dir}/${MEDDEID_BENCHMARK_FIXTURE}"
printf 'Verified synthetic benchmark fixture: %s/%s\n' \
  "${output_dir}" "${MEDDEID_BENCHMARK_FIXTURE}"
