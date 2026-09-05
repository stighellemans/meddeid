#!/usr/bin/env bash
set -euo pipefail

# Run as the unprivileged actions user on the clean T4 VM. The file is removed
# before the runner starts and the JIT configuration accepts at most one job.
config_path="${1:?usage: run_ephemeral_runner.sh JIT_CONFIG_FILE}"
if [[ "$(id -un)" != actions ]]; then
  printf 'Run this command as the actions user.\n' >&2
  exit 1
fi
if [[ ! -s "${config_path}" ]]; then
  printf 'JIT configuration is missing or empty: %s\n' "${config_path}" >&2
  exit 1
fi
if [[ ! -f /opt/actions-runner/.meddeid-dedicated-release-runner ]]; then
  printf 'Refusing cleanup: this is not a dedicated MedDeID release runner.\n' >&2
  exit 1
fi
IFS= read -r encoded_config <"${config_path}"
if command -v shred >/dev/null 2>&1; then
  shred --remove "${config_path}"
else
  rm -f "${config_path}"
fi
cd /opt/actions-runner
# A new JIT registration must not inherit a checkout, container, image, volume,
# or builder cache from the preceding release gate. The sentinel above limits
# this destructive cleanup to a host created by the dedicated bootstrap.
find /opt/actions-runner/_work -xdev -depth -mindepth 1 -delete 2>/dev/null || true
docker ps --all --quiet | xargs --no-run-if-empty docker rm --force
docker system prune --all --force --volumes
exec ./run.sh --jitconfig "${encoded_config}"
