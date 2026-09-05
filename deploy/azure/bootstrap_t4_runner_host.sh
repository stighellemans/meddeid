#!/usr/bin/env bash
set -euo pipefail

# Run as root on a new Ubuntu 24.04 Standard_NC4as_T4_v3 VM after installing
# the Microsoft-supported NVIDIA driver. This installs only release-runner
# tooling; GitHub registration material is deliberately handled separately.

if [[ "$(id -u)" -ne 0 ]]; then
  printf 'Run this bootstrap as root.\n' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'Install and verify the supported Azure NVIDIA driver first.\n' >&2
  exit 1
fi

runner_version=2.337.0
runner_sha256=70920811a4f8ad4328818682bca5c6469c1c942fab52448868071d0063816613
runner_archive="actions-runner-linux-x64-${runner_version}.tar.gz"
runner_url="https://github.com/actions/runner/releases/download/v${runner_version}/${runner_archive}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends ca-certificates curl gnupg jq
install -m 0755 -d /etc/apt/keyrings

curl --fail --location --silent --show-error \
  https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
printf '%s\n' \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  >/etc/apt/sources.list.d/docker.list

curl --fail --location --silent --show-error \
  https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --fail --location --silent --show-error \
  "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  >/etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install --yes --no-install-recommends \
  containerd.io docker-buildx-plugin docker-ce docker-ce-cli \
  docker-compose-plugin git nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

if ! id actions >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash actions
fi
usermod --append --groups docker actions
install -d -o actions -g actions /opt/actions-runner
curl --fail --location --silent --show-error "${runner_url}" \
  -o "/tmp/${runner_archive}"
printf '%s  %s\n' "${runner_sha256}" "/tmp/${runner_archive}" | sha256sum --check
tar -xzf "/tmp/${runner_archive}" -C /opt/actions-runner
rm "/tmp/${runner_archive}"
chown -R actions:actions /opt/actions-runner
install -o root -g root -m 0444 /dev/null \
  /opt/actions-runner/.meddeid-dedicated-release-runner
/opt/actions-runner/bin/installdependencies.sh

nvidia-smi
docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi
docker version
docker buildx version
docker compose version
nvidia-container-cli --version
printf 'T4 release-runner host is ready; create and transfer a one-job JIT config next.\n'
