#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd -P)
cd "${project_dir}"

if ! command -v docker >/dev/null 2>&1; then
  printf '%s\n' "Docker is required. Install Docker Desktop, start it, and run this command again." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  printf '%s\n' "Docker is installed but not running. Start Docker Desktop and run this command again." >&2
  exit 1
fi

if [ ! -f .env ]; then
  if command -v openssl >/dev/null 2>&1; then
    generated_key=$(openssl rand -hex 32)
  elif command -v python3 >/dev/null 2>&1; then
    generated_key=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  else
    printf '%s\n' "OpenSSL or Python 3 is required to generate the local API key." >&2
    exit 1
  fi
  umask 077
  {
    printf 'MEDDEID_BIND_ADDRESS=127.0.0.1\n'
    printf 'MEDDEID_PORT=8000\n'
    printf 'MEDDEID_API_KEY=%s\n' "${generated_key}"
    printf 'MEDDEID_REQUIRE_API_KEY=true\n'
    printf 'MEDDEID_DOCS_ENABLED=true\n'
    printf 'MEDDEID_UI_ENABLED=true\n'
  } > .env
  printf '%s\n' "Created a private .env file with a generated API key."
fi

if [ "${MEDDEID_BUILD_LOCAL:-false}" = "true" ]; then
  printf '%s\n' "Building and starting MedDeID from this checkout."
  docker compose up --build --detach
else
  printf '%s\n' "Downloading and starting the published MedDeID image."
  docker compose pull meddeid
  docker compose up --detach --no-build
fi

container_id=$(docker compose ps --quiet meddeid)
if [ -z "${container_id}" ]; then
  printf '%s\n' "MedDeID did not create a container. Run 'docker compose logs' for details." >&2
  exit 1
fi

health_state=starting
attempt=0
while [ "${attempt}" -lt 60 ]; do
  health_state=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "${container_id}")
  if [ "${health_state}" = healthy ]; then
    break
  fi
  if [ "${health_state}" = exited ] || [ "${health_state}" = dead ]; then
    docker compose logs --tail=100
    printf '%s\n' "MedDeID stopped during startup." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 2
done

if [ "${health_state}" != healthy ]; then
  docker compose logs --tail=100
  printf '%s\n' "MedDeID did not become healthy within two minutes." >&2
  exit 1
fi

configured_port=$(sed -n 's/^MEDDEID_PORT=//p' .env | tail -n 1)
configured_port=${configured_port:-8000}
printf '\n%s\n' "MedDeID is ready."
printf '%s\n' "Open http://127.0.0.1:${configured_port}/ui in your browser."
printf '%s\n' "Paste the MEDDEID_API_KEY value from .env into the API key field."
printf '%s\n' "Technical API documentation: http://127.0.0.1:${configured_port}/docs"
printf '%s\n' "Stop the service with: ./scripts/stop-local.sh"
