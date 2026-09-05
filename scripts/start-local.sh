#!/bin/sh
set -eu

usage() {
  printf '%s\n' \
    "Usage: ./scripts/start-local.sh [--build] [--model HUB_ID] [--revision REVISION] [--language-profile PROFILE]"
  printf '%s\n' "  --build  Build this checkout instead of pulling the published image."
  printf '%s\n' "  --model  Hub model to serve (required)."
  printf '%s\n' "  --revision  Optional Hub model revision (default: latest)."
  printf '%s\n' "  --language-profile  Default profile (required); the UI can select another supported profile."
}

build_local=${MEDDEID_BUILD_LOCAL:-false}
selected_model=
selected_revision=
selected_profile=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --build)
      build_local=true
      ;;
    --model|--revision|--language-profile)
      if [ "$#" -lt 2 ] || [ -z "$2" ]; then
        printf '%s\n' "Option $1 requires a value." >&2
        usage >&2
        exit 2
      fi
      case "$1" in
        --model) selected_model=$2 ;;
        --revision) selected_revision=$2 ;;
        --language-profile) selected_profile=$2 ;;
      esac
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '%s\n' "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [ -z "${selected_model}" ] || [ -z "${selected_profile}" ]; then
  printf '%s\n' \
    "Select both --model and --language-profile; run 'meddeid models' to review the public choices." >&2
  usage >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd -P)
cd "${project_dir}"

compose() {
  docker compose -f compose.yaml -f compose.local.yaml "$@"
}

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

MEDDEID_LANGUAGE_PROFILE=${selected_profile}
export MEDDEID_LANGUAGE_PROFILE

if [ "${build_local}" = "true" ]; then
  MEDDEID_API_IMAGE=${MEDDEID_API_IMAGE:-meddeid-api:local}
  MEDDEID_MODEL_ID=${selected_model}
  MEDDEID_MODEL_REVISION=${selected_revision:-main}
  MEDDEID_MODEL=/opt/meddeid-model
  MEDDEID_OFFLINE=true
  export MEDDEID_API_IMAGE MEDDEID_MODEL_ID MEDDEID_MODEL_REVISION
  export MEDDEID_MODEL MEDDEID_OFFLINE
  printf '%s\n' "Building the local development image from this checkout."
  compose build meddeid
  printf '%s\n' "Starting the locally built image ${MEDDEID_API_IMAGE}."
  compose up --detach --no-build meddeid
else
  MEDDEID_MODEL=${selected_model}
  MEDDEID_OFFLINE=false
  MEDDEID_REVISION=${selected_revision}
  export MEDDEID_MODEL MEDDEID_OFFLINE MEDDEID_REVISION
  printf '%s\n' "Downloading and starting the published MedDeID image."
  compose pull meddeid
  compose up --detach --no-build
fi

container_id=$(compose ps --quiet meddeid)
if [ -z "${container_id}" ]; then
  printf '%s\n' \
    "MedDeID did not create a container. Run 'docker compose -f compose.yaml -f compose.local.yaml logs' for details." >&2
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
    compose logs --tail=100
    printf '%s\n' "MedDeID stopped during startup." >&2
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 2
done

if [ "${health_state}" != healthy ]; then
  compose logs --tail=100
  printf '%s\n' "MedDeID did not become healthy within two minutes." >&2
  exit 1
fi

configured_port=$(sed -n 's/^MEDDEID_PORT=//p' .env | tail -n 1)
configured_port=${configured_port:-8000}
configured_key=$(sed -n 's/^MEDDEID_API_KEY=//p' .env | tail -n 1)
if [ -z "${configured_key}" ]; then
  printf '%s\n' "MEDDEID_API_KEY is missing from .env." >&2
  exit 1
fi

key_copied=false
copy_key_command=
if [ "${MEDDEID_COPY_API_KEY:-true}" = true ]; then
  if command -v pbcopy >/dev/null 2>&1 && printf '%s' "${configured_key}" | pbcopy; then
    key_copied=true
    copy_key_command="sed -n 's/^MEDDEID_API_KEY=//p' .env | pbcopy"
  elif command -v wl-copy >/dev/null 2>&1 && printf '%s' "${configured_key}" | wl-copy 2>/dev/null; then
    key_copied=true
    copy_key_command="sed -n 's/^MEDDEID_API_KEY=//p' .env | wl-copy"
  elif command -v xclip >/dev/null 2>&1 && printf '%s' "${configured_key}" | xclip -selection clipboard 2>/dev/null; then
    key_copied=true
    copy_key_command="sed -n 's/^MEDDEID_API_KEY=//p' .env | xclip -selection clipboard"
  elif command -v xsel >/dev/null 2>&1 && printf '%s' "${configured_key}" | xsel --clipboard --input 2>/dev/null; then
    key_copied=true
    copy_key_command="sed -n 's/^MEDDEID_API_KEY=//p' .env | xsel --clipboard --input"
  elif command -v clip.exe >/dev/null 2>&1 && printf '%s' "${configured_key}" | clip.exe; then
    key_copied=true
    copy_key_command="sed -n 's/^MEDDEID_API_KEY=//p' .env | clip.exe"
  fi
fi

ui_url="http://127.0.0.1:${configured_port}/ui"
browser_url=${ui_url}
if command -v python3 >/dev/null 2>&1; then
  encoded_key=$(printf '%s' "${configured_key}" \
    | python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.stdin.read(), safe=""))')
  browser_url="${ui_url}#api_key=${encoded_key}"
fi
browser_opened=false
if [ "${MEDDEID_OPEN_BROWSER:-true}" = true ]; then
  if command -v open >/dev/null 2>&1 && open "${browser_url}" >/dev/null 2>&1; then
    browser_opened=true
  elif command -v xdg-open >/dev/null 2>&1 && xdg-open "${browser_url}" >/dev/null 2>&1; then
    browser_opened=true
  elif command -v wslview >/dev/null 2>&1 && wslview "${browser_url}" >/dev/null 2>&1; then
    browser_opened=true
  fi
fi

printf '\n%s\n' "MedDeID is ready."
if [ "${browser_opened}" = true ]; then
  printf '%s\n' "Opened ${ui_url} in your browser."
else
  printf '%s\n' "Open ${ui_url} in your browser."
fi
if [ "${key_copied}" = true ]; then
  printf '%s\n' "The local API key is on your clipboard. Paste it into the API key field."
  printf '%s\n' "If the clipboard changes, copy the key again with:"
  printf '  %s\n' "${copy_key_command}"
else
  printf '%s\n' "Show the local API key when ready to paste:"
  printf '%s\n' "  sed -n 's/^MEDDEID_API_KEY=//p' .env"
fi
printf '%s\n' "Technical API documentation: http://127.0.0.1:${configured_port}/docs"
printf '%s\n' "Stop the service with: ./scripts/stop-local.sh"
