#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd -P)
cd "${project_dir}"
docker compose -f compose.yaml -f compose.local.yaml down
