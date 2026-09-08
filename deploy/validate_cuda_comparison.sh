#!/usr/bin/env bash
# Compare or benchmark an already-built CUDA image. Never builds an engine/image.
set -euo pipefail
mode="${1:?usage: validate_cuda_comparison.sh parity|benchmark baseline|optimized [parity options]}"
variant="${2:?specify baseline or optimized}"
shift 2
case "$mode" in parity|benchmark) ;; *) exit 2 ;; esac
case "$variant" in
  baseline) precision=fp32; profile=latency; concurrent=8 ;;
  optimized) precision=fp32; profile=throughput; concurrent=16 ;;
  fp16-throughput) precision=fp16; profile=throughput; concurrent=16 ;;
  fp16-latency) precision=fp16; profile=latency; concurrent=8 ;;
  fp32-throughput) precision=fp32; profile=throughput; concurrent=16 ;;
  *) exit 2 ;;
esac
: "${CANDIDATE_CUDA_IMAGE:?set the exact candidate image}"
: "${MEDDEID_BENCHMARK_FILE:?set the pinned fixture}"
: "${MEDDEID_LANGUAGE_PROFILE:?set the selected catalog language profile}"
: "${MEDDEID_API_KEY:?set the validation API key}"
if [[ "$variant" == optimized ]]; then
  # The release check must not hide an image that still ships the old default.
  image_config=$(docker image inspect "$CANDIDATE_CUDA_IMAGE" --format '{{json .Config}}')
  python -c 'import json,sys; c=json.loads(sys.argv[1]); assert "MEDDEID_TORCH_PRECISION=fp32" in c["Env"], "CUDA image default is not fp32"; assert c["Labels"]["io.meddeid.torch-precision"] == "fp32", "CUDA image precision label is not fp32"' "$image_config"
fi
output_dir="${MEDDEID_COMPARISON_OUTPUT_DIR:-deploy/triton}"
mkdir -p "$output_dir"
prefix="$output_dir/pytorch-cuda-$variant"
container="meddeid-cuda-benchmark"
monitor_pid=""
cleanup() {
  result=$?
  trap - EXIT
  if [[ -n "$monitor_pid" ]]; then
    kill -INT "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" || true
  fi
  docker logs "$container" > "$prefix-$mode.log" 2>&1 || true
  docker inspect "$container" > "$prefix-$mode-inspect.json" 2>/dev/null || true
  docker rm --force "$container" >/dev/null 2>&1 || true
  exit "$result"
}
# A stale container indicates an unsafe overlapping invocation; do not delete it.
if docker container inspect "$container" >/dev/null 2>&1; then
  printf 'Comparison container already exists; inspect it before retrying.\n' >&2
  exit 1
fi
trap cleanup EXIT
started_at_ns=$(date +%s%N)
docker run --detach --name "$container" \
  --gpus "device=${MEDDEID_GPU_DEVICE_ID:-0}" \
  --read-only --tmpfs /tmp:size=64m,mode=1777,noexec,nosuid,nodev \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --publish 127.0.0.1:8002:8000 \
  --volume "${MEDDEID_TRITON_SOURCE_MODEL_DIR:-${PWD}/deploy/triton/model_source}:/model-source:ro" \
  --env MEDDEID_MODEL=/model-source --env MEDDEID_REVISION= \
  --env MEDDEID_LOCAL_BUNDLE=true --env MEDDEID_OFFLINE=true \
  --env MEDDEID_LANGUAGE_PROFILE="$MEDDEID_LANGUAGE_PROFILE" \
  --env MEDDEID_BACKEND=torch --env MEDDEID_DEVICE=cuda \
  --env MEDDEID_API_KEY="$MEDDEID_API_KEY" --env MEDDEID_REQUIRE_API_KEY=true \
  --env MEDDEID_TORCH_PRECISION="$precision" --env MEDDEID_TORCH_COMPILE_MODE=off \
  --env MEDDEID_SERVING_PROFILE="$profile" --env MEDDEID_MICROBATCH_ENABLED=auto \
  --env MEDDEID_SEQUENCE_LENGTH_BUCKETS=off --env MEDDEID_MICROBATCH_MAX_WAIT_MS=1 \
  --env MEDDEID_MAX_CONCURRENT_REQUESTS="$concurrent" \
  "$CANDIDATE_CUDA_IMAGE"
python deploy/wait_for_health.py --base-url http://127.0.0.1:8002 \
  --container "$container" --started-at-ns "$started_at_ns" \
  --output "$prefix-$mode-startup.json"
if [[ "$mode" == parity ]]; then
  report="$prefix-parity-report.json"
  # Retain the existing evidence filename for release consumers.
  if [[ "$variant" == optimized ]]; then report="$output_dir/pytorch-cuda-parity-report.json"; fi
  python deploy/validate_triton_parity.py "$MEDDEID_BENCHMARK_FILE" \
    --candidate-url http://127.0.0.1:8002 --reference-url http://127.0.0.1:8001 \
    --fail-fast --output "$report" "$@"
else
  python deploy/monitor_nvidia.py --output "$prefix-gpu-metrics.json" &
  monitor_pid=$!
  python deploy/benchmark_http.py "$MEDDEID_BENCHMARK_FILE" \
    --base-url http://127.0.0.1:8002 --api-key "$MEDDEID_API_KEY" \
    --batch-size 16 --warmup-requests 8 --requests 60 --concurrency 8 \
    --output "$prefix-benchmark.json"
fi
