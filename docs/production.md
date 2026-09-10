# Production deployment

This guide is for an operator deploying MedDeID inside an approved clinical
data boundary. Local processing reduces data movement; it does not make model
output anonymous or remove the need for institutional validation.

## Production architecture

A production MedDeID service runs behind an organization-managed reverse proxy
or private service mesh. Choose CPU or GPU based on the workload and available
infrastructure:

```text
approved client -> TLS/auth/rate limits -> MedDeID API -> local CPU or GPU model
```

The published CPU and CUDA images contain no model weights. They download the
selected model revision once into a persistent cache, or load a mounted local
bundle. Set `MEDDEID_OFFLINE=true` after staging the model when startup may not
contact the Hub. MedDeID provides API-key authentication and workload limits. The surrounding platform
must provide TLS, client identity where required, network policy, rate limits,
central secret storage, monitoring, backup policy, and incident response.

## Choose a deployment path

Start with the hardware the organization will operate and the traffic the
service must handle. The runtime changes, but the MedDeID API, model contract,
language profiles, and post-processing remain the same.

| Situation | Choose | Current status |
|---|---|---|
| Simplest shared service, or no accelerator | Published CPU API image with `compose.yaml` | Published for AMD64 and ARM64 |
| NVIDIA GPU model may vary between hosts | CUDA-tagged PyTorch API image with `compose.cuda.yaml` | AMD64 release candidate with a completed T4 validation path |
| Fixed NVIDIA T4 or Ampere-and-newer deployment | Weight-free Triton runtime and gateway with a separately compiled plan | Separate T4 and Ampere+ plans are supplied for both public models |
| Native Apple-silicon service | PyTorch MPS from the Python installation | Validated on one M4 Pro; Linux containers cannot use the host Metal device |

The CUDA, Triton runtime, and gateway images are published products only after
their GPU release gates pass and their immutable digests are recorded. The
model-specific TensorRT plan is distributed separately. The Ampere+ plan uses
TensorRT's hardware-compatibility mode for compute capability 8.0 and newer;
the exact T4 plan remains separate.

### Measured T4 snapshot

The earlier FP16 candidate images were measured end to end on one Azure
`Standard_NC4as_T4_v3` VM (4 vCPU, Tesla T4 16 GiB). The benchmark used the
pinned public synthetic fixture at revision
`9b95ebbfb091a5390e4fc39e2ef74e7580aac068`. Each throughput value is the
median of three warm runs at batch size 16. CPU used concurrency 1; the GPU
paths used concurrency 8 and processed 948 documents and 1,307,107 characters
per run.

| Deployment | Documents/s | HTTP p50 / p95 | Peak GPU memory |
|---|---:|---:|---:|
| PyTorch CPU | 1.85 | 6.78 / 15.74 s | — |
| PyTorch CUDA, FP16 | 65.0 | 1.87 / 2.49 s | 1,655 MiB |
| T4 TensorRT | 168.8 | 0.725 / 1.418 s | 911 MiB |

These historical measurements do not establish the performance or correctness
of a repaired FP16 candidate. Fresh comparison reports and benchmark evidence are required.
Latency includes queueing under the stated concurrency.
An exact semantic comparison of CPU and TensorRT output passed for all 300
pinned fixture documents with zero differences.
Treat these as comparative evidence for this host, not a capacity promise or a
like-for-like rerun of an older release.

Each runtime has a checked transfer and unpacked-size ceiling in
`deploy/image-size-budgets.json`. CPU CI and both GPU gates fail if those
ceilings are exceeded. They also inspect installed modules so CPU cannot gain
CUDA/TensorRT/ONNX runtimes, CUDA cannot gain TensorRT/ONNX or compiler-only
Triton, and the gateway cannot gain PyTorch/TensorRT/ONNX. These are separate
downloads; choosing one deployment never downloads the other hardware paths.

### Measured Apple MPS snapshot

Native FP32 eager inference was measured against native CPU on an Apple M4 Pro
MacBook Pro with 14 CPU cores, 20 GPU cores, 48 GB unified memory, macOS
26.6.2, and PyTorch 2.13.0. Both paths used one worker and 32-window batches.
Each row is the median of three runs; latency includes queueing at the stated
concurrency.

| HTTP workload | MPS documents/s, p50 / p95 | CPU documents/s, p50 / p95 | MPS speedup |
|---|---:|---:|---:|
| Short note, concurrency 1 | 92.83, 9.4 / 15.2 ms | 56.79, 15.1 / 25.7 ms | 1.63× |
| Mixed note, concurrency 1 | 50.26, 20.3 / 25.1 ms | 25.44, 39.7 / 49.9 ms | 1.98× |
| Short-note burst, concurrency 8 | 113.70, 66.4 / 90.2 ms | 60.13, 130.9 / 140.5 ms | 1.89× |
| Batch 16, concurrency 8 | 40.08, 3.17 / 3.66 s | 21.11, 6.06 / 6.99 s | 1.90× |
| Batch 32, concurrency 8 | 39.51, 5.89 / 6.73 s | 20.50, 11.47 / 12.88 s | 1.93× |
| Long-note ETL, concurrency 4 | 3.59 and 45,524 chars/s, 3.33 / 3.38 s | 2.11 and 26,728 chars/s, 5.68 / 5.72 s | 1.70× |

MPS matched native CPU semantics on all 300 pinned public fixture documents,
excluding confidence fields under the same parity policy as TensorRT. A single
cached startup observation was 12.1 seconds for MPS and 17.7 seconds for CPU.
The long-note MPS run reached a 2,762 MB peak physical footprint including
unified graphics allocations.

For sustained concurrent traffic, the throughput profile now automatically
uses the same bounded 1 ms PyTorch microbatcher on MPS as on CUDA. It increased
the MPS short-note burst from 113.70 to 167.05 documents/s and batch-16 ETL
from 40.08 to 42.10 documents/s, while isolated short-note p50 latency was
26.06 ms instead of 9.4 ms. Keep the latency profile for isolated calls.
Dynamic compilation was rejected as a default: it added 6.44 seconds for the
first short graph and 7.39 seconds for the first batch-16 graph, then reduced
matched eager throughput. The complete reproducible record is in
[`deploy/mps`](../deploy/mps/README.md).

MPS is a native macOS path. Ordinary Linux containers on Docker Desktop cannot
use the host Metal device, so external Mac users should install
`meddeid[server]`; automatic selection uses MPS when CUDA is absent and MPS is
available. `MEDDEID_DEVICE=mps` remains an explicit diagnostic override, not
required setup. Do not publish an “MPS Docker image.” Batch 16 is the starting
ETL request size on this host because batch 32 did not improve throughput and
roughly doubled queued latency.

The model is not included in the CPU or CUDA images. The measured CUDA virtual
environment was dominated by the CUDA/NVIDIA runtime and PyTorch. The compiler
package, headers,
tests, and static CUDA archives are absent. The TensorRT path keeps its compiled
plan outside the runtime and gateway images.

CPU is the simplest choice for occasional traffic. Use PyTorch CUDA when the
NVIDIA GPU class may vary. For a fixed, validated T4, the TensorRT pair is
preferable in this test for throughput: it delivered 2.6 times the CUDA rate.
The plan remains model- and T4-specific.

### Latency and throughput serving profiles

GPU deployments have a process-wide `MEDDEID_SERVING_PROFILE` with two values:

| Profile | PyTorch CUDA | Apple MPS | TensorRT/Triton |
|---|---|---|---|
| `latency` | Eager FP16 inference and request-local window batches, with no cross-request wait | Eager FP32 native inference and request-local window batches, with no cross-request wait | Request-local gateway batches and a Triton config without dynamic queueing |
| `throughput` | A bounded gateway queue coalesces windows from concurrent requests for at most 1 ms | The same bounded 1 ms queue, enabled after the M4 Pro burst and ETL measurements | Four weight-free gateway workers feed request-local batches directly to one Triton model instance; no second queue is enabled by default |

Use `latency` for on-demand de-identification. Use `throughput` together with
`/deidentify-batch` for sustained ETL traffic and one PyTorch worker per
accelerator. `MEDDEID_MAX_CONCURRENT_REQUESTS=auto` selects 1 for latency, 16
for accelerated PyTorch throughput, and 8 per TensorRT gateway worker. The
profile can also be supplied as
`meddeid-server --serving-profile latency|throughput`.

This is the only performance setting in the published GPU environment
templates. The selected image supplies its validated precision, compilation,
window batching, transport, worker, and admission defaults. The remaining
environment controls later in this guide are advanced overrides, not required
setup; changing them creates a new deployment configuration that should be
benchmarked again.

The profile is intentionally not selectable per request because batching and
compiled/runtime scheduler state is shared. Mixed interactive and ETL traffic
should use separate replica pools so bulk work cannot consume the interactive
latency budget. `MEDDEID_MICROBATCH_ENABLED=true` remains an advanced TensorRT
experiment for nested gateway batching; `auto` keeps the measured request-local
TensorRT path. At the selected four-worker count, nested batching improved only
the eight-way short-note burst (11.6%). It was 2–3% slower for batch 16/32, 13%
slower for long-note ETL, and 24–39% slower for isolated short/mixed notes, so it
is an opt-in for a measured small-request fan-out workload rather than the
external-user default. The final note window is always anchored to the end of
the note, so the windowing policy fills a short tail with additional overlap
rather than padding a half-empty final context.

Use these as the two production starting points:

| Workload | Endpoint and profile | Starting request shape | Scaling rule |
|---|---|---|---|
| On-demand, on-the-fly de-identification | `/deidentify`, `latency` | One note per request; one admitted request per worker | Add replicas to meet the latency SLO; do not add a batching delay to a lightly loaded replica |
| Bulk ETL | `/deidentify-batch`, `throughput` | 16 notes per request with sustained concurrency | Keep one PyTorch model worker per GPU; on T4, batch 32 added 3.6% for CUDA but lost 5.5% for the four-worker TensorRT path while increasing latency |

One image can run either profile, so this does not double the artifact set. Two
simultaneous pools do add routing, capacity, and monitoring work, but avoid
head-of-line blocking from long ETL batches and make each latency objective
measurable. If only one workload runs at a time, changing the environment value
and restarting the same deployment is sufficient.

For PyTorch CUDA, copy the dedicated environment template, replace its example
secret, and select the CUDA Compose overlay:

```bash
cp .env.cuda.example .env.cuda
chmod 600 .env.cuda
docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

The versioned tag exposes the CUDA runtime line, for example
`ghcr.io/stighellemans/meddeid-api:0.4.2-cuda12.9`. Pin its registry digest in
production. The image contains the official CUDA-enabled PyTorch wheel but no
model weights; the host supplies a compatible NVIDIA driver, Docker Engine, and NVIDIA
Container Toolkit. The overlay requests one selected GPU and forces the CUDA
device, so an unavailable GPU fails startup instead of silently falling back to
CPU.

Each worker loads a separate model copy onto the GPU. Start with one worker.
The CUDA release target remains FP16, a 32-window batch, and eager execution
(`MEDDEID_TORCH_COMPILE_MODE=off`) to preserve throughput. The 0.4.0 candidate
audit found that FP16 changed a redaction span on the pinned Dutch fixture, even
without cross-request batching. The release policy accepts semantic differences, including reduced masking,
with a complete discrepancy report. FP32 remains a diagnostic reference.
Execution, model identity and result completeness must still pass; fresh
throughput measurements must describe the actual FP16 configuration. The following measurements describe the earlier FP16 configuration. Dynamic `reduce-overhead` compilation raised
warm batch-16 throughput from 65.0 to 71.1 documents/s (9.5%) and batch-32
throughput from 67.3 to 71.0 documents/s (5.5%). It also added 0.30 GB to the
compressed pull proxy and 1.00 GB unpacked, took 28.7 seconds to compile the
first graph, and later produced a 31.4-second p95 stall when another shape
compiled. Static compilation recompiled variable shapes, hit PyTorch's
recompile limit, and regressed ETL throughput. The release image therefore
stays eager and omits compiler-only payload. Treat compilation as a target-host,
fully prewarmed experiment rather than a precompiled portable image. The
published CPU image is not a CUDA image.

For optimized NVIDIA serving, use the [TensorRT/Triton delivery
kit](../deploy/triton/README.md). TensorRT plans depend on the GPU class and
CUDA/TensorRT stack, so every target needs its own build, output-parity test,
benchmark, digest, and compatibility record. This release supplies an exact
NVIDIA T4 target (`t4-sm75`) and an Ampere+ family target. The runtime and
gateway images are model-independent; the compiled
repository mounted at `/models` contains the target- and model-specific plan.
The public Dutch and English models share the dual-head classifier and label
set, but their different base encoders, tokenizers, and vocabulary sizes still
require separate plans.
Logits cross Triton's binary HTTP tensor extension as FP32. At startup, the
gateway resolves only the matching bundle metadata and tokenizer; it does not
download checkpoint weights. NVIDIA's pinned container
composer defines the TensorRT-only source. The final runtime projects its
measured dependency closure onto the matching CUDA base, excluding compiler
toolchains, profilers, headers, unused backends, and TensorRT engine-builder
resources. The gateway uses 64-window chunks, the best measured setting within
the plan's validated maximum shape.

The checked target catalog separates a plan that is ready to use from targets
that can be built on request:

| TensorRT target | Compute capability | Availability |
|---|---:|---|
| `t4-sm75` | 7.5 | Ready-to-use optimized T4 plan and release path |
| `ampere-plus` | 8.0 or newer | Ready-to-use hardware-compatible Ampere+ plan and release path |
| `a10g-sm86` | 8.6 | Build on request; not a supported image until its target gate passes |
| `l4-sm89` | 8.9 | Build on request; not a supported image until its target gate passes |

Run `python deploy/triton_targets.py list` to inspect the machine-readable
catalog in `deploy/triton/targets.json`. A request for another NVIDIA GPU class
adds one reviewed catalog record and a matching self-hosted runner; the build,
manifest, parity, benchmark, image-size, vulnerability, SBOM, and attestation
steps are shared. The generic manual workflow can validate an `on-request`
target. Release publication promotes only retained candidates whose reviewed
catalog target is `ready`. Ask the maintainers for a target build rather than
treating either released plan as universal.

The T4 target and Ampere+ family are real compatibility boundaries, not merely
names. A serialized TensorRT plan is not the portable GPU artifact. Use the PyTorch CUDA image when
one runtime must accept different compatible MedDeID models without a compile
step; use TensorRT only when the exact model plan has parity and performance
evidence for that GPU class and runtime stack.

For external users, prefer the published runtime and gateway images pinned by
digest and a separately checksummed model repository. A local TensorRT build is
for a new model revision or GPU target, not an installation shortcut; its
pinned builder is substantially larger than the serving runtime.

Release `0.4.2` is available for AMD64 and ARM64. Resolve the release tag to
its current multi-platform digest, record that digest in the deployment
manifest, and pin the immutable digest in production:

```bash
docker buildx imagetools inspect ghcr.io/stighellemans/meddeid-api:0.4.2
```

## Environment variable reference

The checked-in environment templates contain only the choices most operators
need to make. The settings below are available when a deployment has a measured
reason to override a default. Values shown are the supplied Compose defaults;
the CUDA and Triton overlays replace the runtime-specific values noted below.

### Model and public service settings

| Setting | Default | Purpose |
|---|---|---|
| `MEDDEID_MODEL` | Required | Hugging Face model ID or mounted MedDeID bundle directory. |
| `MEDDEID_REVISION` | Model default | Immutable Hub revision. Leave empty for a local directory or to use the model's declared default. |
| `MEDDEID_LANGUAGE_PROFILE` | Model default | Language and regional profile used when a request omits `metadata.lang`. |
| `MEDDEID_OFFLINE` | `false` | Require model files to be present locally instead of contacting the Hub. |
| `MEDDEID_SERVING_PROFILE` | `latency` | Use `latency` for individual requests or `throughput` for sustained accelerated traffic. |
| `MEDDEID_API_KEY` | Empty | Bearer token or `X-API-Key` value accepted by the service. |
| `MEDDEID_REQUIRE_API_KEY` | `false` | Reject unauthenticated inference requests. The production templates set this to `true`. |
| `MEDDEID_BIND_ADDRESS` | `127.0.0.1` | Host address published by Compose. Direct Python startup otherwise uses `0.0.0.0`. |
| `MEDDEID_PORT` | `8000` | Host port for the API. |
| `MEDDEID_DOCS_ENABLED` | `false` | Expose `/docs`, `/redoc`, and `/openapi.json`. Direct Python startup otherwise enables them. |
| `MEDDEID_UI_ENABLED` | `false` | Expose the single-note browser interface. Direct Python startup otherwise follows the documentation setting. |

### HTTP and workload limits

| Setting | Default | Purpose |
|---|---:|---|
| `MEDDEID_MAX_INPUT_CHARS` | `20000` | Maximum characters in one note. |
| `MEDDEID_MAX_BATCH_DOCUMENTS` | `32` | Maximum documents in one batch request. |
| `MEDDEID_MAX_BATCH_CHARS` | `200000` | Maximum combined characters in one batch request. |
| `MEDDEID_MAX_REQUEST_BYTES` | `2000000` | Maximum HTTP request-body size in bytes. |
| `MEDDEID_MAX_CONCURRENT_REQUESTS` | `auto` | Requests admitted per worker. Automatic values follow the backend and serving profile. |
| `MEDDEID_QUEUE_TIMEOUT_SECONDS` | `30` | Maximum wait for an inference slot before returning HTTP 503. |
| `MEDDEID_WORKERS` | `1` | API worker processes. Each worker loads its own model; the Triton overlay uses four gateway workers. |
| `MEDDEID_ACCESS_LOG` | `true` | Write HTTP access logs. Request and response bodies are never included. |
| `MEDDEID_PROXY_HEADERS` | `false` | Trust supported forwarded headers from a reverse proxy. |
| `MEDDEID_FORWARDED_ALLOW_IPS` | `127.0.0.1` | Comma-separated proxy IPs allowed to supply forwarded headers. |
| `MEDDEID_AGE_GRANULARITY_CONFIG` | Packaged policy | Path to a reviewed age-generalization policy. |
| `MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS` | `366` | Warn when the supplied absolute date shift is smaller than this value. |

### Inference runtime

These are implementation controls, not additional setup choices. Keep the
values supplied by the selected deployment unless you benchmark the changed
configuration on the target hardware.

| Setting | Default | Purpose |
|---|---|---|
| `MEDDEID_BACKEND` | `torch` | Inference backend: `torch` or `triton`. |
| `MEDDEID_DEVICE` | `cpu` in Compose | Torch device. Direct Python startup selects an available local device automatically; the CUDA overlay uses `cuda`. |
| `MEDDEID_TORCH_PRECISION` | `fp32` | Torch precision. The CUDA release target defaults to `fp16`; `fp32` is available for diagnosis. |
| `MEDDEID_TORCH_COMPILE_MODE` | `off` | Optional `torch.compile` mode. The published configuration remains eager. |
| `MEDDEID_TORCH_COMPILE_DYNAMIC` | `true` | Allow dynamic shapes when Torch compilation is enabled. |
| `MEDDEID_WINDOW_BATCH_SIZE` | `32` | Maximum note windows in one runtime call. The Triton overlay uses `64`. |
| `OMP_NUM_THREADS` | `4` | OpenMP threads per process in Compose. |
| `MKL_NUM_THREADS` | `4` | MKL threads per process in Compose. |

### Advanced throughput controls

The `throughput` profile selects the measured behavior automatically. These
settings exist for workload-specific experiments; changing them should not be
part of an initial deployment.

| Setting | Default | Purpose |
|---|---|---|
| `MEDDEID_MICROBATCH_ENABLED` | `auto` | Enable bounded cross-request batching for accelerated Torch throughput. |
| `MEDDEID_MICROBATCH_MAX_WINDOWS` | `32` | Maximum windows combined in one microbatch; Triton uses `16`. |
| `MEDDEID_MICROBATCH_MAX_TOKENS` | `16384` | Maximum tokens combined in one microbatch; Triton uses `8192`. |
| `MEDDEID_MICROBATCH_MAX_WAIT_MS` | `1` | Maximum time spent collecting a microbatch. |
| `MEDDEID_MICROBATCH_QUEUE_MAX_WINDOWS` | `8192` | Maximum queued windows. |
| `MEDDEID_MICROBATCH_QUEUE_MAX_REQUESTS` | `256` | Maximum queued requests. |
| `MEDDEID_SEQUENCE_LENGTH_BUCKETS` | `auto` | Token-length buckets, `off`, or comma-separated sizes. |

### Triton connection and plan selection

The optimized NVIDIA Compose file supplies these values. Normal users select
only the model, revision, and language profile in `.env.triton`; the plan
service detects the GPU and obtains the matching release artifact.

| Setting | Default | Purpose |
|---|---|---|
| `MEDDEID_TRITON_URL` | `http://triton:8000` | Internal Triton inference endpoint. |
| `MEDDEID_TRITON_TIMEOUT` | `30` | Triton request timeout in seconds. |
| `MEDDEID_TRITON_TRANSPORT` | `binary` | Tensor transport used by the gateway; the generic backend default is `json`. |
| `MEDDEID_TRITON_MODEL_ARTIFACT` | Automatic | Explicit OCI plan artifact override. |
| `MEDDEID_TRITON_MODEL_REPOSITORY` | `./deploy/triton/model_repository` | Local directory where the selected plan is unpacked and then mounted read-only into Triton. |
| `MEDDEID_TRITON_RUNTIME_IMAGE` | Release runtime image | Weight-free Triton serving image. |
| `MEDDEID_GPU_DEVICE_ID` | `0` | NVIDIA device exposed to the CUDA or Triton deployment. |
| `MEDDEID_REGISTRY_USERNAME` | Empty | Registry username when the selected plan artifact is private. |
| `MEDDEID_REGISTRY_TOKEN` | Empty | Registry token when the selected plan artifact is private. |
| `HF_TOKEN` | Empty | Hugging Face token only when a selected Hub model is private or gated. |

`MEDDEID_API_IMAGE` selects the CPU image or, in the Triton Compose file, the
weight-free gateway image. `MEDDEID_PYTORCH_CUDA_IMAGE` selects the CUDA image.
Pin either to an approved digest in production. Source-build pins and local
TensorRT builder settings are documented with the corresponding advanced build
workflow rather than in the end-user templates.

## Minimum secure configuration

1. Pull an immutable image digest, not a moving tag.
2. Store a random `MEDDEID_API_KEY` in the platform secret manager.
3. Set `MEDDEID_REQUIRE_API_KEY=true`.
4. Keep `MEDDEID_DOCS_ENABLED=false` for an unattended service unless approved
   operators need interactive documentation. When enabled for integration or
   acceptance testing, restrict `/docs`, `/redoc`, and `/openapi.json` at the
   reverse proxy because MedDeID's API-key check does not protect those routes.
5. Keep `MEDDEID_UI_ENABLED=false` unless the single-note browser interface is
   explicitly needed.
6. Bind MedDeID only to a private interface. The supplied Compose file defaults
   to `127.0.0.1`.
7. Terminate TLS at a maintained reverse proxy or service mesh and enforce its
   request-body limit at or below `MEDDEID_MAX_REQUEST_BYTES`.
8. Do not log request bodies, response bodies, headers containing API keys, or
   metadata. Treat all input, output, manifests, caches, and traces as
   sensitive.
10. Validate recall and unnecessary redaction on representative local notes
   before operational use.
11. Pin and audit `MEDDEID_AGE_GRANULARITY_CONFIG` when overriding the packaged
    default, and record its SHA-256 from the local `meddeid model-info` output.
12. Before rollout, run
    `meddeid model-info --model <approved-model> --verify-runtime` in the
    deployment environment to verify its backend and device, not only the
    bundle metadata.

Generate a key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Start with Docker Compose

Clone the repository and create a permission-restricted deployment environment
file from the reviewed template:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
cp .env.example .env.cpu
chmod 600 .env.cpu
```

Set `MEDDEID_API_IMAGE` to the approved immutable image digest. Require an API
key and start with the interactive UI and API documentation disabled:

```dotenv
MEDDEID_API_IMAGE=ghcr.io/stighellemans/meddeid-api@sha256:<approved-digest>
MEDDEID_API_KEY=<secret>
MEDDEID_REQUIRE_API_KEY=true
MEDDEID_DOCS_ENABLED=false
MEDDEID_UI_ENABLED=false
```

Prefer injecting `MEDDEID_API_KEY` from the platform secret manager. If it must
be stored in the environment file, treat that file as a secret and never commit
it.

The API documentation is useful during integration and acceptance testing. Set
`MEDDEID_DOCS_ENABLED=true` in that restricted environment. If it remains
enabled in production, limit the documentation routes to an authenticated
operator group or administrative network at the reverse proxy. Disabling the
documentation routes does not disable the inference API.

Use ordinary Compose commands to start and inspect the service:

```bash
docker compose --env-file .env.cpu pull meddeid
docker compose --env-file .env.cpu up --detach meddeid
docker compose --env-file .env.cpu ps
docker compose --env-file .env.cpu logs --follow meddeid
```

For a planned stop, run:

```bash
docker compose --env-file .env.cpu down
```

To upgrade, change the pinned digest, repeat `pull` and `up --detach`, and
verify readiness before returning traffic. `./scripts/start-local.sh` remains
the shortcut for a temporary local evaluation; it creates local configuration
and enables the browser and API documentation for that purpose.

## Direct Python alternative

When containers are not suitable, copy `server.env.example` to an ignored,
permission-restricted file and start with:

```bash
cp server.env.example meddeid-server.env
chmod 600 meddeid-server.env
meddeid-server --env-file meddeid-server.env
```

The checked-in template makes the non-secret configuration reviewable and
reproducible. Existing process variables override file values, so inject the
API key from the platform secret manager where possible. The loader rejects
unknown and duplicate keys rather than silently accepting configuration typos.

Send it as a bearer token:

```bash
curl --fail-with-body https://meddeid.example.org/deidentify \
  -H "Authorization: Bearer ${MEDDEID_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Patiënt Jan Peeters kwam op controle."}'
```

## Capacity and availability

Start with one API worker, one admitted inference request, four Torch threads,
4 vCPU, and 8 GiB RAM. Prefer `/deidentify-batch` for throughput. Every extra
worker loads another model copy and requires independent memory measurement.

Use:

- `GET /live` for process liveness;
- `GET /health` for minimal readiness and enabled model/profile information;
- `X-Request-ID` for request correlation without recording patient text; and
- HTTP 503 plus `Retry-After` as the back-pressure signal.

Measure p50/p95/p99 latency, documents and characters per second, memory, and
restart time on the real note-length distribution. Configure client timeouts
and retries only for idempotent requests, using bounded exponential backoff.

## Upgrades and rollback

Record the image digest, model revision, bundle hash, package versions,
language profile, age-policy identity, and date-shift warning threshold from the
deployment manifest and local `meddeid model-info` output. `/health` deliberately
does not expose these administrator details. Before an upgrade:

1. run the same local validation set against old and new digests;
2. compare span and rendered-text changes;
3. confirm resource and memory limits;
4. deploy a canary without logging clinical payloads; and
5. retain the previous digest for rollback.

Never replace the model directory or TensorRT plan in place. Deploy a new
identified artifact and roll back by digest.

## Operational acceptance checklist

- Image signature/provenance, SBOM, digest, and vulnerability review accepted.
- API key stored outside Compose and source control.
- TLS, firewall/network policy, request limit, and rate limit verified.
- Container runs non-root, read-only, capability-free, and without runtime Hub
  access.
- Health alerts and restart policy tested.
- Local recall and unnecessary-redaction acceptance thresholds signed off.
- Human review and incident handling paths documented.
- Data retention and deletion behavior verified for logs and outputs.
