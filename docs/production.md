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

The published CPU image contains the model and sets `MEDDEID_OFFLINE=true`.
GPU deployments should provide the same local model and offline configuration,
so no note or model request needs to leave the deployment boundary. MedDeID
provides API-key authentication and workload limits. The surrounding platform
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
| Fixed NVIDIA T4 deployment | Target-specific TensorRT server with the weight-free MedDeID gateway | T4 release candidate; the only optimized target prepared for publication |
| Native Apple-silicon service | PyTorch MPS from the Python installation | Validated on one M4 Pro; Linux containers cannot use the host Metal device |

The CUDA and T4 images are published products only after their GPU release
gates pass and their immutable digests are recorded. A10G and L4 are
build-on-request TensorRT targets, not interchangeable alternatives to the T4
plan; each needs its own target evidence before publication.

### Measured T4 snapshot

The current images were measured end to end on one Azure
`Standard_NC4as_T4_v3` VM (4 vCPU, Tesla T4 16 GiB). The benchmark used the
pinned public synthetic fixture at revision
`9b95ebbfb091a5390e4fc39e2ef74e7580aac068`. Each throughput value is the
median of three warm runs at batch size 16. CPU used concurrency 1; the GPU
paths used concurrency 8 and processed 948 documents and 1,307,107 characters
per run.

| Deployment | Pull-size proxy | Unpacked | Ready | Documents/s | HTTP p50 / p95 | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch CPU | 0.73 GB | 1.61 GB | 20.1 s | 1.85 | 6.78 / 15.74 s | — |
| PyTorch CUDA, FP16 | 4.54 GB | 7.40 GB | 68.8 s | 65.0 | 1.87 / 2.49 s | 1,655 MiB |
| T4 TensorRT server + gateway | 1.13 GB (1.017 + 0.116) | 2.47 GB | 10.0 s | 168.8 | 0.725 / 1.418 s | 911 MiB |

The pull-size proxy is a gzip-compressed `docker image save`; registry transfer
size varies with compression and shared layers. Readiness was measured after
the image was present. Latency includes queueing under the stated concurrency.
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
from 40.08 to 42.10 documents/s. Keep the latency profile for isolated calls.
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

The embedded PyTorch model occupies about 503 MB in both CPU and CUDA images.
The CUDA virtual environment is 6.72 GB unpacked, dominated by 4.61 GB of
NVIDIA runtime libraries and 1.71 GB of PyTorch; the compiler package, headers,
tests, and static CUDA archives are already absent. The TensorRT pair replaces
that general runtime with a 250 MB target plan: its gateway has only a 5 MB
weight-free model contract and its separate runtime retains the TensorRT/Triton
shared-library closure. Further large CUDA reductions would therefore require a
different execution runtime, not another cache cleanup.

CPU is the simplest choice for occasional traffic. Use PyTorch CUDA when the
NVIDIA GPU class may vary. For a fixed, validated T4, the TensorRT pair is
preferable in this test: about one quarter of CUDA's pull proxy and 2.6 times
its throughput. The plan remains T4-specific.

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
templates. The selected image supplies its measured precision, compilation,
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
`ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9`. Pin its registry digest in
production. The image contains the official CUDA-enabled PyTorch wheel and the
model; the host supplies a compatible NVIDIA driver, Docker Engine, and NVIDIA
Container Toolkit. The overlay requests one selected GPU and forces the CUDA
device, so an unavailable GPU fails startup instead of silently falling back to
CPU.

Each worker loads a separate model copy onto the GPU. Start with one worker.
The CUDA image defaults to FP16 autocast, a 32-window batch, and eager execution
(`MEDDEID_TORCH_COMPILE_MODE=off`). Dynamic `reduce-overhead` compilation raised
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
benchmark, digest, and compatibility record. The initial target is NVIDIA T4
(`t4-sm75`), with a versioned name such as
`ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.3.0-trt26.07-fp16`.
There is no universal TensorRT `latest` tag. Until an exact target image and its
evidence are published, treat it as a locally validated custom build rather
than a supported MedDeID release. The `fp16` suffix identifies the plan's
weights and compute. Logits cross Triton's binary HTTP tensor extension as FP32.
The gateway image contains the suite's current API validation, tokenization,
language profiles, post-processing, and provenance logic, but intentionally
omits PyTorch and the original model weights. NVIDIA's pinned container
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
| `a10g-sm86` | 8.6 | Build on request; not a supported image until its target gate passes |
| `l4-sm89` | 8.9 | Build on request; not a supported image until its target gate passes |

Run `python deploy/triton_targets.py list` to inspect the machine-readable
catalog in `deploy/triton/targets.json`. A request for another NVIDIA GPU class
adds one reviewed catalog record and a matching self-hosted runner; the build,
manifest, parity, benchmark, image-size, vulnerability, SBOM, and attestation
steps are shared. The generic manual workflow can validate an `on-request`
target, but publication is refused until its reviewed catalog status changes to
`ready` and the workflow is dispatched from a version tag. Ask the maintainers
for a target build rather than treating the T4 plan as portable.

The T4 suffix is a real compatibility boundary, not merely a name. A serialized
TensorRT plan is not the portable GPU artifact. Use the PyTorch CUDA image when
one artifact must run across supported NVIDIA GPU models; use TensorRT only when
the exact target image has parity and performance evidence for that GPU class
and runtime stack.

For external users, prefer the published images pinned by digest. A local
TensorRT build is for auditing or validating a new target, not an installation
shortcut: its pinned builder is about 18.2 GB unpacked before the Triton build
inputs, versus about 1.13 GB compressed and 2.47 GB unpacked for the final T4
server and gateway.

Release `0.3.0` is available for AMD64 and ARM64. Resolve the release tag to
its current multi-platform digest, record that digest in the deployment
manifest, and pin the immutable digest in production:

```bash
docker buildx imagetools inspect ghcr.io/stighellemans/meddeid-api:0.3.0
```

## Minimum secure configuration

1. Pull an immutable image digest, not a moving tag.
2. Store a random `MEDDEID_API_KEY` in the platform secret manager.
3. Set `MEDDEID_REQUIRE_API_KEY=true`.
4. Set `MEDDEID_ALLOWED_MODELS` and `MEDDEID_ALLOWED_LANGUAGE_PROFILES` when the
   deployment must refuse unapproved server configuration or request locales.
5. Keep `MEDDEID_DOCS_ENABLED=false` for an unattended service unless approved
   operators need interactive documentation. When enabled for integration or
   acceptance testing, restrict `/docs`, `/redoc`, and `/openapi.json` at the
   reverse proxy because MedDeID's API-key check does not protect those routes.
6. Keep `MEDDEID_UI_ENABLED=false` unless the single-note browser interface is
   explicitly needed.
7. Bind MedDeID only to a private interface. The supplied Compose file defaults
   to `127.0.0.1`.
8. Terminate TLS at a maintained reverse proxy or service mesh and enforce its
   request-body limit at or below `MEDDEID_MAX_REQUEST_BYTES`.
9. Do not log request bodies, response bodies, headers containing API keys, or
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
cp .env.example meddeid-production.env
chmod 600 meddeid-production.env
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
docker compose --env-file meddeid-production.env pull meddeid
docker compose --env-file meddeid-production.env up --detach meddeid
docker compose --env-file meddeid-production.env ps
docker compose --env-file meddeid-production.env logs --follow meddeid
```

For a planned stop, run:

```bash
docker compose --env-file meddeid-production.env down
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
