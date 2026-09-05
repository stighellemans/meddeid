# TensorRT/Triton delivery kit

This directory defines a reproducible **target-specific** TensorRT/Triton
artifact. It does not claim that one TensorRT plan is portable across NVIDIA
GPU classes or TensorRT/CUDA stacks.

The deliverable has two parts:

1. an immutable Triton image containing `model.plan`, `config.pbtxt`, and
   `build-manifest.json`; and
2. a weight-free gateway image containing the current MedDeID API,
   tokenization, profile, post-processing, and provenance logic, joined by
   `compose.triton.yaml` so only the authenticated gateway is exposed.

Compose alone is not a complete deliverable because a TensorRT plan must first
be compiled and validated on the supported target GPU. Conversely, publishing
only the model image would leave operators to reconstruct the gateway,
authentication, health, and network configuration.

NVIDIA's pinned `compose.py` and pinned minimal/full images produce the
auditable TensorRT-only source image. MedDeID then projects the measured runtime
closure onto the matching pinned CUDA base. The final image contains Triton,
the TensorRT backend, and their required TensorRT, NCCL, DCGM, and CUPTI shared
libraries, but not headers, compilers, profilers, unused backends, or engine
builder resources. Removing files in a later layer would not reduce pull size,
so both composition and projection happen before the release image is emitted.

## Current release target

`versions.env` pins the model, bundle contract, Triton server and minimal
images, NVIDIA container-composer revision, TensorRT builder image, compute
precision, output precision, and optimization profile. `targets.json` is the
machine-readable publication catalog; `triton_targets.py` validates target
identity, host compatibility, image naming, and the target-spec hash recorded
in every build manifest.
The T4 plan uses FP16 weights and compute, then casts logits to FP32 at the
Triton boundary. The gateway uses Triton's binary HTTP tensor extension while
preserving the existing MedDeID JSON API contract.

| Target | Status | Meaning |
|---|---|---|
| `t4-sm75` | `ready` | The optimized T4 plan and release path are ready to use. |
| `a10g-sm86` | `on-request` | A matching runner can build and validate a candidate on request. |
| `l4-sm89` | `on-request` | A matching runner can build and validate a candidate on request. |

Inspect the catalog with `python deploy/triton_targets.py list`. Adding another
GPU class is a data-only target declaration plus a matching self-hosted runner;
the remaining build and evidence machinery is shared. An `on-request` target
may be validated, but the workflow refuses publication until a reviewed change
makes it `ready` and the run uses a version tag.

No target image is a released artifact merely because these source files
exist. A release requires a successful run of the GPU gate below and published
evidence.

`t4-sm75` is an intentional compatibility limit of that serialized plan. It is
not a universal NVIDIA image. External operators who need one GPU image across
different supported NVIDIA devices should use the PyTorch CUDA artifact;
operators choosing TensorRT must select an image whose GPU target and runtime
stack match their validated deployment.

## Current T4 validation snapshot

The checked machine-readable record is
[`t4-benchmark-summary.json`](t4-benchmark-summary.json). It binds the figures
below to the target, immutable model and fixture revisions, test shape, parity
result, and stated measurement limitations.

On an Azure `Standard_NC4as_T4_v3` (4 vCPU, Tesla T4 16 GiB), NVIDIA's official
TensorRT-only composition measured 17,243,385,469 unpacked bytes. The projected
runtime measured 1,674,701,600 bytes, a 90.3% reduction. With the T4 plan, the
model-server image was 1.925 GB unpacked and 1.017 GB as a gzip-compressed
`docker image save`; the weight-free gateway was 549 MB unpacked and 116 MB
compressed. The complete pair therefore measured 2.474 GB unpacked with a
1.133 GB pull-size proxy. The serialized plan itself is 250 MB; the gateway's
weight-free model contract and tokenizer payload are about 5 MB.

Using the pinned 300-document public synthetic fixture, 60 HTTP requests at
batch size 16 and concurrency 8 processed 948 documents and 1,307,107
characters after warmup. With four weight-free gateway workers and 64-window
request-local chunks, the pair delivered 168.8 documents/s and 232,731
characters/s. Request latency was 0.725 s p50 and 1.418 s p95; peak GPU memory
was 911 MiB. Exact current-suite semantic parity
passed for all 300 pinned fixture documents with zero differences. Registry
compression and real clinical note distributions can differ, so retain these
as comparative validation evidence, not a deployment capacity promise.

The scheduling sweep used three repetitions per case and two gateway workers;
the selected default and nested variant were then repeated with four workers.
Short notes averaged 314
characters, the mixed fixture averaged 1,399, and the long fixture covered
8,000, 12,000, and 18,000 characters. Medians were:

| HTTP workload | Request-local, 4 workers: documents/s, p50 / p95 | Nested 1 ms, 4 workers | Request-local, 2 workers | Triton 1 ms queue, 2 workers | Nested 1 ms, 2 workers | Older-inspired nested 5 ms, 2 workers |
|---|---:|---:|---:|---:|---:|---:|
| One short note, concurrency 1 | 113.7, 8.7 / 10.0 ms | 69.4, 13.7 / 15.0 ms | 94.0, 10.4 / 11.9 ms | 83.2, 11.8 / 13.1 ms | 73.8, 13.4 / 14.5 ms | 54.6, 18.2 / 19.4 ms |
| One mixed note, concurrency 1 | 60.6, 15.8 / 21.9 ms | 45.9, 22.4 / 27.0 ms | 60.0, 16.5 / 21.6 ms | 54.9, 18.2 / 23.2 ms | 45.7, 22.2 / 27.5 ms | 38.7, 25.7 / 31.9 ms |
| Short-note burst, concurrency 8 | 293.3, 25.4 / 37.9 ms | 327.3, 23.3 / 34.2 ms | 264.5, 29.4 / 42.7 ms | 253.7, 29.1 / 44.6 ms | 285.1, 27.3 / 39.4 ms | 256.8, 29.7 / 43.7 ms |
| Batch 16, concurrency 8 | 168.8, 0.725 / 1.418 s | 165.3, 0.711 / 1.306 s | 132.5, 0.919 / 1.484 s | 130.1, 0.894 / 1.601 s | 132.4, 0.858 / 1.637 s | 135.4, 0.837 / 1.596 s |
| Batch 32, concurrency 8 | 159.4, 1.266 / 2.156 s | 154.7, 1.348 / 2.396 s | 134.2, 1.698 / 2.875 s | 133.0, 1.549 / 2.789 s | 127.2, 1.659 / 3.006 s | 131.1, 1.669 / 2.626 s |

Four request-local gateway workers are the T4 external-user default. They
improved batch-16 throughput by 27.4% over two workers, while the GPU still held
one TensorRT model instance, and they avoid a second queue. At the same
four-worker count, nested 1 ms batching helped only the short-note burst
(11.6%); it reduced batch throughput by 2–3%, long-note ETL by 12.7%, and
isolated short/mixed throughput by 24–39%. The older 5 ms settings from the
previous server improved two-worker batch-16 throughput by only 2.2%, regressed
the other request shapes, and are not defaults for the current suite.

## Build on the target NVIDIA host

Prerequisites are Linux, a compatible NVIDIA driver, Docker Engine, NVIDIA
Container Toolkit, Python 3, the `hf` CLI, and a source environment containing
MedDeID plus the ONNX export dependencies. Review and accept the applicable
NVIDIA NGC/deep-learning-container terms before building or distributing the
combined image; the MedDeID code and model terms do not replace the base-image
terms.

```bash
python3 -m venv .venv-triton
source .venv-triton/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]' distro requests onnx onnxscript

./deploy/preflight_triton_host.sh t4-sm75 0
./deploy/stage_triton_model.sh deploy/triton/model_source
./deploy/build_triton_repository.sh \
  deploy/triton/model_source \
  deploy/triton/model_repository \
  t4-sm75 \
  0

./deploy/build_triton_image.sh \
  deploy/triton/model_repository \
  ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.3.0-trt26.07-fp16 \
  t4-sm75

./deploy/build_triton_gateway_image.sh \
  ghcr.io/stighellemans/meddeid-triton-gateway:0.3.0
```

The scripts refuse to overwrite non-empty model source or repository
directories. Review `deploy/triton/model_repository/build-manifest.json`
before testing or publishing.

## Start the candidate and PyTorch reference

```bash
cp .env.triton.example .env.triton
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Put that value in MEDDEID_API_KEY in .env.triton.

docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.validation.yaml \
  up --detach

docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.validation.yaml \
  ps
```

This exposes the TensorRT-backed gateway on `127.0.0.1:8000` and the PyTorch
CPU reference on `127.0.0.1:8001`. Triton itself is confined to an internal
Compose network.

## Mandatory parity gate

```bash
set -a
source .env.triton
set +a
python deploy/validate_triton_parity.py \
  deploy/triton/parity-fixture.jsonl \
  --candidate-url http://127.0.0.1:8000 \
  --reference-url http://127.0.0.1:8001 \
  --output deploy/triton/parity-report.json

docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.validation.yaml \
  down
```

The parity gate requires the same model identity, de-identified text, span
boundaries/labels/replacements, language profile, warnings, and processing
metadata. Floating-point confidence fields are deliberately excluded; they
can differ slightly under FP16 without changing semantics. Run the same gate
on the 50-note manual fixture and institution-approved representative data.

Stage the immutable public benchmark fixture, then benchmark the candidate and
reference separately with the same batching, warmup, request count, and
concurrency. The report retains aggregate timing and identity only, not note or
response text:

```bash
./deploy/stage_benchmark_fixture.sh deploy/triton/benchmark_source
python deploy/benchmark_http.py deploy/triton/benchmark_source/data/test.jsonl \
  --base-url http://127.0.0.1:8000 --batch-size 16 \
  --warmup-requests 8 --requests 60 --concurrency 8 \
  --output deploy/triton/triton-benchmark.json
python deploy/benchmark_http.py deploy/triton/benchmark_source/data/test.jsonl \
  --base-url http://127.0.0.1:8001 --batch-size 16 \
  --warmup-requests 8 --requests 60 --concurrency 8 \
  --output deploy/triton/torch-reference-benchmark.json
```

## Publication unit

For every supported GPU target publish and retain together:

- the target-specific image tag and immutable registry digest;
- the paired gateway image tag and immutable registry digest;
- `build-manifest.json` and its SHA-256;
- the parity report and fixture version/hash;
- an SBOM, provenance/attestation, and vulnerability-scan result;
- NVIDIA GPU class, compute capability, minimum accepted driver, Triton stack,
  TensorRT version, compute precision, and output precision;
- cold-start, p50/p95/p99 latency, throughput, and peak GPU-memory results; and
- the exact API image digest used for parity.

Never publish a universal `latest` TensorRT tag. Operators should put the
released image **digest** in `MEDDEID_TRITON_IMAGE`, then use
`compose.triton.yaml` as the deployment wiring.

Official compatibility references:

- [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Triton model repositories](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)
