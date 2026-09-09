# TensorRT/Triton delivery kit

This directory contains the advanced build and validation path for MedDeID's
TensorRT deployment. Most operators should use `compose.triton.yaml`: they
select a model, optional revision, and language profile, and Compose detects the GPU and retrieves
the matching released plan.

The deliverable has three parts:

1. a weight-free Triton runtime image;
2. a weight-free MedDeID API gateway; and
3. a separately supplied model repository containing `model.plan`,
   `config.pbtxt`, and `build-manifest.json`.

The TensorRT plan is the compiled model and contains the learned parameters.
It must match the exact model revision, tokenizer and label contract, TensorRT
stack, and GPU target. A plan cannot be reused for another checkpoint merely
because the API output schema is the same.

`deploy/triton/release.json` connects each MedDeID suite release to its gateway,
runtime, and validated model plans. `compose.triton.yaml` verifies and keeps the
selected plan under `deploy/triton/model_repository`, then mounts it read-only
into Triton. Operators do not need to configure the internal gateway or model
repository path.

NVIDIA's pinned `compose.py` and pinned minimal/full images produce the
auditable TensorRT-only source image. MedDeID then projects the measured runtime
closure onto the matching pinned CUDA base. The final image contains Triton,
the TensorRT backend, and their required TensorRT, NCCL, DCGM, and CUPTI shared
libraries, but not headers, compilers, profilers, unused backends, or engine
builder resources. Removing files in a later layer would not reduce pull size,
so both composition and projection happen before the release image is emitted.

## Current release target

`versions.env` pins the build stack and its default model. `release.json` lists
the public model plans belonging to this suite release. `targets.json` defines
the supported hardware targets; `triton_targets.py` validates the detected GPU,
artifact naming, and the target-spec hash recorded in every build manifest.
The T4 and Ampere+ plans use FP16 weights and compute, then cast logits to FP32
at the Triton boundary. The gateway uses Triton's binary HTTP tensor extension
while preserving the existing MedDeID JSON API contract.

| Target | Status | Meaning |
|---|---|---|
| `t4-sm75` | `ready` | The optimized T4 plan and release path are ready to use. |
| `ampere-plus` | `ready` | One hardware-compatible plan supports Ampere and newer GPUs. |
| `a10g-sm86` | `on-request` | A matching runner can build and validate a candidate on request. |
| `l4-sm89` | `on-request` | A matching runner can build and validate a candidate on request. |

Inspect the catalog with `python deploy/triton_targets.py list`. Adding another
GPU class is a data-only target declaration plus a matching self-hosted runner;
the remaining build and evidence machinery is shared. An `on-request` target
may be validated, but it is not part of the release unless a reviewed change
makes it `ready` and a publication workflow selects the retained candidate.

No plan is a released artifact merely because these source files exist. A
release requires a successful run of the GPU gate below and published evidence.

`t4-sm75` is an intentional compatibility limit of its serialized plan. The
separate `ampere-plus` plan is compiled with TensorRT's Ampere+ compatibility
mode; neither is a universal NVIDIA plan. Operators choosing TensorRT must
select a plan whose model revision, hardware family, and runtime stack match
their deployment. Use the PyTorch CUDA artifact when no released plan matches.

## Current T4 validation snapshot

The checked machine-readable record is
[`t4-benchmark-summary.json`](t4-benchmark-summary.json). It binds the figures
below to the target, immutable model and fixture revisions, test shape, parity
result, and stated measurement limitations.

On an Azure `Standard_NC4as_T4_v3` (4 vCPU, Tesla T4 16 GiB), NVIDIA's official
TensorRT-only composition measured 17,243,385,469 unpacked bytes. The projected
runtime measured 1,674,701,600 bytes, a 90.3% reduction. The serialized plan
measured 250 MB and the weight-free gateway measured 549 MB unpacked. Earlier
validation packaged the runtime and plan together; current delivery keeps the
plan outside both images so a different model does not require a new runtime
or gateway image.

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

## Build locally on the target NVIDIA host

This path is for hospitals that must build inside their own boundary, custom
MedDeID models, and maintainers publishing a new target. It requires Linux,
Docker Engine, an NVIDIA driver, and NVIDIA Container Toolkit. Review and
accept the applicable NVIDIA container terms before building or distributing
the runtime.

Create `.env.triton` from the supplied example and set `MEDDEID_MODEL`,
`MEDDEID_REVISION`, and `MEDDEID_LANGUAGE_PROFILE`. The builder reads the
official GPU name and compute capability from `nvidia-smi`; you do not select a
hardware label. Start by building only the plan tool:

```dotenv
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth
# Empty uses the default Hub revision; a commit makes the build repeatable.
MEDDEID_REVISION=1f20655454dcbd042647cacdfff6b6802a970959
MEDDEID_LANGUAGE_PROFILE=nl-BE
```

New local builds derive a key directly from NVIDIA's name, for example
`NVIDIA A10G` becomes `nvidia-a10g`. Compute capability remains a separate field.
Old target IDs remain readable in the release catalog for existing artifacts.
No catalog entry is required to detect a GPU or build a local plan. A family plan can be tested on another GPU covered by its recorded compatibility
mode. The driver/runtime and memory requirements still apply.

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  build plan-builder

docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  run --rm plan-builder
```

`plan-builder` downloads the exact Hub revision into a Docker cache volume,
checks its MedDeID bundle, compiles it on the selected GPU, and writes
`deploy/triton/local_model_repository`. It refuses to overwrite a different plan.
To use an existing model directory, add the read-only local-model overlay:

```bash
MEDDEID_LOCAL_MODEL_DIRECTORY=/approved/models/my-meddeid-model \
docker compose \
  --env-file .env.triton \
  -f compose.triton.build.yaml \
  -f compose.triton.build.local-model.yaml \
  run --rm plan-builder
```

Start with the release images and the local plan:

```bash
docker compose \
  --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.local-plan.yaml \
  up --detach
```

For a mounted model directory, set `MEDDEID_LOCAL_MODEL_DIRECTORY` in
`.env.triton` and add `-f compose.triton.local-model.yaml` at startup as well,
so the gateway uses the same bundle. Set `MEDDEID_MODEL=/input-model`, the
container mount path, and leave `MEDDEID_REVISION` empty (or use an institutional
version label). No Hub token is needed for a local directory. Hospital models
do not need to be published to Hugging Face. For a Hub model, an empty revision
resolves `main` automatically; the gateway inherits the exact plan commit.

To build the gateway and runtime too, use
`docker compose --env-file .env.triton -f compose.triton.build.yaml build`, then
`docker compose --env-file .env.triton -f compose.triton.build.yaml run --rm runtime-builder`.
Use `compose.triton.local-images.yaml` instead of `compose.triton.local-plan.yaml`
at startup. This is needed for an institution's build policy or a changed
runtime stack, not merely for every new GPU. A newer stack must support the GPU
and match between the builder and runtime.

The plan stays local and the build finishes without contribution prompts.
Local builds automatically use `sameComputeCapability` on Turing (7.5), or
`ampere+` on supported newer GPUs. The manifest records the build GPU and the
compatibility mode; older exact-GPU plans are never treated as family plans.
The low-level release builder keeps native mode unless the release workflow
explicitly selects a published family target. Family artifacts must pass GPU
validation before being added to `release.json`.

The plan builder is deliberately large: it contains PyTorch, ONNX, TensorRT,
and compilation tools. It is not a parent layer of either deployed image. The
runtime-builder also mounts `/var/run/docker.sock`; that grants it control over
the host Docker daemon, so use only a reviewed checkout on a dedicated build
host.

Review `deploy/triton/local_model_repository/build-manifest.json` and run the parity
gate below before treating a local build as deployable. The source checkpoint
is not needed after compilation, but the generated plan contains its learned
parameters and remains subject to the model's access and licence controls.

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

Before allocating a GPU or building images, stage the pinned benchmark and
exercise its request contract in an environment with `meddeid[dev]` installed:

```bash
python deploy/preflight_benchmark_contract.py "$MEDDEID_BENCHMARK_FILE" \
  --language-profiles "$MEDDEID_LANGUAGE_PROFILES" \
  --output deploy/triton/fixture-contract-preflight.json
```

Use the fixture path and comma-separated language profiles from the selected
model's catalog entry. This sends every normalized parity, benchmark batch, and
single-document request through the actual API with an echo-only engine. It
checks nested metadata, request limits, language selection, unique document IDs,
and agreement between the fixture loaders without loading weights or using a
GPU. It does not establish inference correctness. The GPU workflow runs this
check before image builds and retains the report even when validation fails.

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

- the weight-free Triton runtime and gateway image digests;
- the target- and model-specific repository archive and checksum;
- `build-manifest.json` and its SHA-256;
- the parity report and fixture version/hash;
- an SBOM, provenance/attestation, and vulnerability-scan result;
- NVIDIA GPU class, compute capability, minimum accepted driver, Triton stack,
  TensorRT version, compute precision, and output precision;
- cold-start, p50/p95/p99 latency, throughput, and peak GPU-memory results; and
- the exact API image digest used for parity.

Never describe a compiled plan as universal. The suite release records the
gateway, runtime, and plan digests together. Normal operators select the model;
MedDeID detects the GPU and resolves the matching artifact. Release engineering
is responsible for validating, publishing, and pinning those artifacts. Users
who build a public model on an unsupported GPU are directed to the
[GPU contribution path](https://stighellemans.github.io/meddeid/project/contributing/#add-optimized-support-for-another-nvidia-gpu).

Official compatibility references:

- [ORAS artifact push and pull](https://oras.land/docs/quickstart/)
- [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Triton model repositories](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/)
