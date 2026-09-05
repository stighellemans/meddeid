# PyTorch CUDA image

This release variant packages the ordinary MedDeID API and pinned model with a
CUDA-enabled PyTorch wheel. It preserves the CPU image's API, offline model,
non-root user, read-only-root compatibility, health check, and security
defaults. Only the neural execution device changes.

The public image naming contract is:

```text
ghcr.io/stighellemans/meddeid-api:<meddeid-version>-cuda<cuda-version>
```

For release `0.3.0`, the candidate tag is
`ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9`. Production deployments
must resolve and pin its immutable digest. There is deliberately no `gpu` or
`latest-gpu` tag: the CUDA compatibility line remains visible in every tag.

## Build and validate

Run these commands on a Linux NVIDIA host with Docker Engine, Docker Buildx,
the NVIDIA Container Toolkit, and a compatible NVIDIA driver:

```bash
./deploy/build_pytorch_cuda_image.sh meddeid-api:0.3.0-cuda12.9-test
./deploy/validate_pytorch_cuda_image.sh meddeid-api:0.3.0-cuda12.9-test
```

Validation performs a real CUDA matrix multiplication, starts the complete
API with `MEDDEID_DEVICE=cuda`, runs authenticated single-document and batch
inference, and verifies the container hardening settings. The CUDA check is
saved to `deploy/pytorch-cuda/validation-report.json`.

## Run the released image

Copy `.env.cuda.example` to an ignored environment file, replace the example
secret, replace the image tag with its validated digest, and run:

```bash
docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

Use one worker initially. Each worker loads another complete model copy onto
the GPU. The measured defaults are FP16 autocast, a 32-window batch, and eager
execution (`MEDDEID_TORCH_COMPILE_MODE=off`). With dynamic shapes,
`reduce-overhead` compilation raised warm batch-16 throughput from 65.0 to 71.1
documents/s (9.5%) and batch-32 throughput from 67.3 to 71.0 documents/s
(5.5%). It added 0.30 GB to the compressed pull proxy and 1.00 GB unpacked,
took 28.7 seconds to compile the first graph, and later produced a 31.4-second
p95 stall when another shape compiled. Static compilation recompiled input
shapes, hit PyTorch's recompile limit, and regressed ETL throughput. The release
image therefore removes the Triton compiler package, PyTorch headers and tests,
and static CUDA archives. Treat compilation as a target-host, fully prewarmed
experiment rather than a precompiled portable image. Tune window batch size,
concurrency, and workers only from measurements on the target workload.

On an Azure `Standard_NC4as_T4_v3`, the final image measured 7.40 GB unpacked
and 4.54 GB as a gzip-compressed `docker image save` pull-size proxy. With the
throughput profile it processed 948 documents at 65.0 documents/s after
warmup, with 1.87 s p50 and 2.49 s p95 HTTP batch latency at batch size 16 and
concurrency 8. Peak GPU memory was 1,655 MiB. Registry transfer size and real
note distributions can differ; use these values as comparison evidence, not a
capacity promise.

The embedded model directory accounts for about 503 MB unpacked. Of the 6.72
GB virtual environment, 4.61 GB is the CUDA/NVIDIA runtime and 1.71 GB is
PyTorch. The compiler-only Triton package is absent. This makes the remaining
size a portability trade-off of the general PyTorch CUDA runtime rather than
download-cache or build-tool residue.

The serving-profile sweep used three measured repetitions per case. Short
notes averaged 314 characters; the mixed fixture averaged 1,399 characters;
the long fixture covered 8,000, 12,000, and 18,000 characters. Medians were:

| HTTP workload | `latency` documents/s, p50 / p95 | `throughput` documents/s, p50 / p95 |
|---|---:|---:|
| One short note, concurrency 1 | 73.0, 13.6 / 14.7 ms | 65.9, 15.0 / 16.2 ms |
| One mixed note, concurrency 1 | 51.9, 19.1 / 23.5 ms | 47.1, 20.9 / 27.0 ms |
| Short-note burst, concurrency 8 | 75.3, 106 / 109 ms | 141.5, 55.5 / 63.5 ms |
| Batch 16, concurrency 8 | 53.8, 2.37 / 2.87 s | 65.0, 1.87 / 2.49 s |
| Batch 32, concurrency 8 | 51.5, 4.52 / 5.20 s | 67.3, 3.41 / 4.14 s |

This supports two defaults rather than one compromise: `latency` avoids the
cross-request wait for isolated calls, while `throughput` nearly doubles a
short concurrent burst and improves batch-16 throughput by 20.8%. Batch 32
adds only 3.6% over batch 16 in the throughput profile while increasing p50
from 1.87 to 3.41 seconds, so batch 16 is the balanced ETL starting point.
Long-note ETL with batches of 8 and concurrency 4 reached 77,648 characters/s.
The older-inspired 5 ms / 16-window queue reached 58.3 documents/s at batch 16
and 55.6 at batch 32, so it is not the default for the current suite.

The host supplies the NVIDIA kernel driver. The image supplies PyTorch's CUDA
runtime. `versions.env` records the exact PyTorch version, wheel channel, CUDA
runtime, and validated platform used for this release line.
