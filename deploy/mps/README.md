# Native Apple MPS benchmark

Apple MPS is a native macOS PyTorch backend, not a container image and not a
TensorRT plan. MedDeID keeps the same model bundle, windowing, post-processing,
API, and provenance behavior while moving neural inference to the Apple GPU.

## Current validation

The checked summary in `benchmark-summary.json` was collected on an Apple M4
Pro MacBook Pro with 14 CPU cores, 20 GPU cores, 48 GB unified memory, macOS
26.6.2, Python 3.12.11, and PyTorch 2.13.0. Both devices used native FP32 eager
inference, one server worker, and 32-window request-local batches. Every timing
case was run three times and the median is reported.

MPS matched CPU semantics on all 300 documents in the pinned public synthetic
fixture, with confidence fields excluded under the same parity policy used by
the TensorRT gate. It was 1.6–2.0 times faster than this Mac's CPU across the
interactive, ETL, and long-note cases. Batch 16 reached 40.08 documents/s;
batch 32 reached 39.51 documents/s while roughly doubling p50 HTTP latency, so
16 is the native Mac ETL starting point.

The latency profile keeps eager request-local execution. In the throughput
profile, a 1 ms cross-request microbatch increased an eight-way short-note
burst from 113.70 to 167.05 documents/s, batch-1 ETL from 39.60 to 40.72, and
batch-16 ETL from 40.08 to 42.10. Automatic throughput microbatching is
therefore enabled for both MPS and CUDA, while the latency profile never waits
for an unrelated request.

`torch.compile(mode="default", dynamic=True)` was not useful on this target. It
added 6.44 seconds for the first short graph and another 7.39 seconds for the
first batch-16 graph. Warm short throughput fell from 92.83 to 70.74
documents/s, and the matched batch-16/concurrency-1 comparison fell from 40.98
to 12.70 documents/s. MPS therefore stays eager by default.

The macOS `footprint` tool reported a 2,762 MB peak physical footprint after
the long-note MPS matrix, including unified graphics allocations. This is a
host measurement, not a container limit or a dedicated-VRAM requirement.

## Reproduce

Install the server extra natively on an Apple-silicon Mac, stage the exact
model and public fixture pins shared with the GPU release gate, and verify MPS:

```bash
python -m pip install '.[server]'
./deploy/stage_triton_model.sh /tmp/meddeid-mps-model
./deploy/stage_benchmark_fixture.sh /tmp/meddeid-mps-benchmark
python -c 'import torch; assert torch.backends.mps.is_available()'
```

Run the service with an explicit device and FP32 eager inference:

```bash
MEDDEID_MODEL=/tmp/meddeid-mps-model \
MEDDEID_OFFLINE=true \
MEDDEID_DEVICE=mps \
MEDDEID_TORCH_PRECISION=fp32 \
MEDDEID_TORCH_COMPILE_MODE=off \
MEDDEID_WINDOW_BATCH_SIZE=32 \
MEDDEID_SERVING_PROFILE=latency \
MEDDEID_API_KEY=<secret> \
MEDDEID_REQUIRE_API_KEY=true \
meddeid-server
```

Use `deploy/benchmark_http.py` for identical HTTP cases. Generate the long
performance-only shapes from the checked public synthetic parity fixture:

```bash
python deploy/make_benchmark_lengths.py \
  deploy/triton/parity-fixture.jsonl /tmp/meddeid-long.jsonl \
  --length 8000 --length 12000 --length 18000
```

These measurements establish a useful native Mac path; they do not claim that
all Apple chips have the same capacity. Re-run the matrix on each supported Mac
class before making a deployment-size or latency promise.
