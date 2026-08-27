# Inference and deployment

This is the operational reference for the public `meddeid-dutch-synth` and
`meddeid-english-synth` model bundles. PyTorch and TensorRT/Triton use the same
tokenizer, windowing, decoder, locale-selected post-processing, metadata
recovery, and response contract. Only the neural runtime changes.

## What is available now

Python release `0.2.0`, the public Dutch and English models, the shared public
demo, and the production `0.2.0` CPU container are available now.

| Path | Status now | Notes |
|---|---|---|
| Public model bundles | Available | `stighellemans/meddeid-dutch-synth` and `stighellemans/meddeid-english-synth` can be downloaded without authentication. |
| Python API | Available from PyPI | Install `meddeid==0.2.0`; dependencies resolve from PyPI. |
| Single-file CLI | Available from PyPI | `meddeid deidentify` uses the same local engine. |
| Canonical JSONL batch | Available from PyPI | `meddeid batch` writes results and a sidecar manifest. |
| HTTP API | Available from PyPI and GHCR | `meddeid-server` exposes single, batch, and health endpoints. It is an application server, not a complete production security boundary. |
| PyTorch devices | AMD64 and ARM64 CPU containers verified; MPS/CUDA source paths present | Device selection supports `cpu`, `mps`, and `cuda`; the production container is CPU-only. |
| PyPI install | Available | `meddeid`, `meddeid-core`, `meddeid-language-en`, and `meddeid-language-nl` are released at `0.2.0`. |
| PyTorch container | Available | `ghcr.io/stighellemans/meddeid-api:0.2.0` supports AMD64 and ARM64 and includes the pinned default model, SBOM, and provenance. |
| Compose deployment | Available | `./scripts/start-local.sh` generates authentication, pulls, starts, and health-checks the hardened local service with a browser UI. |
| TensorRT/Triton deployment | Prototype source path only | The client, export scripts, and Compose shape exist, but no TensorRT plan, populated model repository, or GPU-specific image is published. |
| Hosted demo | Available for non-sensitive text | The [MedDeID interactive demo](https://huggingface.co/spaces/stighellemans/meddeid-demo) offers Dutch and English selection. Do not submit patient information. |
| Managed clinical endpoint | Not available | No managed service for sensitive clinical text is provided. |

Install all Python interfaces from PyPI:

```bash
python -m pip install 'meddeid[server]'
```

The `server` extra includes every implemented interface. Add `==0.2.0` when an
exact package version is required. Docker users can pull the release directly:

```bash
docker pull ghcr.io/stighellemans/meddeid-api:0.2.0
```

GPU-optimized TensorRT targets still require separate hardware-specific builds
and validation as described below.

The hosted demo is a convenience for synthetic examples and is not a clinical
deployment surface. A managed endpoint is not a prerequisite for local
inference.

## Model acquisition and offline use

After the PyPI installation above, normal use downloads the self-contained
model bundle automatically on the first `from_pretrained` or CLI call and
reuses the Hugging Face cache afterwards:

```bash
meddeid deidentify note.txt
```

You do **not** need to run `hf download` first. Explicit download is useful only
when the machine will be offline, when an administrator stages a shared model
directory, or when TensorRT export needs a concrete directory:

```bash
hf download stighellemans/meddeid-dutch-synth \
  --revision <immutable-hub-commit> \
  --local-dir ./meddeid-dutch-synth
meddeid model-info --model ./meddeid-dutch-synth
```

The ordinary Hub cache is likewise reused automatically without `--offline`.
Use `--offline` only to enforce that MedDeID may not check the network and must
fail if the requested snapshot is absent. When `--model` points to an existing
local directory, the flag is unnecessary because Hub resolution is bypassed.

For a reproducible deployment, pin the immutable Hub commit in `--revision`,
`revision=`, or `MEDDEID_REVISION`. `meddeid model-info` reports the resolved
revision, model bundle hash, backend, device, profile, versions, and offline
readiness.

## Python

```python
from meddeid import Deidentifier

engine = Deidentifier.from_pretrained(
    "stighellemans/meddeid-dutch-synth",
    revision="<immutable-hub-commit>",
    device="cpu",
)
result = engine(
    "Patiënt Jan Peeters belde 0470 12 34 56.",
    metadata={
        "lang": "nl-BE",
        "patient": {"given_name": "Jan", "family_name": "Peeters"},
        "known_values": [
            {"value": "0470 12 34 56", "label": "Contactdetails"}
        ],
    },
)
print(result.deid_text)
print(result.spans)
engine.close()
```

For throughput, batch documents explicitly. Model windows from all supplied
documents are flattened into bounded runtime batches while decoding and
metadata stay isolated per document:

```python
results = engine.deidentify_many([
    ("Patiënt Jan Peeters.", {"patient": {
        "given_name": "Jan", "family_name": "Peeters"
    }}),
    ("Dr. Noor Aerts.", {"caregivers": [
        {"given_name": "Noor", "family_name": "Aerts"}
    ]}),
])
```

## What metadata does

Metadata is implemented in the `nl-BE` post-processing profile. It
is not concatenated to the note and is not an input feature of the transformer.
After neural spans are decoded, the profile can add or extend matches that the
caller already knows:

- `lang`: optional profile check; use `nl-BE` for this release.
- `patient`: an object with `given_name`, `family_name`, and optional
  `birth_date`. A valid full birth date is expanded into locale-equivalent,
  full-year `Age_Birthdate` assertions.
- `caregivers`: a list of objects with `given_name` and `family_name`. Common initials and clinical title
  forms are handled by the Dutch profile.
- `known_values`: a list of `{value, label}` assertions for identifiers,
  contacts, names, or other canonical labels. Matching tolerates common
  separators.
- `document_creation_date`: optional reference date used to convert a detected
  birthdate into a generalized age.
- `date_shift_days`: optional explicit integer shift. If omitted, `Date` and
  `Age_Birthdate` become placeholders. Zero also produces placeholders and a
  warning; MedDeID never generates an offset automatically.

### Date and age replacement

With a nonzero `date_shift_days`, parseable dates are shifted while preserving
their source format. Age expressions and birthdates with a usable document
date are generalized through one deployment-wide JSON policy shared by every
language profile. Birthdates without a usable document date fall back to the
shifted year. A configured absolute shift below the recommended minimum is
applied but reported as a structured warning.

Every output span has a `replacement` containing the exact bracketed text used
in `deid_text`. Results also include deduplicated `warnings` and `processing`
counters plus the age-policy ID, version, and SHA-256. Date shifts are measured
in days; span `begin`/`end` offsets are Unicode code points.

Load one custom policy and warning threshold when constructing an engine:

```python
deid = Deidentifier.from_pretrained(
    "stighellemans/meddeid-dutch-synth",
    age_granularity_config="age-policy.json",
    min_recommended_date_shift_days=500,
)
```

The equivalent CLI flags are `--age-granularity-config` and
`--min-recommended-date-shift-days`. These are deployment/run settings, not
request metadata; HTTP requests attempting to select them are rejected. See
[Age-granularity policy](age-granularity.md) for the complete JSON contract.

The 14 canonical labels are:

```text
Address_Location:Caregiver  Address_Location:Other
Address_Location:Patient    Age_Birthdate
Contactdetails              Date
ID:Caregiver                ID:Patient
Name:Caregiver              Name:Other
Name:Patient                Organization:Healthcare
Organization:Other          Profession
```

Only send trusted record metadata. `known_values` is a caller assertion: an
incorrect value or label can intentionally create a false-positive span. The
HTTP service validates its shape, and the language profile merges injected
spans deterministically with model spans. HTTP responses do not echo metadata,
because names and identifiers should not be duplicated unnecessarily in logs;
the canonical JSONL batch output does preserve it for controlled research
workflows.

### Measured effect

On the 300-note hospital benchmark, `meddeid-dutch-synth` rose from **95.2% to
96.7% core-PII recall** with patient/caregiver metadata. The non-PII redaction
rate also rose from **0.780% to 0.846%**. On the open synthetic benchmark it was
unchanged at 99.8%; on the 100-note primary-care set it rose from 89.9% to 90.3%
with a 1.363% to 1.369% non-PII redaction change. These are character-level
benchmark results, not a guarantee for a new institution. Metadata improves
safety when accurate names are available, but it does not make the neural model
faster and may reduce specificity.

## Canonical JSONL batch input and output

Input is one canonical document per line:

```json
{"document_id":"note-001","text":"Patiënt Jan Peeters belde 0470 12 34 56.","metadata":{"lang":"nl-BE","patient":{"given_name":"Jan","family_name":"Peeters"},"known_values":[{"value":"0470 12 34 56","label":"Contactdetails"}]},"spans":[]}
```

Run:

```bash
meddeid batch input.jsonl --output predictions.jsonl
```

The default command uses `stighellemans/meddeid-dutch-synth` and selects CUDA
when available, otherwise CPU. Pin `--revision` or set `--device` only when the
run requires that explicit control.

Output keeps ID, text, metadata, detected spans, rendered text, and the exact
language profile used for post-processing:

```json
{"document_id":"note-001","text":"Patiënt Jan Peeters belde 0470 12 34 56.","spans":[{"begin":8,"end":19,"text":"Jan Peeters","label":"Name:Patient","replacement":"[Name:Patient]"},{"begin":26,"end":39,"text":"0470 12 34 56","label":"Contactdetails","replacement":"[Contactdetails]"}],"deid_text":"Patiënt [Name:Patient] belde [Contactdetails].","metadata":{"lang":"nl-BE","patient":{"given_name":"Jan","family_name":"Peeters"},"known_values":[{"value":"0470 12 34 56","label":"Contactdetails"}]},"language_profile":{"profile_id":"nl-BE"},"warnings":[],"processing":{"date_replacement":{"mode":"placeholder","requested_shift_days":null,"minimum_recommended_abs_shift_days":366,"detected_spans":0,"shifted_spans":0,"age_generalized_spans":0,"year_fallback_spans":0,"placeholder_spans":0},"age_granularity_policy":{"policy_id":"meddeid-default","policy_version":"1","sha256":"..."}}}
```

The adjacent `.manifest.json` records hashes, immutable model identity, profile,
runtime/device, dependency versions, document/span counts, and timing. Use
`--resume` after interruption or `--overwrite` explicitly; existing output is
never silently replaced.

## HTTP API

Start the embedded PyTorch service from the PyPI installation above:

```bash
MEDDEID_DEVICE=cpu meddeid-server
```

For a model bundle containing several locales, either send `metadata.lang` per
document or configure a service fallback:

```bash
MEDDEID_MODEL=path/to/english-bundle \
MEDDEID_LANGUAGE_PROFILE=en-GB \
meddeid-server
```

The bundle declares the supported locale profiles.

Single document:

```bash
curl --fail-with-body http://localhost:8000/deidentify \
  -H 'content-type: application/json' \
  -d '{
    "text": "Patiënt Jan Peeters belde 0470 12 34 56.",
    "metadata": {
      "lang": "nl-BE",
      "patient": {"given_name": "Jan", "family_name": "Peeters"},
      "known_values": [{"value": "0470 12 34 56", "label": "Contactdetails"}]
    }
  }'
```

Response:

```json
{
  "deid_text": "Patiënt [Name:Patient] belde [Contactdetails].",
  "spans": [
    {"begin": 8, "end": 19, "text": "Jan Peeters", "label": "Name:Patient", "replacement": "[Name:Patient]"},
    {"begin": 26, "end": 39, "text": "0470 12 34 56", "label": "Contactdetails", "replacement": "[Contactdetails]"}
  ],
  "language_profile": {"profile_id": "nl-BE"},
  "warnings": [],
  "processing": {
    "date_replacement": {
      "mode": "placeholder",
      "requested_shift_days": null,
      "minimum_recommended_abs_shift_days": 366,
      "detected_spans": 0,
      "shifted_spans": 0,
      "age_generalized_spans": 0,
      "year_fallback_spans": 0,
      "placeholder_spans": 0
    },
    "age_granularity_policy": {
      "policy_id": "meddeid-default",
      "policy_version": "1",
      "sha256": "..."
    }
  }
}
```

HTTP and single-file CLI JSON omit the complete original note because the
caller already has it. Span source fragments remain for offset verification.
Python results and canonical research JSONL retain the note for validation,
evaluation, and resumable batch integrity.

Batch endpoint (recommended when the caller can group work):

```bash
curl --fail-with-body http://localhost:8000/deidentify-batch \
  -H 'content-type: application/json' \
  -d '{"documents":[
    {"document_id":"note-1","text":"Patiënt Jan Peeters.","metadata":{"patient":{"given_name":"Jan","family_name":"Peeters"}}},
    {"document_id":"note-2","text":"Dr. Noor Aerts.","metadata":{"caregivers":[{"given_name":"Noor","family_name":"Aerts"}]}}
  ]}'
```

`GET /health` includes model/profile/revision and runtime readiness. Validation
errors are HTTP 422 with a structured `detail.code` and `detail.message`;
oversized aggregate batches are HTTP 413; unavailable inference is HTTP 503.
Defaults are 20,000 characters per document, 32 documents per request, and
200,000 aggregate characters. Configure them with
`MEDDEID_MAX_INPUT_CHARS`, `MEDDEID_MAX_BATCH_DOCUMENTS`, and
`MEDDEID_MAX_BATCH_CHARS`.

Set `MEDDEID_API_KEY` to require either `Authorization: Bearer <key>` or
`X-API-Key: <key>` on inference endpoints. Set
`MEDDEID_REQUIRE_API_KEY=true` to make startup fail if the key is missing.
`/health`, `/live`, and `/` remain unauthenticated for orchestration. Additional
operational controls are:

| Setting | Default | Purpose |
|---|---:|---|
| `MEDDEID_LANGUAGE_PROFILE` | bundle default when unambiguous | Set a locale fallback for a multi-profile service. Trusted request `metadata.lang` still wins. |
| `MEDDEID_AGE_GRANULARITY_CONFIG` | packaged `meddeid-default` policy | Load one validated age-granularity JSON policy for the complete service. |
| `MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS` | 366 | Warn when a nonzero absolute date shift is smaller than this positive integer. |
| `MEDDEID_MAX_REQUEST_BYTES` | 2,000,000 | Reject oversized requests with HTTP 413 when `Content-Length` is present. Enforce the same limit at the reverse proxy. |
| `MEDDEID_MAX_CONCURRENT_REQUESTS` | 1 | Bound inference work admitted per API worker. |
| `MEDDEID_QUEUE_TIMEOUT_SECONDS` | 30 | Return HTTP 503 instead of waiting indefinitely for an inference slot. |
| `MEDDEID_DOCS_ENABLED` | true for a local process; false in the image | Enable `/docs`, `/redoc`, and `/openapi.json`. |
| `MEDDEID_UI_ENABLED` | follows docs for a local process; false in the image | Enable the simple single-note browser interface at `/ui`. |
| `MEDDEID_ACCESS_LOG` | true | Log method/path/status only; request bodies and metadata are never intentionally logged. |

Successful and error responses include `X-Request-ID`, `Cache-Control:
no-store`, and `X-Content-Type-Options: nosniff`.

## Containers

The released CPU container is the default Docker route:

```bash
./scripts/start-local.sh
```

Open `http://127.0.0.1:8000/ui`, enter the generated API key from `.env`, and
paste a note. The key remains in the browser tab rather than browser storage.

Or run Compose directly:

```bash
cp .env.example .env
# Set MEDDEID_API_KEY and MEDDEID_REQUIRE_API_KEY=true in .env.
docker compose pull
docker compose up --detach
docker compose ps
```

The image pins the public core and Dutch-profile source revisions and embeds the
pinned model under `/opt/meddeid-model`. Runtime network access is not needed.
Compose binds to `127.0.0.1`, runs as UID/GID 10001, uses a read-only root
filesystem and restricted temporary filesystem, drops every Linux capability,
sets `no-new-privileges`, bounds process count, rotates logs, and performs a
model-aware health check.

To mount a different complete model bundle without allowing Hub resolution:

```bash
export MEDDEID_MODEL_DIR=/absolute/path/to/model
docker compose -f compose.yaml -f compose.offline.yaml up --detach
```

Release `0.2.0` is published for both `linux/amd64` and `linux/arm64`. Its tag
workflow produced an SBOM and provenance and published only after authenticated
offline smoke inference and the fixable-high/critical vulnerability gate
passed. Production operators should pin the immutable digest documented in the
[production guide](production.md), not only the version tag.

## TensorRT and Triton: prototype path

TensorRT/Triton is not an end-user deployment option yet. What exists is a
tested Triton V2 HTTP client plus source scripts that export ONNX, render a
Triton configuration, and invoke `trtexec`. What is absent is the deployable
artifact: `deploy/triton/model_repository` is intentionally empty/ignored and
no TensorRT plan image has been published.

The procedure below is therefore a maintainer prototype for an NVIDIA Docker
host, not a released installation recipe. It additionally requires the source
installation shown above.

TensorRT plans are tied to the TensorRT/CUDA stack and GPU compatibility. Build
the plan on the target GPU class, record the immutable model revision and bundle
hash, and rebuild after any model, label-order, window, or runtime-stack change.

1. Download an immutable model directory and install export requirements:

   ```bash
   hf download stighellemans/meddeid-dutch-synth \
     --revision <immutable-hub-commit> \
     --local-dir ./meddeid-dutch-synth
   python -m pip install onnx
   ```

2. Build the FP16 plan and Triton model repository on an NVIDIA Docker host:

   ```bash
   ./deploy/build_triton_repository.sh \
     ./meddeid-dutch-synth deploy/triton/model_repository
   ```

   Defaults are min/opt/max shapes `1x8`, `16x256`, and `64x512`, with dynamic
   batching at 8/16/32 windows and a 5 ms queue delay. The API sends INT32 token
   tensors matching the generated `config.pbtxt`. Adjust profile environment
   variables only if the client batch limit and model maximum remain compatible.

3. Start the API gateway and Triton:

   This command remains a maintainer path until a target-specific plan has been
   built and validated.

   ```bash
   MEDDEID_REVISION=<immutable-hub-commit> \
     docker compose -f compose.triton.yaml up
   curl --fail http://localhost:8000/health
   ```

4. Optionally package that exact plan. Use a GPU-specific image name, not a
   universal `latest` tag:

   ```bash
   docker build -f deploy/triton-model.Dockerfile \
     --build-arg MODEL_REVISION=<immutable-hub-commit> \
     --build-arg BUNDLE_SHA256=<model-info-bundle-sha256> \
     --build-arg GPU_TARGET=t4-sm75-trt-24.02 \
     -t ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.1.0 .
   docker push ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.1.0
   ```

The generic workflow intentionally does not publish TensorRT plans: a plan
built for one GPU/runtime combination must not be presented as portable.

### TensorRT publication policy

- The portable PyTorch API image is the universal default.
- Maintainers build, validate, benchmark, and publish GPU-specific TensorRT
  images for supported targets, starting with NVIDIA T4.
- Institutions use `build_triton_repository.sh` only for unsupported GPUs or a
  deliberately different TensorRT/CUDA stack.
- TensorRT compilation never happens during API or Triton startup. Startup only
  loads a previously built, identified, and tested plan.

For the first supported target, release work must build the plan on that GPU
class, run PyTorch-versus-TensorRT output parity, validate startup/readiness and
single/batch HTTP requests, benchmark representative notes, publish the
GPU/runtime compatibility metadata and immutable image digest, and document
the NVIDIA driver/container-toolkit prerequisites.

## Throughput, sizing, and concurrency

Use these as starting configurations, not benchmark claims:

| Deployment | Starting point | Concurrency guidance |
|---|---:|---|
| PyTorch CPU | 4 vCPU, 8 GiB RAM | One API worker and 4 Torch threads. Prefer `/deidentify-batch`; adding workers duplicates model memory. |
| PyTorch CUDA | 1 NVIDIA GPU with at least 8 GiB | One worker per GPU. Increase `MEDDEID_WINDOW_BATCH_SIZE` until memory or tail latency becomes limiting. |
| Triton/TensorRT | T4 16 GiB or a plan rebuilt for the chosen GPU | Two lightweight API workers, one Triton model instance, 16-window client batches, Triton dynamic batching. |

Within one `Deidentifier`, a lock protects model and pipeline state. A single
request can still use all available batch capacity through `deidentify_many` /
`/deidentify-batch`; independent HTTP requests do not get silently combined by
the Python process. Triton can dynamically combine simultaneous window calls
from multiple API workers.

No release-grade latency, memory, or saturation table has been published yet.
Measure on the actual note-length distribution and target hardware, report
p50/p95/p99 latency plus documents and characters per second, and test both
metadata-on and metadata-off payloads. Do not tune only on short example notes.
