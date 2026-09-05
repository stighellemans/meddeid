# Inference and deployment

This is the operational reference for the public `meddeid-dutch-synth` and
`meddeid-english-synth` model bundles. PyTorch and TensorRT/Triton use the same
tokenizer, windowing, decoder, locale-selected post-processing, metadata
recovery, and response contract. Only the neural runtime changes.

## What is available now

Python release `0.3.0`, the public Dutch and English models, the shared public
demo, and the production `0.3.0` CPU container are available now.

| Path | Status now | Notes |
|---|---|---|
| Public model bundles | Available | `stighellemans/meddeid-dutch-synth` and `stighellemans/meddeid-english-synth` can be downloaded without authentication. |
| Python API | Available from PyPI | Install `meddeid==0.3.0`; dependencies resolve from PyPI. |
| Single-file CLI | Available from PyPI | `meddeid deidentify` uses the same local engine. |
| Canonical JSONL batch | Available from PyPI | `meddeid batch` writes results and a sidecar manifest. |
| HTTP API | Available from PyPI and GHCR | `meddeid-server` exposes single, batch, and health endpoints. It is an application server, not a complete production security boundary. |
| PyTorch devices | AMD64 and ARM64 CPU image; native Apple MPS; AMD64 PyTorch/CUDA image recipe | Device selection supports `cpu`, `mps`, and `cuda`; MPS was validated natively on an M4 Pro, while the CUDA image has its own T4 validation and publishing gate. |
| PyPI install | Available | `meddeid==0.3.0` is compatible with `meddeid-core`, `meddeid-language-en`, and `meddeid-language-nl` at `0.2.0`. |
| PyTorch CPU container | Available | `ghcr.io/stighellemans/meddeid-api:0.3.0` supports AMD64 and ARM64 and explicitly includes the pinned Dutch synthetic model, SBOM, and provenance. |
| PyTorch CUDA container | Release candidate | The AMD64 tag contract is `ghcr.io/stighellemans/meddeid-api:<version>-cuda<runtime>`; `compose.cuda.yaml` requests the selected NVIDIA device and refuses CPU fallback. |
| Local Compose evaluation | Available | `./scripts/start-local.sh` generates authentication, pulls, starts, and health-checks the local service with a browser UI. Production operators use Compose directly. |
| TensorRT/Triton deployment | Target-specific source release-candidate kit | Pinned export/build scripts, build manifest, image recipe, Compose wiring, and parity gate are present; no GPU-specific image is published until its target evidence passes. |
| Hosted demo | Available for non-sensitive text | The [MedDeID interactive demo](https://huggingface.co/spaces/stighellemans/meddeid-demo) offers Dutch and English selection. Do not submit patient information. |
| Managed clinical endpoint | Not available | No managed service for sensitive clinical text is provided. |

Install all Python interfaces from PyPI:

```bash
python -m pip install 'meddeid[server]'
```

The `server` extra includes every implemented interface. Add `==0.3.0` when an
exact package version is required. Docker users can pull the release directly:

```bash
docker pull ghcr.io/stighellemans/meddeid-api:0.3.0
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
meddeid models
meddeid deidentify note.txt \
  --model stighellemans/meddeid-dutch-synth
```

Model selection is mandatory. The public entries are synthetic-data baselines,
not institution-validated clinical models. `meddeid models` shows their
languages, regional profiles, and scope before an inference run.

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
revision, model bundle hash, profiles, package versions, actual cache or local
root, and the complete model-file inventory. It is the detailed local
administrator view. By default it validates the bundle without constructing an
inference runtime. Add `--verify-runtime` to load the weights and verify the
selected backend and device:

```bash
meddeid model-info \
  --model ./meddeid-dutch-synth \
  --device cpu \
  --verify-runtime
```

The output reports `runtime.checked: false` for the quick inspection and
`runtime.checked: true` for full verification. The command never saves a model
as a default for later commands. Because its output contains local paths and
environment details, treat saved copies as operationally sensitive.

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
meddeid batch input.jsonl \
  --output predictions.jsonl \
  --model stighellemans/meddeid-dutch-synth
```

For scripts where named arguments are clearer, the equivalent command starts
with `meddeid batch --input input.jsonl`. `deidentify` accepts the same two
input forms. Supplying both forms is an error. Input existence, file type, and
readability are checked before any model resolution or loading.

Automatic device selection prefers CUDA, then Apple MPS, and falls back to CPU.
Pin `--revision` or set `--device` when the run requires explicit control.

Native MPS on the measured M4 Pro preserved semantics over all 300 pinned
public fixture documents and was 1.6--2.0 times faster than this Mac's CPU
across interactive, batched ETL, and long-note HTTP workloads. Use eager FP32
execution, one worker, batch 16 as the initial ETL request size, and the
`throughput` profile for sustained concurrency. The profile enables a bounded
1 ms cross-request microbatch on MPS; the `latency` profile keeps isolated
requests queue-free. `torch.compile` regressed the measured MPS workloads and
remains off. MPS is a native macOS path, not an MPS container image. See the
[reproducible MPS record](../deploy/mps/README.md) and
[production comparison](production.md#measured-apple-mps-snapshot).

Output keeps ID, text, metadata, rendered text, detected spans, processing
details, warnings, and per-result provenance. The selected language profile is
part of provenance because it describes how that specific result was produced:

```json
{"document_id":"note-001","text":"Patiënt Jan Peeters belde 0470 12 34 56.","metadata":{"lang":"nl-BE","patient":{"given_name":"Jan","family_name":"Peeters"},"known_values":[{"value":"0470 12 34 56","label":"Contactdetails"}]},"deid_text":"Patiënt [Name:Patient] belde [Contactdetails].","spans":[{"begin":8,"end":19,"text":"Jan Peeters","label":"Name:Patient","replacement":"[Name:Patient]"},{"begin":26,"end":39,"text":"0470 12 34 56","label":"Contactdetails","replacement":"[Contactdetails]"}],"processing":{"date_replacement":{"mode":"placeholder","requested_shift_days":null,"minimum_recommended_abs_shift_days":366,"detected_spans":0,"shifted_spans":0,"age_generalized_spans":0,"year_fallback_spans":0,"placeholder_spans":0},"age_granularity_policy":{"policy_id":"meddeid-default","policy_version":"1","sha256":"..."}},"warnings":[],"provenance":{"contract_version":"meddeid.inference-provenance.v1","software":{"name":"meddeid","version":"0.3.0"},"model":{"name":"meddeid-dutch-synth","version":"1","resolved_revision":"<immutable-hub-commit>","bundle_sha256":"..."},"language_profile":{"profile_id":"nl-BE"}}}
```

The adjacent `.manifest.json` records hashes, immutable model identity, profile,
runtime/device, dependency versions, document/span counts, and timing. Use
`--resume` after interruption or `--overwrite` explicitly; existing output is
never silently replaced.

## HTTP API

Start the embedded PyTorch service from the PyPI installation above:

```bash
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth \
MEDDEID_DEVICE=cpu \
meddeid-server
```

The service also requires an explicit model. The published CPU container is a
model-specific artifact and already sets `MEDDEID_MODEL` to its embedded,
pinned Dutch synthetic bundle.

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
  },
  "warnings": [],
  "provenance": {
    "contract_version": "meddeid.inference-provenance.v1",
    "software": {
      "name": "meddeid",
      "version": "0.3.0"
    },
    "model": {
      "name": "meddeid-dutch-synth",
      "version": "1",
      "resolved_revision": "<immutable-hub-commit>",
      "bundle_sha256": "..."
    },
    "language_profile": {"profile_id": "nl-BE"}
  }
}
```

This is the contract starting with release `0.3.0`. Release `0.2.0` used the
previous HTTP schema with a top-level `language_profile`; consumers upgrading
to `0.3.0` should read the nested `provenance.language_profile` field.

Swagger's names such as `additionalProp1` are generated placeholders, not real
MedDeID fields. They mean the OpenAPI schema declared an arbitrary key/value
dictionary. The current HTTP schema explicitly types every request-metadata,
span, and provenance field, so those placeholders are absent. Unknown HTTP
request fields are rejected with status 422; this catches misspellings and
prevents callers from assuming unused metadata affected inference. Canonical
research JSONL remains independently extensible outside the serving contract.

Single-result JSON is deliberately flat and serialized in this order:
`deid_text`, `spans`, `processing`, `warnings`, `provenance`. Only provenance is
nested as a separate concern. It contains the model identity, selected language
profile, and MedDeID version needed by an inference consumer. Runtime, package,
cache, and filesystem details are deliberately excluded. Operators obtain
those details locally with `meddeid model-info` or from the batch manifest.

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

`GET /health` reports readiness, the safe model name/version, enabled language
profiles, and the active serving profile, gateway-batching state, and admission
limit. It deliberately omits revisions, paths, runtime internals, and other
environment details. Validation
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
| `MEDDEID_ALLOWED_MODELS` | unrestricted | Optional comma-separated allowlist of exact `MEDDEID_MODEL` values. Startup fails before model loading when the selected Hub ID or local directory is absent. |
| `MEDDEID_ALLOWED_LANGUAGE_PROFILES` | all profiles declared by the model | Optional comma-separated allowlist of request-selectable regional profiles. Unknown configured profiles fail startup; disallowed request profiles return HTTP 422. A single allowed profile becomes the service fallback when none is configured. |
| `MEDDEID_AGE_GRANULARITY_CONFIG` | packaged `meddeid-default` policy | Load one validated age-granularity JSON policy for the complete service. |
| `MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS` | 366 | Warn when a nonzero absolute date shift is smaller than this positive integer. |
| `MEDDEID_MAX_REQUEST_BYTES` | 2,000,000 | Reject oversized requests with HTTP 413 when `Content-Length` is present. Enforce the same limit at the reverse proxy. |
| `MEDDEID_SERVING_PROFILE` | `latency` | Select process-wide low-wait or bounded-throughput scheduling. Use a separate replica pool when both workloads coexist. |
| `MEDDEID_MICROBATCH_ENABLED` | `auto` | In throughput mode, automatically batch concurrent accelerated PyTorch windows on CUDA or MPS. TensorRT keeps the measured request-local path; `true` forces gateway microbatching for an explicit fan-out experiment. |
| `MEDDEID_MAX_CONCURRENT_REQUESTS` | `auto` | Bound inference work admitted per API worker. Auto selects 1 for latency, 16 for accelerated PyTorch throughput, and 8 per TensorRT gateway worker. |
| `MEDDEID_QUEUE_TIMEOUT_SECONDS` | 30 | Return HTTP 503 instead of waiting indefinitely for an inference slot. |
| `MEDDEID_DOCS_ENABLED` | true for a local process; false in the image | Enable `/docs`, `/redoc`, and `/openapi.json`. |
| `MEDDEID_UI_ENABLED` | follows docs for a local process; false in the image | Enable the simple single-note browser interface at `/ui`. |
| `MEDDEID_ACCESS_LOG` | true | Log method/path/status only; request bodies and metadata are never intentionally logged. |

Administrators running the Python service directly can keep these settings in
one validated environment file. Edit the copied file and set or inject a random
`MEDDEID_API_KEY` before starting it:

```bash
cp server.env.example meddeid-server.env
chmod 600 meddeid-server.env
meddeid-server --env-file meddeid-server.env
```

The format is one literal `KEY=VALUE` per line. Blank lines, comments, an
optional `export` prefix, and single- or double-quoted values are accepted;
variable expansion is deliberately not performed. Unknown or duplicate keys
fail fast. Existing process variables take precedence so secrets can come from
a service manager. Treat any file containing the API key as sensitive and do
not commit it. Compose users can instead select a file with
`docker compose --env-file <path> up --detach`.

The model allowlist controls the configured source; it does not replace
`MEDDEID_REVISION` or an immutable container digest when an exact model release
must be pinned. Model selection is fixed at server startup and cannot be changed
by an inference request. Language-profile selection can be per request, which is
why that allowlist is also enforced on every document.

Successful and error responses include `X-Request-ID`, `Cache-Control:
no-store`, and `X-Content-Type-Options: nosniff`.

## Containers

The released CPU container is the default Docker route:

```bash
./scripts/start-local.sh \
  --model stighellemans/meddeid-dutch-synth \
  --language-profile nl-BE
```

Open `http://127.0.0.1:8000/ui` and paste the generated API key. The launcher
opens the browser when supported and copies the key to the clipboard. It also
prints a command that restores the key if the clipboard changes. The key
remains in the browser tab rather than browser storage.

Or run Compose directly:

```bash
cp .env.example .env
# Set MEDDEID_API_KEY and MEDDEID_REQUIRE_API_KEY=true in .env.
docker compose pull
docker compose up --detach
docker compose ps
```

The image pins the public core and language-package source revisions and embeds
a fallback model under `/opt/meddeid-model`. The local launcher explicitly
selects the served Hub model and profile and persists downloads in a Docker
volume. Direct production deployments can retain the embedded offline model.
Compose binds to `127.0.0.1`, runs as UID/GID 10001, uses a read-only root
filesystem and restricted temporary filesystem, drops every Linux capability,
sets `no-new-privileges`, bounds process count, rotates logs, and performs a
model-aware health check.

To mount a different complete model bundle without allowing Hub resolution:

```bash
export MEDDEID_MODEL_DIR=/absolute/path/to/model
docker compose -f compose.yaml -f compose.offline.yaml up --detach
```

Release `0.3.0` is published for both `linux/amd64` and `linux/arm64`. Its tag
workflow produced an SBOM and provenance and published only after authenticated
offline smoke inference and the fixable-high/critical vulnerability gate
passed. Production operators should pin the immutable digest documented in the
[production guide](production.md), not only the version tag.

## PyTorch CUDA image

The portable GPU artifact uses the same Dockerfile, API process, embedded
model, and hardening controls as the CPU image, but selects PyTorch's official
CUDA 12.9 wheel and sets `MEDDEID_DEVICE=cuda`. FP16 autocast with eager
execution is its measured default. The image omits PyTorch's compiler-only
Triton package, headers, and static archives. Its version contract is:

```text
ghcr.io/stighellemans/meddeid-api:<meddeid-version>-cuda<cuda-version>
```

The initial AMD64 candidate is
`ghcr.io/stighellemans/meddeid-api:0.3.0-cuda12.9`. A host must provide a
compatible NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. Copy
`.env.cuda.example`, inject a real API key, and use the CUDA overlay:

```bash
docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

The release gate performs a real CUDA operation and authenticated single and
batch API inference on NVIDIA T4 before publishing the image with SBOM and
provenance. It also verifies that the service runs as UID/GID 10001 with a
read-only root filesystem. The image does not silently fall back to CPU.

## TensorRT and Triton: target-specific release candidate

The repository contains a complete **source delivery kit**, but no GPU image is
considered released until it has passed the documented target-GPU gate and its
immutable digest and evidence have been published. Compose is intentionally not
used to compile a plan at startup.

A complete delivery contains both:

1. a target-specific Triton image with `model.plan`, `config.pbtxt`, and a
   checksummed build manifest; and
2. a gateway-only image with the normal MedDeID API contract and language logic
   but no PyTorch or checkpoint weights, connected through
   `compose.triton.yaml` with authentication, health checks, and an internal
   inference network.

Compose alone lacks the compiled model. A model image alone lacks the safe
gateway wiring. TensorRT plans are tied to the TensorRT/CUDA stack and GPU
compatibility, so do not publish a universal `latest` image.

On the target Linux NVIDIA host, install the source environment and run:

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
  t4-sm75 0
./deploy/build_triton_image.sh \
  deploy/triton/model_repository \
  ghcr.io/stighellemans/meddeid-triton-t4-sm75:0.3.0-trt26.07-fp16 \
  t4-sm75
```

The pinned stack, NVIDIA container-composer revision, and model contract live in
`deploy/triton/versions.env`. The resulting Triton runtime contains only the
TensorRT backend. `deploy/build_triton_gateway_image.sh` produces the separate
weight-free gateway and verifies the exact Hub revision before projecting away
the checkpoint.
Defaults are min/opt/max shapes `1x8`, `16x256`, and `64x512`. The latency
and throughput model configurations use the measured request-local path without
cross-request queueing. A non-negative queue delay remains an explicit build
option for a target/workload-specific dynamic-batching experiment. The gateway
uses 64-window request-local chunks.
The API sends INT32 token tensors and receives FP32 logits through Triton's
binary HTTP extension, matching the generated `config.pbtxt`.

Copy `.env.triton.example` to the ignored `.env.triton`, replace its secret,
and start the TensorRT candidate beside the PyTorch reference:

```bash
docker compose --env-file .env.triton \
  -f compose.triton.yaml \
  -f compose.triton.validation.yaml \
  up --detach
set -a && source .env.triton && set +a
python deploy/validate_triton_parity.py deploy/triton/parity-fixture.jsonl
```

The parity runner ignores floating-point confidence fields but requires exact
model identity, de-identified text, span boundaries/labels/replacements,
language profile, warnings, and processing metadata. The complete build,
validation, publication-unit, and compatibility instructions are in the
[TensorRT/Triton delivery kit](../deploy/triton/README.md).

### TensorRT publication policy

- The portable PyTorch API image is the universal default.
- The checked catalog has one ready target, NVIDIA T4 (`t4-sm75`). A10G
  (`a10g-sm86`) and L4 (`l4-sm89`) can be built and validated on request but
  are not supported image claims yet.
- Maintainers use the same target-driven workflow to build, validate,
  benchmark, and publish GPU-specific TensorRT images. It refuses publication
  for an `on-request` target until its evidence is reviewed and its catalog
  status is promoted to `ready`.
- Institutions use `build_triton_repository.sh` only for a requested target,
  an unsupported GPU, or a deliberately different TensorRT/CUDA stack.
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
| PyTorch CUDA | 1 NVIDIA GPU with at least 8 GiB | One worker per GPU, FP16 autocast, 32-window batches, and compilation off. |
| PyTorch MPS | Apple silicon Mac, measured on an M4 Pro with 48 GiB unified memory | Native Python installation, one worker, FP32 eager execution, and batch 16 as the measured ETL starting point. Use throughput microbatching only for sustained concurrency. |
| Triton/TensorRT | T4 16 GiB or a plan rebuilt for the chosen GPU | Four weight-free API workers on the measured 4-vCPU host, one Triton model instance, 64-window request-local chunks, and binary tensors. Dynamic or nested batching is target/workload-specific rather than the default. |

The local PyTorch runtime serializes inference within each `Deidentifier`. A
single request can still use all available batch capacity through
`deidentify_many` / `/deidentify-batch`. The remote Triton runtime safely admits
concurrent request-local batches from the weight-free gateway workers without
replacing any API or post-processing business logic. Dynamic and nested queues
remain opt-in experiments because the measured default was faster for the
representative ETL mix and isolated requests.

The Azure T4 validation snapshot in [Production deployment](production.md#measured-t4-snapshot)
records image pull proxies, readiness, p50/p95, end-to-end
throughput, and peak GPU memory for the current suite. It is a comparative
baseline, not a capacity promise. Re-run it on the actual note-length
distribution and target hardware, and test both metadata-on and metadata-off
payloads. Do not tune only on short example notes.
