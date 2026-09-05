# MedDeID

MedDeID is a local-first, multilingual framework for clinical text
de-identification. It provides a Python API, command-line interface, batch
processor, and optional FastAPI service for compatible model bundles and
locale-specific language profiles. Public models support Dutch and English;
data generation, training, evaluation, and annotation packages are not required
for inference.

For cross-suite navigation and task-oriented guidance, see the
[MedDeID website and documentation](https://stighellemans.github.io/meddeid/). This repository remains
authoritative for inference APIs, CLI options, service settings, and deployment.

## Language support

| Language | Language profiles | Public model |
|---|---|---|
| Dutch | `nl-BE` in the current public bundle | [`stighellemans/meddeid-dutch-synth`](https://huggingface.co/stighellemans/meddeid-dutch-synth) |
| English | `en-GB`, `en-US` | [`stighellemans/meddeid-english-synth`](https://huggingface.co/stighellemans/meddeid-english-synth) |
| Additional languages | Pluggable | Requires a compatible model bundle and language profile |

Model bundles pin their supported profiles. MedDeID selects a regional profile
explicitly and never guesses between locales such as British and US English.

## Easiest start: Python CLI

For the shortest first run, install the released package and use the CLI. This
does not require writing Python code or installing Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install meddeid
meddeid models
meddeid deidentify note.txt \
  --model stighellemans/meddeid-dutch-synth
```

`meddeid models` ends by stating where to use the selected model ID.

## Try MedDeID in your browser

Install and start Docker Desktop, then clone this repository once:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
```

Choose Dutch:

```bash
./scripts/start-local.sh \
  --model stighellemans/meddeid-dutch-synth \
  --language-profile nl-BE
```

Or choose English:

```bash
./scripts/start-local.sh \
  --model stighellemans/meddeid-english-synth \
  --language-profile en-GB
```

The browser opens with the API key already filled in. Paste a note and select
**De-identify**. Stop MedDeID with `./scripts/stop-local.sh`.

Developers testing source changes can add `--build`. For a shared service, see
[Production deployment](docs/production.md).

## Python HTTP service

Add the HTTP dependencies when another application needs the service or when
deploying MedDeID as an HTTP service in a production environment:

```bash
python -m pip install 'meddeid[server]'
```

This installs the service runtime, not the surrounding production controls.
See [Production deployment](docs/production.md) for authentication, TLS,
network, resource, monitoring, and validation requirements, and
[Inference and deployment](docs/inference.md) for the availability matrix.

For dataset review, benchmarking, evaluation, training, synthetic-data, and
contributor workflows, install the suite front door with the matching extras:

```bash
python -m pip install 'meddeid[research]'
# Language/profile and model-bundle contributors:
python -m pip install 'meddeid[contributor]'
```

## Guided suite workflows

Use one entry point when you know the outcome but do not yet know which suite
components or optional stages apply:

```bash
meddeid start
meddeid status ./my-workflow
meddeid next ./my-workflow
```

`start` first groups the suite into six familiar goals: de-identify text,
prepare data, train, evaluate, deploy, or contribute. A second menu appears only
when that goal has several possible workflows. Choices are numbered and `?`
explains why a scientific decision is being requested. `next` runs exactly one
eligible stage and stops instead of guessing an unanswered branch. Hardware,
browser runtime, and other operational choices are requested only when needed.

Run `status` or `next` without a path from anywhere inside the workflow
directory. Add `--details` for every technical stage and exclusion reason.

Every workspace contains a checksummed `meddeid.workflow.v1` manifest. Inspect
the rationale and the underlying component command at any time:

```bash
meddeid status ./my-workflow --details
meddeid workflow explain ./my-benchmark
meddeid workflow run ./my-benchmark score --dry-run
```

The existing `meddeid workflow ...` commands remain the advanced and automation
interface. Changing a scientific decision after work begins shows which stages
become invalid and requires `configure --yes` before outputs are archived.

## Quick start

First review the available public baselines and their validation scope:

```bash
meddeid models
```

Then select the model explicitly. MedDeID does not silently choose Dutch or a
synthetic baseline for you:

```bash
meddeid deidentify note.txt \
  --model stighellemans/meddeid-dutch-synth
```

```python
from meddeid import Deidentifier

deid = Deidentifier.from_pretrained("stighellemans/meddeid-dutch-synth")
result = deid("Patiënt Alex Voorbeeld kwam op controle.")
print(result.deid_text)
deid.close()
```

Use the public English model by selecting its regional profile:

```bash
meddeid deidentify note.txt \
  --model stighellemans/meddeid-english-synth \
  --language-profile en-GB
```

Dates use placeholders unless the caller explicitly supplies
`metadata.date_shift_days`. A nonzero shift produces deterministic shifted
dates; zero produces placeholders and a structured warning. One declarative
age-granularity JSON policy is loaded for the complete engine, independent of
the selected language profile.

The self-contained model bundle is downloaded and cached on first use. Document
text is processed locally and is not sent to Hugging Face. Quickly inspect the
resolved model, immutable revision, language profiles, files, and package
versions with:

```bash
meddeid model-info --model stighellemans/meddeid-dutch-synth
```

The default inspection does not load the model weights into an inference
runtime. Use `--verify-runtime` when an administrator also needs to verify that
the configured backend and device can initialize:

```bash
meddeid model-info \
  --model stighellemans/meddeid-dutch-synth \
  --verify-runtime
```

`model-info` is read-only and does not set a persistent default. Each inference
command still names its model explicitly. It is also the detailed local
administrator view: model/cache paths, file inventory, and complete package
versions are reported, so saved output should be treated as operationally
sensitive. `runtime.checked` distinguishes inspection from runtime verification.
When no revision is supplied, the command reports
`requested_revision: "latest"` together with the immutable
`resolved_revision` that was resolved.

CLI JSON and HTTP results remain flat and use the logical order `deid_text`,
`spans`, `processing`, `warnings`, `provenance`. The nested `provenance` field
contains only the MedDeID version, model identity, and selected language
profile. Runtime, dependencies, cache details, and filesystem paths stay in
`model-info` and batch manifests. Clinical text is not sent to Hugging Face.

For offline or air-gapped use, download an immutable snapshot in advance:

```bash
hf download stighellemans/meddeid-dutch-synth \
  --revision <immutable-hub-sha> \
  --local-dir ./meddeid-dutch-synth
meddeid deidentify note.txt --model ./meddeid-dutch-synth
```

Use `revision=` in Python or `--revision` on the CLI to pin reproducible
deployments.

## Batch inference

Canonical MedDeID JSONL can be processed directly. The batch command preserves
document IDs and order, supports interruption-safe resume, and writes a sidecar
manifest with input, output, model, profile, runtime, and timing metadata.
Existing output is never overwritten implicitly.

```bash
meddeid batch project/splits/test.jsonl \
  --output predictions.jsonl \
  --model stighellemans/meddeid-dutch-synth
```

The input path may instead be written explicitly as
`meddeid batch --input project/splits/test.jsonl ...`. Both forms are
equivalent for `batch` and `deidentify`; do not supply both in one command.
MedDeID checks that the input is a readable file before resolving or loading
the model, so path mistakes fail immediately.

The model is downloaded once and reused from the Hugging Face cache. Each
online run checks whether the selected Hub revision changed; use `--offline`
to prohibit that check. Automatic device selection prefers CUDA, then Apple
MPS, and falls back to CPU. Use `--device` to override that order. Native MPS
on the measured M4 Pro preserved semantics over all 300 public fixture notes
and ran 1.6--2.0 times faster than native CPU across interactive, ETL, and
long-note workloads. It is a native Python path rather than a container image.

## HTTP service

PyTorch inference works on CPU, Apple MPS, and CUDA:

```bash
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth \
MEDDEID_DEVICE=cpu \
meddeid-server
```

Multi-profile services can set a locale fallback once while still allowing
trusted `metadata.lang` on a request to override it:

```bash
MEDDEID_MODEL=stighellemans/meddeid-english-synth \
MEDDEID_LANGUAGE_PROFILE=en-GB \
meddeid-server
```

The bundle declares its post-processing locales; users choose only the locale.
The browser UI shows the active model and profile, and enables profile
selection when the model supports more than one.

For an authenticated service:

```bash
export MEDDEID_MODEL=stighellemans/meddeid-dutch-synth
export MEDDEID_API_KEY='<random secret>'
export MEDDEID_REQUIRE_API_KEY=true
meddeid-server
```

For a reproducible direct installation, keep the non-secret settings in the
provided environment-file template. Edit the copied file to select the model,
profile policy, and API-key handling, then run:

```bash
cp server.env.example meddeid-server.env
chmod 600 meddeid-server.env
meddeid-server --env-file meddeid-server.env
```

Unknown and duplicate settings are rejected. Existing process environment
variables take precedence, so a service manager can inject `MEDDEID_API_KEY`
without storing it in the file. Docker Compose continues to use `.env` or
`docker compose --env-file <path>`.

The service provides:

- `POST /deidentify` for one document;
- `POST /deidentify-batch` for throughput-oriented batches; and
- `GET /health` for minimal readiness and enabled model/profile information.

Server operators can optionally restrict startup and request profile selection:

```bash
export MEDDEID_ALLOWED_MODELS=stighellemans/meddeid-dutch-synth
export MEDDEID_ALLOWED_LANGUAGE_PROFILES=nl-BE
```

The model allowlist contains exact accepted `MEDDEID_MODEL` values. The profile
allowlist limits the regional profiles that requests may select. Both are
optional and do not affect ordinary Python, CLI, or batch use.

For a shared service, choose CPU for the simplest portable deployment, PyTorch
CUDA when the NVIDIA GPU may vary, or a target-specific TensorRT runtime for a
fixed, optimized GPU target. Native Apple MPS is also available through the
Python installation. All paths retain the same API, tokenization, decoding,
and locale-selected post-processing contract. The PyTorch CUDA image runs with:

```bash
docker compose \
  --env-file .env.cuda \
  -f compose.yaml \
  -f compose.cuda.yaml \
  up --detach meddeid
```

TensorRT is optimized for a specific GPU. The first supported target is NVIDIA
T4; choose CUDA instead when the GPU model may vary. For an optimized A10G, L4,
or another target, contact
[stig.hellemans@uantwerpen.be](mailto:stig.hellemans@uantwerpen.be) without
sending sensitive data.

Use `/deidentify` for individual notes and `/deidentify-batch` for planned
batches. If interactive traffic and large batch jobs run simultaneously,
operate separate service instances so batch work cannot delay interactive
requests.

See the reader-focused [production deployment
guide](site/docs/workflows/production-deployment.md) for setup. Benchmark
evidence and advanced tuning remain in the [technical production
reference](docs/production.md), [CUDA guide](deploy/pytorch-cuda/README.md), and
[TensorRT guide](deploy/triton/README.md).

## Language profiles and metadata

The current public Dutch model bundle declares `nl-BE`. The Dutch language
package also provides `nl-NL`, but it can only be selected when a compatible
model bundle declares that profile. The public English model bundle supports
both `en-GB` and `en-US`; `metadata.lang` or `--language-profile` selects the
regional resources and post-processing. Bare `en` is rejected because it does
not choose a regional profile.

Model bundles pin installed profiles provided through
`meddeid.language_profiles` entry points, with no source-tree fallback. For a
multi-profile bundle, document metadata is used first, followed by an explicit
load-time default and then a single-profile bundle default; MedDeID never
guesses between multiple profiles. Additional languages require a compatible
model bundle and language profile. See
[Language profile selection](docs/language-profile-selection.md).

Optional trusted metadata can recover known patient or caregiver names and
birth-date representations or other known values after neural inference.
Metadata is not concatenated to the
note or sent to the model as an input feature. Incorrect metadata can create false-positive
redactions, so callers must validate it.

Belgian DEDUCE is an independent comparison system and is not installed by this
package.

## Privacy and limitations

Local processing reduces data movement but does not guarantee anonymity.
Validate the model on representative data from the intended setting, monitor
both missed PII and unnecessary redaction, and use human review where errors can
create material privacy risk. Secure cached models, inputs, outputs, manifests,
and service logs according to your organization’s requirements.

## Development

```bash
python -m pip install \
  -e ../meddeid-core \
  -e ../meddeid-language-nl \
  -e '.[dev]'
pytest
```

## Licence

AGPL-3.0-only. Model weights are distributed separately under the terms stated
in their model card.
