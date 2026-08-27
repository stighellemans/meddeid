# meddeid

Local inference for the `meddeid-dutch-synth` clinical de-identification model,
available as a Python API, command-line interface, batch processor, and optional
FastAPI service. Data generation, training, evaluation, and annotation packages
are not required for inference.

For cross-suite navigation and task-oriented guidance, see the
[MedDeID documentation](https://stighellemans.github.io/meddeid.github.io/). This repository remains
authoritative for inference APIs, CLI options, service settings, and deployment.

## Easiest start: Docker

Docker is the recommended path for people who do not need the Python API. The
published image contains the pinned model, starts without Hub access, runs as a
non-root user, and is exposed only on your own computer by default.

1. Install and start Docker Desktop.
2. Clone this repository and start MedDeID:

   ```bash
   git clone https://github.com/stighellemans/meddeid.git
   cd meddeid
   ./scripts/start-local.sh
   ```

The script generates a private API key, pulls the published multi-architecture
image, starts the service, waits for the model to become ready, and prints the
browser address. Open
<http://127.0.0.1:8000/ui> and paste the `MEDDEID_API_KEY` value from `.env`.
The page lets you de-identify and copy a note without writing code. Technical
API documentation is at <http://127.0.0.1:8000/docs>. Stop the service with
`./scripts/stop-local.sh`.

See [Production deployment](docs/production.md) before exposing the service to
another machine or processing real clinical data.

## Python install

Install the released Python API, CLI, batch runner, and HTTP service from PyPI:

```bash
python -m pip install 'meddeid[server]'
```

See [Inference and deployment](docs/inference.md) for the exact availability
matrix and operational guidance.

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

```bash
meddeid deidentify note.txt
```

```python
from meddeid import Deidentifier

deid = Deidentifier.from_pretrained("stighellemans/meddeid-dutch-synth")
result = deid("Patiënt Alex Voorbeeld kwam op controle.")
print(result.deid_text)
deid.close()
```

Dates use placeholders unless the caller explicitly supplies
`metadata.date_shift_days`. A nonzero shift produces deterministic shifted
dates; zero produces placeholders and a structured warning. One declarative
age-granularity JSON policy is loaded for the complete engine, independent of
the selected language profile.

The self-contained model bundle is downloaded and cached on first use. Document
text is processed locally and is not sent to Hugging Face. Inspect the resolved
model, immutable revision, runtime, device, language profile, and package
versions with:

```bash
meddeid model-info
```

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
meddeid batch project/splits/test.jsonl --output predictions.jsonl
```

This normal path uses the default model, downloads it on first use, and chooses
the local device automatically. Add `--revision`, `--device`, or an alternate
`--model` only when the deployment requires those controls.

## HTTP service

PyTorch inference works on CPU, Apple MPS, and CUDA:

```bash
MEDDEID_DEVICE=cpu meddeid-server
```

Multi-profile services can set a locale fallback once while still allowing
trusted `metadata.lang` on a request to override it:

```bash
MEDDEID_MODEL=path/to/english-bundle \
MEDDEID_LANGUAGE_PROFILE=en-GB \
meddeid-server
```

The bundle declares its post-processing locales; users choose only the locale.
The browser UI shows a locale selector only when the loaded model
supports more than one profile.

For an authenticated service:

```bash
export MEDDEID_API_KEY='<random secret>'
export MEDDEID_REQUIRE_API_KEY=true
meddeid-server
```

The service provides:

- `POST /deidentify` for one document;
- `POST /deidentify-batch` for throughput-oriented batches; and
- `GET /health` for model identity and backend readiness.

For NVIDIA production serving, MedDeID can use a TensorRT engine hosted by
NVIDIA Triton while retaining the same tokenization, decoding, and Dutch
post-processing contract:

```bash
MEDDEID_BACKEND=triton \
MEDDEID_TRITON_URL=http://triton:8000 \
meddeid-server
```

See [Inference and deployment](docs/inference.md) for the complete Python,
JSONL, metadata, HTTP, Docker, TensorRT/Triton, sizing, and concurrency guide.
Operators should also read [Production deployment](docs/production.md).

## Language profile and metadata

The Dutch model bundle uses one set of model weights for both `nl-BE` and
`nl-NL`; `metadata.lang` selects the locale-specific language resources and
post-processing. Installed
bundles may pin the separate `meddeid-language-en` profiles `en-GB` or
`en-US`; bare `en` is rejected. Profile resolution uses installed
`meddeid.language_profiles` entry points, with no source-tree fallback. This
language-pack integration does not itself provide an English inference model.
For a combined GB/US bundle, document metadata is used first, followed by an
explicit load-time default and then a single-profile bundle default; MedDeID
never guesses between multiple profiles. See
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
