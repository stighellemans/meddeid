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

The Dutch model bundle pins `meddeid-language-nl` profile `nl-BE@1`. Optional
trusted metadata can recover known patient or caregiver names and other known
values after neural inference. Metadata is not concatenated to the note or sent
to the model as an input feature. Incorrect metadata can create false-positive
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
