# Install and run

Choose the path that matches how you intend to use MedDeID:

| Goal | Start here |
|---|---|
| De-identify a note or a batch of files | [Run MedDeID locally](#run-meddeid-locally) |
| Add de-identification to a Python application | [Use the Python API](#use-the-python-api) |
| Call a gateway or reuse detections on CPU with 0.4.3 | [Install the release packages](../reference/compatibility.md#meddeid-043) and [render a note](../workflows/rendering.md) |
| Explore the browser interface or HTTP API | [Try MedDeID in your browser](#try-meddeid-in-your-browser) |
| Operate a shared institutional service | [Choose a runtime](../workflows/production-deployment.md) or [install the hospital service and connect Databricks](../workflows/hospital/index.md) |

Training, evaluation, and annotation tools are separate workflows linked at
the end of this page.

## Run MedDeID locally

This is the shortest path for individual notes, scripts, and batches. It
requires Python 3.10 or newer but does not require writing Python code.

### Install

Install MedDeID 0.4.3 from PyPI. See the [component versions](../reference/compatibility.md#meddeid-043) for a pinned deployment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'meddeid==0.4.3'
meddeid models
```

`meddeid models` lists the available models and their validation scope. Select
one explicitly with `--model`; it is downloaded once and used locally.

### Choose Dutch or English

=== "Dutch (`nl-BE`)"

    ```bash
    meddeid deidentify --text "Patiënt Alex Peters kwam op controle." \
      --model stighellemans/meddeid-dutch-synth
    ```

=== "English (`en-GB`)"

    ```bash
    meddeid deidentify --text "Alex Peters returned for a check-up." \
      --model stighellemans/meddeid-english-synth \
      --language-profile en-GB  # Use en-US for US formats
    ```

For a real note, pass its file path in place of `--text` and the example
string so the note stays out of shell history. The public Dutch model declares
`nl-BE` and the English model supports `en-GB` and `en-US`.

### Use the Python API

```python
from meddeid import Deidentifier

deidentifier = Deidentifier.from_pretrained(
    "stighellemans/meddeid-dutch-synth"
)
result = deidentifier("Patiënt Alex Voorbeeld kwam op controle.")

print(result.deid_text)
print(result.spans)
deidentifier.close()
```

### Process a batch of notes

```bash
meddeid batch documents.jsonl \
  --output predictions.jsonl \
  --model stighellemans/meddeid-dutch-synth
```

The batch command keeps document order and records how the results were
produced.

## Try MedDeID in your browser

Install and start Docker Desktop. For the published image, use its matching
Compose files.

```bash
git clone --branch v0.4.3 https://github.com/stighellemans/meddeid.git
cd meddeid
```

Choose the language you want to test:

=== "Dutch"

    ```bash
    ./scripts/start-local.sh \
      --model stighellemans/meddeid-dutch-synth \
      --language-profile nl-BE
    ```

=== "English"

    ```bash
    ./scripts/start-local.sh \
      --model stighellemans/meddeid-english-synth \
      --language-profile en-GB
    ```

When MedDeID is ready, the browser opens with the API key already filled in.
Paste a note and select **De-identify**. The English model also lets you switch
between `en-GB` and `en-US` in the browser.

API documentation is available at `http://127.0.0.1:8000/docs`.

Stop the service with `./scripts/stop-local.sh`.

For a shared service, see [production deployment](../workflows/production-deployment.md).
For a hospital deployment, see
[prepare a hospital release](../workflows/hospital/deploy-service.md#choose-the-release).

## Reproducible or offline runs

Inspect and record the exact model version when results must be reproducible:

```bash
meddeid model-info --model stighellemans/meddeid-dutch-synth
```

To stage and manage a model bundle yourself, pass its local directory with
`--model`. The [local inference
guide](../workflows/inference.md#run-a-model-from-a-local-directory) covers
revision pinning, complete local bundles, and air-gapped environments.

## Next steps

- [Explore local inference options](../workflows/inference.md)
- [Deploy a shared institutional service](../workflows/production-deployment.md)
- [Review privacy and security boundaries](../project/privacy-and-security.md)
- [Choose a workflow for data preparation, evaluation, or training](../workflows/index.md)
