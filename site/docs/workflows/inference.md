# Local inference

<span class="source-label">Owner: meddeid</span>

Use `meddeid` to process individual notes, batches, or text inside a Python
application. If you have not installed MedDeID or run your first note yet,
start with [Install and run](../start/quickstart.md).

| Goal | Interface |
|---|---|
| Process one or a few files | Command-line interface |
| Process a JSONL dataset | Batch command |
| Add MedDeID to a Python application | Python API |
| Provide a shared service | [Production deployment](production-deployment.md) |

## Select a model and language

Review the available public models before running inference:

```bash
meddeid models
```

| Model | Supported profile |
|---|---|
| `stighellemans/meddeid-dutch-synth` | `nl-BE` |
| `stighellemans/meddeid-english-synth` | `en-GB`, `en-US` |

Model selection is explicit so the wrong language or model family cannot be
chosen silently. The public models are synthetic-data baselines, not
institution-validated clinical models. Validate the selected model on
representative data from your setting.

## Process one note from the command line

=== "Dutch (`nl-BE`)"

    ```bash
    meddeid deidentify note.txt \
      --model stighellemans/meddeid-dutch-synth
    ```

=== "English (`en-GB`)"

    ```bash
    meddeid deidentify note.txt \
      --model stighellemans/meddeid-english-synth \
      --language-profile en-GB  # Use en-US for US formats
    ```

Add `--json` when another tool needs the de-identified text, detected spans,
warnings, and provenance as structured output. The note is processed locally;
Hugging Face is used only to acquire the selected model when it is not already
cached.

## Process a batch of notes

```bash
meddeid batch project/splits/test.jsonl \
  --output predictions/test.jsonl \
  --model stighellemans/meddeid-dutch-synth
```

The batch command preserves document order and writes a manifest containing
the model identity, settings, and timing information needed to understand the
run. Its output can be reviewed with the annotation tools or evaluated with
`meddeid-eval`.

## Use the Python API

```python
from meddeid import Deidentifier

deidentifier = Deidentifier.from_pretrained(
    "stighellemans/meddeid-dutch-synth"
)
result = deidentifier(
    "Patiënt Alex Voorbeeld kwam op controle.",
    metadata={"patient": {
        "given_name": "Alex",
        "family_name": "Voorbeeld",
    }},
)

print(result.deid_text)
print(result.spans)
deidentifier.close()
```

Trusted information already known by the organization, such as a patient or
caregiver name, can help detect identifiers the model missed. It is applied
during local post-processing and is not added to the model input. Incorrect
metadata can cause unnecessary redaction, so validate it carefully.

## Make a run reproducible

Inspect the resolved model identity, profiles, and files:

```bash
meddeid model-info --model stighellemans/meddeid-dutch-synth
```

For a study or validated workflow, pin the immutable Hub revision reported by
`model-info` using `--revision`. By default, `model-info` inspects the model
without loading its weights; add `--verify-runtime` to confirm that the
configured backend and device can initialize.

MedDeID selects CUDA when available, followed by Apple MPS and then CPU.
Specify `--device cpu`, `--device mps`, or `--device cuda` only when the runtime
must be fixed explicitly. Because `model-info` can include local paths and
environment details, treat saved output as operationally sensitive.

On the measured M4 Pro, native MPS preserved semantics over the 300-document
public fixture and ran 1.6--2.0 times faster than native CPU across interactive,
batched ETL, and long-note workloads. Use one worker, eager FP32, and batch 16
as the initial ETL request size. The throughput profile enables the measured
bounded microbatcher; the latency profile stays queue-free. MPS runs through a
native Python installation because Linux Docker containers cannot use the host
Metal device. See [Production deployment](production-deployment.md#run-on-apple-silicon).

## Run a model from a local directory

MedDeID normally reuses models from the Hugging Face cache. Point `--model` at
a local directory when you want to stage and manage the model bundle yourself,
for example in an air-gapped environment:

```bash
hf download stighellemans/meddeid-dutch-synth \
  --revision <immutable-hub-sha> \
  --local-dir ./meddeid-dutch-synth

meddeid deidentify note.txt --model ./meddeid-dutch-synth
```

When `--model` points to an existing directory, MedDeID uses that bundle
directly without resolving it through the Hub. Transfer and validate the
complete directory rather than copying only the model weights: the tokenizer,
configuration, language resources, and other bundle files are also required.

## Need an HTTP service?

For local API development, install the optional server dependencies and start
the service with an explicit model:

```bash
python -m pip install 'meddeid[server]'
MEDDEID_MODEL=stighellemans/meddeid-dutch-synth meddeid-server
```

The service provides `POST /deidentify`, `POST /deidentify-batch`, and
`GET /health`. To inspect the browser interface and HTTP API in a local
container, use the [local Docker
setup](../start/quickstart.md#try-meddeid-in-your-browser).

Before exposing a service to a network or using it with clinical text, follow
[Production deployment](production-deployment.md) for authentication, TLS,
network isolation, request limits, monitoring, and operational validation.

## Next steps

- [Understand the data and result contract](../concepts/data-contract.md)
- [Deploy a shared institutional service](production-deployment.md)
- [Check released versions and compatibility](../reference/compatibility.md)
- [Open the complete component reference](https://github.com/stighellemans/meddeid/blob/main/docs/inference.md)
