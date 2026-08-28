# Install and run

## Requirements

- Docker Desktop or Docker Engine with Compose
- **CPU:** 8 GiB RAM recommended
- About 4 GB free disk space and internet access for the first image download

These are practical starting points; production needs depend on workload.

## Easiest start with Docker

Clone the `meddeid` repository and run:

```bash
git clone https://github.com/stighellemans/meddeid.git
cd meddeid
./scripts/start-local.sh
```

The script generates a private API key, pulls the published CPU image with the
pinned default Dutch model, starts it only on your computer, and waits until it is ready. Open
`http://127.0.0.1:8000/ui`, paste the API key from the generated `.env` file,
then paste a note. No Python or API command is required. Technical API
documentation remains available at `http://127.0.0.1:8000/docs`.

Stop the service with `./scripts/stop-local.sh`.

## Python option

Install the released Python API, CLI, batch runner, and optional HTTP service:

```bash
python -m pip install 'meddeid[server]'
meddeid deidentify note.txt
```

MedDeID downloads and caches the default
`stighellemans/meddeid-dutch-synth` model on first use, chooses a local device,
and processes the note locally.

## Choose Dutch or English

The default command uses the public Dutch model. For English, select the
English model and an explicit regional profile:

```bash
meddeid deidentify note.txt \
  --model stighellemans/meddeid-english-synth \
  --language-profile en-GB
```

Use `en-US` for US formats. The Dutch model supports `nl-BE` and `nl-NL`;
the English model supports `en-GB` and `en-US`. MedDeID never guesses between
regional profiles.

See [local inference](../workflows/inference.md#availability-today) for current
deployment boundaries.

## Use the Python API

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

## Process a batch of notes

```bash
meddeid batch documents.jsonl --output predictions.jsonl
```

The batch command keeps the document order and saves the information needed to identify how the results were produced.

## Check the model version

```bash
meddeid model-info
```

Save the exact model version shown by this command when the result must be reproducible.

!!! tip "Air-gapped environments"
    Download a fixed copy of the model outside the secure environment, validate it, transfer it according to local policy, and pass the local directory with `--model`. See [local inference](../workflows/inference.md#offline-and-air-gapped-use).

## Next steps

- [Run batch or service inference](../workflows/inference.md)
- [Learn how MedDeID structures data](../concepts/data-contract.md)
- [Review privacy and security boundaries](../project/privacy-and-security.md)
