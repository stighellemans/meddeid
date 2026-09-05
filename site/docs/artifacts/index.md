# Public artifacts

The [MedDeID Hugging Face collection](https://huggingface.co/collections/stighellemans/meddeid)
contains the public Dutch and English models, development data, and independent
benchmarks.

## Choose what you need

| Goal | Artifact |
|---|---|
| Try MedDeID without installing it | [Non-clinical demo](https://huggingface.co/spaces/stighellemans/meddeid-demo) |
| De-identify Dutch clinical text | [Dutch synthetic model](https://huggingface.co/stighellemans/meddeid-dutch-synth) |
| De-identify English clinical text | [English synthetic model](https://huggingface.co/stighellemans/meddeid-english-synth) |
| Develop or adapt a model | A synthetic corpus for the applicable language |
| Evaluate a system independently | The applicable synthetic benchmark |
| Prepare human annotations | The Dutch or English annotation guideline |

The published models are synthetic baselines. Before using one with clinical
text, validate it on representative data from your own setting.

## Run a published model

### [`stighellemans/meddeid-dutch-synth`](https://huggingface.co/stighellemans/meddeid-dutch-synth)

Choose this model for Dutch text. MedDeID downloads it on first use and then
reuses the local Hugging Face cache:

```bash
pip install meddeid
meddeid deidentify note.txt \
  --model stighellemans/meddeid-dutch-synth
```

### [`stighellemans/meddeid-english-synth`](https://huggingface.co/stighellemans/meddeid-english-synth)

Choose this model for English text and select the regional profile that matches
the document formats:

```bash
meddeid deidentify note.txt \
  --model stighellemans/meddeid-english-synth \
  --language-profile en-GB  # Use en-US for US formats
```

For a reproducible workflow, record the model's immutable Hub revision. To run
from a local copy, keep the complete model directory together.

## Develop or evaluate with published data

Use a corpus for model development, including training and validation. Keep the
benchmark separate until final evaluation so it remains an independent test.

| Language | Development corpus | Independent benchmark |
|---|---|---|
| Dutch | [`meddeid-dutch-synthetic-corpus`](https://huggingface.co/datasets/stighellemans/meddeid-dutch-synthetic-corpus) | [`meddeid-dutch-synthetic-benchmark`](https://huggingface.co/datasets/stighellemans/meddeid-dutch-synthetic-benchmark) |
| English | [`meddeid-english-synthetic-corpus`](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-corpus) | [`meddeid-english-synthetic-benchmark`](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark) |

Before downloading or citing an artifact, read its card for the exact contents,
provenance, evaluation results, limitations, licence, and available revisions.
Pin the immutable revision when the result must be reproducible.

## Annotation guidelines

The current `ProductionLabels_v1` guidelines are available as PDF reading
copies and editable DOCX sources in the Zenodo releases for
[Dutch](https://doi.org/10.5281/zenodo.21992866) and
[English](https://doi.org/10.5281/zenodo.22129255). They explain what
annotators should mark; the precise software rules are documented separately
in [`meddeid-core`](https://github.com/stighellemans/meddeid-core).

## Public demo

The [MedDeID hosted demo](https://huggingface.co/spaces/stighellemans/meddeid-demo)
runs the public synthetic model for non-sensitive examples. It executes on
Hugging Face infrastructure: never paste real patient or caregiver information.
Use `meddeid` locally for clinical text.
