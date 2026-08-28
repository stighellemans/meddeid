# Train and evaluate

Use reviewed development data to train a model. Keep a separate test set untouched until the final evaluation.

Install the released workflow packages from PyPI:

```bash
python -m pip install \
  meddeid-data \
  'meddeid-eval[plots]' \
  'meddeid-training[train]'
```

## Prepare data and protect the test set

<span class="source-label">Owner: meddeid-data</span>

```bash
meddeid-data project prepare-training my-project \
  --development assignments/development-reviewer-a.jsonl \
  --test-gold evaluation/meddeid-dutch-synthetic-benchmark.jsonl
```

This checks that the reviewed files belong to the project and prepares three folders for training and evaluation:

| View | Purpose |
|---|---|
| `prepared/fit` | Run one ordinary training experiment |
| `prepared/selection` | Decide how long to train without looking at test answers |
| `prepared/refit` | Train on all development data and evaluate once on the separate test set |

## Ordinary training run

<span class="source-label">Owner: meddeid-training</span>

```bash
meddeid-train fit \
  --config configs/release.yaml \
  --data prepared/fit \
  --run runs/fit
```

Validation chooses the best saved model. Test evaluation happens after training. The result is `runs/fit/checkpoints/best.pt`.

## Publication protocol

For a release-quality experiment, first use validation data to choose how long to train. Then start fresh, train on all development data, and evaluate once on the separate test set:

```bash
meddeid-train select-epochs \
  --config configs/release.yaml \
  --data prepared/selection \
  --run runs/selection

meddeid-train refit \
  --config configs/release.yaml \
  --selection runs/selection/run.json \
  --data prepared/refit \
  --run runs/refit

meddeid-train export \
  --checkpoint runs/refit/checkpoints/best.pt \
  --run-metadata runs/refit/train_metrics.json \
  --output release/my-model
```

Both runs start independently from the configured initial model. The final run does not continue from the earlier one. The exported directory contains everything `meddeid` needs to use the model.

## Evaluate predictions

<span class="source-label">Owner: meddeid-eval</span>

Generate predictions with the exact exported bundle:

```bash
meddeid batch prepared/refit/test.jsonl \
  --model release/my-model \
  --output predictions/test.jsonl
```

Then score the prediction file:

```bash
meddeid-eval score \
  --gold prepared/refit/test.jsonl \
  --predictions predictions/test.jsonl \
  --name my-model \
  --seconds 18.4 \
  --device gpu \
  --output results/my-model.json
```

The installation above includes plotting support. Add the `infer` extra when
evaluation itself must run model inference.

The score artifact reports exact-span and character metrics, core-PII recall,
non-PII redaction, and privacy-safe aggregate tables by gold label,
sub-annotation category, and predicted label when the test data supports them.

To compare another system, run it separately, convert its results to the MedDeID prediction format, and score them with `meddeid-eval`.

Render the resulting score artifacts together:

```bash
meddeid-eval plot \
  --scores results/my-model.json results/comparator.json \
  --output-dir results/plots
```

PNG and searchable vector PDF figures are written by default. The figure family
includes a performance overview, label and sub-annotation heatmaps, non-PII
redactions by predicted label, exact-boundary label confusion, and an
accuracy-versus-runtime plot when timing metadata was recorded.

## Keep with every result

- the dataset versions and development/test split;
- the starting model and exact version;
- the language profile and version;
- the annotation labels used;
- settings and random seeds;
- package versions and hardware/runtime information;
- the saved model version;
- exact prediction and metric commands.

MedDeID records much of this automatically. Keep those run records with the paper or report. See [artifact lineage](../concepts/artifact-lineage.md) for the detailed reproducibility fields.
