# Train and evaluate a model

Use this workflow when you have reviewed development data and a separate,
reviewed test set. Development data may influence training decisions; test
answers must remain unseen until the final evaluation.

You can follow either of two training routes:

| Goal | Route |
|---|---|
| Explore whether training works | One ordinary fit |
| Produce a study or release result | Select the training duration, then restart and refit on all development data |

If you only want to evaluate existing predictions, skip directly to
[Evaluate predictions](#5-evaluate-predictions).

## Before you begin

You need:

- a MedDeID project that was split before review;
- completed train, validation, and test assignments whose document IDs still
  match those project splits; and
- a training configuration that identifies the starting model and language
  profile.

See [Prepare and annotate data](prepare-and-annotate.md) if those reviewed
files do not exist yet.

Install the released tools:

```bash
python -m pip install \
  meddeid-data \
  'meddeid-training[train]' \
  'meddeid-eval[plots]'
```

The training extra also installs `meddeid`, which is used later to generate
predictions from the exported model. If you only need to score and plot
predictions that already exist, install `meddeid-eval[plots]` instead.

## 1. Prepare the training files

<span class="source-label">Owner: meddeid-data</span>

Pass the three completed assignments to `prepare-training`:

```bash
meddeid-data project prepare-training my-project \
  --selection-train my-project/assignments/train-reviewed.jsonl \
  --selection-validation my-project/assignments/validation-reviewed.jsonl \
  --test-gold my-project/assignments/test-reviewed.jsonl
```

The command verifies completion, document membership, text, labels, and
checksums before creating three views under `my-project/prepared/`:

| Directory | What it contains | Use it for |
|---|---|---|
| `fit` | Separate train, validation, and test files | One ordinary experiment |
| `selection` | Train and validation files; the test file is empty | Choosing the training duration without seeing test answers |
| `refit` | All development documents for training and the sealed test set | Restarting, fitting all development data, and evaluating once |

Use `--development` instead of the two `--selection-*` options only when train
and validation were deliberately reviewed as one combined development
assignment.

The command does not modify the reviewed source files and will not overwrite a
non-empty `prepared/` directory.

## 2. Create the training configuration

<span class="source-label">Owner: meddeid-training</span>

Save a small YAML file such as `training.yaml`:

```yaml
model_name: stighellemans/meddeid-dutch-synth
model_revision: <immutable-hub-revision>
language_profile: nl-BE
device: auto
epochs: 8
seed: 42
```

`model_name` is the model from which training starts. Use its immutable Hub
revision so separate runs cannot resolve different model versions. You can
inspect the model and copy its reported revision with:

```bash
meddeid model-info \
  --model stighellemans/meddeid-dutch-synth
```

For English data, use the English starting model and `en-GB` or `en-US`. When
one training dataset deliberately contains both profiles, replace
`language_profile` with:

```yaml
language_profiles:
  - en-GB
  - en-US
```

The remaining training parameters have defaults. Add or change them only when
they are part of the experiment you intend to run. The
[`meddeid-training` repository](https://github.com/stighellemans/meddeid-training)
documents the complete configuration.

## 3. Run one ordinary fit

Use this route for exploration or an ordinary train/validation/test
experiment:

```bash
meddeid-train fit \
  --config training.yaml \
  --data my-project/prepared/fit \
  --run runs/fit
```

Training uses the validation set to retain the best checkpoint, then evaluates
that checkpoint on the test set. The selected checkpoint is written to
`runs/fit/checkpoints/best.pt`.

Do not use repeated ordinary fits on the same test set to select
hyperparameters. Once test results influence another training decision, the
test set is no longer independent.

## 4. Select and refit for a study or release

Use this two-stage route when the final test result must remain independent of
training-duration selection.

First, select the number of epochs using only the development train and
validation data:

```bash
meddeid-train select-epochs \
  --config training.yaml \
  --data my-project/prepared/selection \
  --run runs/selection
```

This writes the selected epoch count to `runs/selection/run.json`. The test
file in this view is empty, so this stage cannot score the test set.

Then restart from the original model and train on all development data for that
fixed number of epochs:

```bash
meddeid-train refit \
  --config training.yaml \
  --selection runs/selection/run.json \
  --data my-project/prepared/refit \
  --run runs/refit
```

Refit does not continue from the selection checkpoint. It starts again from the
same model revision, combines the development train and validation data, and
evaluates once on the sealed test set.

## Export the trained model

Export the checkpoint from the route you chose. This example uses the refit
run; replace `runs/refit` with `runs/fit` after an ordinary fit:

```bash
meddeid-train export \
  --checkpoint runs/refit/checkpoints/best.pt \
  --run-metadata runs/refit/train_metrics.json \
  --output release/my-model
```

The `release/my-model` directory is a self-contained MedDeID model bundle. Use
that exported directory—not an in-memory model or an unexported checkpoint—for
the final inference and evaluation.

## 5. Evaluate predictions

<span class="source-label">Owner: meddeid-eval</span>

Generate predictions from the exact exported bundle. The example uses the
refit test view; use `prepared/fit/test.jsonl` for an ordinary fit:

```bash
meddeid batch my-project/prepared/refit/test.jsonl \
  --model release/my-model \
  --output predictions/test.jsonl
```

Score those predictions against the unchanged test gold:

```bash
meddeid-eval score \
  --gold my-project/prepared/refit/test.jsonl \
  --predictions predictions/test.jsonl \
  --name my-model \
  --output results/my-model.json
```

The result includes exact-span and character-level metrics, core-PII recall,
and unnecessary redaction outside the reviewed identifiers. Detailed
subannotation metrics appear only when the test gold contains those optional
labels.

Add `--seconds` and `--device` only when you measured runtime and want that
context stored with the score. They do not run a benchmark themselves.

## Compare systems and create figures

To compare another system, run it in its own environment and convert its
predictions to the MedDeID result format. Score it against the same gold file
with the same `meddeid-eval` version.

Render one or more score files together:

```bash
meddeid-eval plot \
  --scores results/my-model.json results/comparator.json \
  --output-dir results/plots
```

The command writes PNG and searchable PDF figures. Consult the
[`meddeid-eval` repository](https://github.com/stighellemans/meddeid-eval)
for metric definitions, stability analysis, and additional plot options.

## Keep with the result

Keep enough information to identify the complete run:

- the reviewed development and test manifests;
- the fixed project split;
- `training.yaml` and its random seed;
- the starting model and immutable revision;
- the selected language profile and package versions;
- the training run metadata and exported bundle;
- the prediction manifest, score file, and exact commands; and
- relevant hardware and measured runtime information.

See [Artifact lineage](../concepts/artifact-lineage.md) for how these files
connect.
