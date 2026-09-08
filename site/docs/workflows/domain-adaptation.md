# Adapt a model with local notes

Use this workflow when an existing model performs poorly on notes from your
hospital, speciality, or document type. You will train a local version and
check whether it is better than the original model.

## First, check that adaptation is worthwhile

Evaluate the existing model on representative local notes. Continue only when
you find a meaningful performance gap and have enough reviewed notes to create
separate training and test sets.

## Use different data to train and test

Split the local documents before annotation, model suggestions, or training:

| Data | Use it to |
|---|---|
| Development data | Train the local model and choose its settings |
| Test data | Compare the original and local models once |

Do not use test answers or test results to change the training. Otherwise, the
test no longer shows how the model performs on unseen notes.

## Follow five steps

### 1. Write down the plan

Record the setting, document types, data split, original model version,
training choices, and measures of success before comparing results.

### 2. Prepare and review the notes

Use [Prepare and annotate data](prepare-and-annotate.md) to create the split
and review both sets. Development annotations can guide training. Keep test
annotations hidden from the training process.

### 3. Measure the original model

Use [Evaluate predictions](train-and-evaluate.md#5-evaluate-predictions) to run
the unchanged model on the test documents and score its predictions. This is
the **baseline**: the result that the local model must improve on.

### 4. Train the local model

Use [Train and evaluate](train-and-evaluate.md) to train with development data
only. Start from the same model version that you used for the baseline.

### 5. Compare the models

Use [Evaluate predictions](train-and-evaluate.md#5-evaluate-predictions) to run
the local model on the same test documents. Score both models with the same
reviewed answers, metrics, and `meddeid-eval` version.

Check both sides of the result:

- Did the local model miss fewer identifiers?
- Did it remove less useful clinical text?

## What to keep

Keep the data split, reviewed-data records, training settings, model version,
predictions, and scores together. See
[Artifact lineage](../concepts/artifact-lineage.md) for details.
