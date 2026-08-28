# Workflow overview

MedDeID consists of focused tools that work together. You can use the complete workflow or choose only the steps your project needs.

## Clinical inference

```text
note.txt or documents.jsonl
  → meddeid
  → de-identified text or predictions.jsonl
```

Use this path when a trained model already meets the needs of the target setting. [Run local inference](inference.md).

## Dataset preparation and review

```text
hospital TXT / CSV / TSV / Parquet
  → meddeid-data project
  → optional model pre-annotations
  → meddeid-annotate
  → completed reviewed annotations
```

One completed review is sufficient for training. Add another reviewer and `meddeid-curate` only when your study requires independent review. [Prepare and annotate data](prepare-and-annotate.md).

## Detailed evaluation

```text
completed reviewed annotations
  → optional meddeid-curate
  → meddeid-subannotate
  → detailed evaluation benchmark
  → meddeid-eval
```

This optional step marks which parts of each identifier are essential to be detected. It supports detailed evaluation and is not required for training.

## Model adaptation

```text
reviewed development data + separate test data
  → meddeid-data prepare-training
  → meddeid-training
  → exported model bundle
  → meddeid batch
  → meddeid-eval
```

For published research, MedDeID can first determine how long to train and then train a fresh model on all development data. A simpler one-time training run is available for ordinary experiments. [Train and evaluate](train-and-evaluate.md).

## What stays consistent

Across the workflow:

- one tool's output can be used directly by the next;
- source files are not silently overwritten;
- MedDeID records which data, model, and settings produced a result;
- the independent test set stays separate while training decisions are made.

Implementation details are available in the [data contract](../concepts/data-contract.md) and [artifact lineage](../concepts/artifact-lineage.md) pages.
