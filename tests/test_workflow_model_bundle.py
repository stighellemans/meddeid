from __future__ import annotations

import json

import pytest
import torch

from meddeid.workflow import WorkflowError, _validate_model_checkpoint
from meddeid_core import BERT_ENTITY_LABELS


def _training_run(tmp_path, *, profiles=("en-GB", "en-US")):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "epoch": 4,
            "model_state_dict": {
                "label_classifier.weight": torch.zeros(len(BERT_ENTITY_LABELS), 3),
                "bio_classifier.weight": torch.zeros(3, 3),
            },
        },
        checkpoint,
    )
    metrics = {
        "entity_labels": list(BERT_ENTITY_LABELS),
        "protocol": {
            "contract": "meddeid.selection-refit-benchmark.v1",
            "benchmark_evaluations": 1,
        },
        "config": {
            "language_profiles": [{"profile_id": profile} for profile in profiles],
            "base_encoder": "FacebookAI/roberta-base",
            "base_revision": "immutable-revision",
        },
    }
    (run / "train_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run


def test_checkpoint_validation_reads_actual_lineage(tmp_path) -> None:
    run = _training_run(tmp_path)
    workspace = tmp_path / "workspace"
    _validate_model_checkpoint(
        workspace,
        {
            "checkpoint": str(run),
            "profiles": ["en-GB", "en-US"],
            "base_encoder": "FacebookAI/roberta-base",
            "base_revision": "immutable-revision",
        },
    )
    report = json.loads(
        (workspace / "artifacts" / "checkpoint-validation.json").read_text()
    )
    assert report["passed"] is True
    assert report["profiles"] == ["en-GB", "en-US"]
    assert report["entity_labels"] == list(BERT_ENTITY_LABELS)


def test_checkpoint_validation_rejects_decision_metadata_conflicts(tmp_path) -> None:
    run = _training_run(tmp_path, profiles=("en-GB",))
    with pytest.raises(WorkflowError, match="profiles"):
        _validate_model_checkpoint(
            tmp_path / "workspace",
            {
                "checkpoint": str(run),
                "profiles": ["en-GB", "en-US"],
                "base_encoder": "FacebookAI/roberta-base",
                "base_revision": "immutable-revision",
            },
        )
