from __future__ import annotations

from types import SimpleNamespace

import torch

from meddeid.model import DualHeadTokenClassifier


def test_dual_head_token_classifier_passes_attn_implementation(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    config = SimpleNamespace(hidden_size=16, hidden_dropout_prob=0.2)

    def fake_config_from_pretrained(model_name: str, **kwargs):
        recorded["config_model_name"] = model_name
        recorded["config_kwargs"] = kwargs
        return config

    def fake_model_from_pretrained(model_name: str, **kwargs):
        recorded["encoder_model_name"] = model_name
        recorded["encoder_kwargs"] = kwargs
        return torch.nn.Identity()

    monkeypatch.setattr(
        "meddeid.model.AutoConfig.from_pretrained",
        fake_config_from_pretrained,
    )
    monkeypatch.setattr(
        "meddeid.model.AutoModel.from_pretrained",
        fake_model_from_pretrained,
    )

    model = DualHeadTokenClassifier(
        model_name="example/model",
        num_bio_labels=3,
        num_entity_labels=7,
        attn_implementation="eager",
    )

    assert isinstance(model.encoder, torch.nn.Identity)
    assert recorded["config_model_name"] == "example/model"
    assert recorded["config_kwargs"] == {"local_files_only": False}
    assert recorded["encoder_model_name"] == "example/model"
    assert recorded["encoder_kwargs"] == {
        "config": config,
        "attn_implementation": "eager",
    }


def test_dual_head_token_classifier_can_initialize_from_local_config(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    config = SimpleNamespace(hidden_size=16, hidden_dropout_prob=0.2)

    monkeypatch.setattr(
        "meddeid.model.AutoConfig.from_pretrained",
        lambda model_name, **kwargs: config,
    )

    def fake_model_from_config(received_config, **kwargs):
        recorded["config"] = received_config
        recorded["kwargs"] = kwargs
        return torch.nn.Identity()

    monkeypatch.setattr("meddeid.model.AutoModel.from_config", fake_model_from_config)
    model = DualHeadTokenClassifier(
        model_name="/local/model",
        num_bio_labels=3,
        num_entity_labels=7,
        initialize_from_pretrained=False,
        local_files_only=True,
    )
    assert isinstance(model.encoder, torch.nn.Identity)
    assert recorded == {"config": config, "kwargs": {}}
