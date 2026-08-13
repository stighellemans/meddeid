from __future__ import annotations

from meddeid import cli


class FakeEngine:
    def model_info(self) -> dict:
        return {}

    def close(self) -> None:
        return None


def test_cli_uses_downloadable_default_without_extra_flags(monkeypatch) -> None:
    captured = {}

    def fake_load_engine(args):
        captured.update(vars(args))
        return FakeEngine()

    monkeypatch.setattr(cli, "_load_engine", fake_load_engine)

    assert cli.main(["model-info", "--quiet"]) == 0
    assert captured["model"] == "stighellemans/meddeid-dutch-synth"
    assert captured["revision"] is None
    assert captured["offline"] is False
    assert captured["device"] is None
