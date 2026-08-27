from __future__ import annotations

import json
from types import SimpleNamespace

from meddeid import cli


class FakeEngine:
    def model_info(self) -> dict:
        return {}

    def close(self) -> None:
        return None

    def __call__(self, text, *, metadata=None):
        return SimpleNamespace(
            text=text,
            deid_text=text,
            spans=[],
            language_profile="en-GB",
        )


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
    assert captured["language_profile"] is None
    assert captured["age_granularity_config"] is None
    assert captured["min_recommended_date_shift_days"] == 366


def test_cli_json_reports_bundle_pinned_profile(monkeypatch, tmp_path, capsys) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Example", encoding="utf-8")
    monkeypatch.setattr(cli, "_load_engine", lambda _args: FakeEngine())

    assert cli.main(["deidentify", str(source), "--json", "--quiet"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "text" not in payload
    assert payload["language_profile"] == {"profile_id": "en-GB"}
