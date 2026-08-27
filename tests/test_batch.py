import json
from types import SimpleNamespace

import pytest

from meddeid.batch import run_batch


class FakeEngine:
    bundle = SimpleNamespace(
        name="test-model",
        model_version="1",
        postprocess=SimpleNamespace(profile_id="nl-BE"),
        contract_hash=lambda: "a" * 64,
    )

    def __init__(self):
        self.calls = []

    def __call__(self, text, *, metadata):
        self.calls.append(text)
        return SimpleNamespace(
            spans=[],
            deid_text=text,
            language_profile=metadata.get("lang", "nl-BE"),
        )

    def model_info(self):
        return {
            "model": {
                "source": "test/model",
                "requested_revision": "main",
                "resolved_revision": "abc123",
            },
            "runtime": {"backend": "torch", "device": "cpu", "ready": True},
            "environment": {"python": "test"},
        }


def test_batch_preserves_ids_order_and_writes_manifest(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        '{"document_id":"b","text":"😀 note","spans":[],"metadata":{"lang":"nl-BE"}}\n'
        '{"document_id":"a","text":"other","spans":[]}\n',
        encoding="utf-8",
    )
    output = tmp_path / "predictions.jsonl"
    engine = FakeEngine()
    manifest = run_batch(engine, source, output)
    assert engine.calls == ["😀 note", "other"]
    assert [line.split('"')[3] for line in output.read_text(encoding="utf-8").splitlines()] == ["b", "a"]
    assert manifest["contracts"]["offset_unit"] == "unicode_codepoints"
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["text"] == "😀 note"
    assert rows[0]["warnings"] == []
    assert rows[0]["processing"] == {}
    assert manifest["counts"]["language_profiles"] == [
        {"profile_id": "nl-BE", "documents": 2},
    ]
    assert output.with_suffix(".jsonl.manifest.json").is_file()


def test_batch_resume_reuses_finished_documents(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"document_id":"a","text":"one","spans":[]}\n', encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    first = FakeEngine()
    run_batch(first, source, output)
    second = FakeEngine()
    run_batch(second, source, output, resume=True)
    assert second.calls == []
    with pytest.raises(FileExistsError):
        run_batch(second, source, output)


def test_batch_reports_progress_timing_and_runtime(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text('{"document_id":"a","text":"one","spans":[]}\n', encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    events = []
    manifest = run_batch(FakeEngine(), source, output, progress=events.append)
    assert events[0]["completed"] == 1
    assert events[0]["total"] == 1
    assert manifest["runtime"]["backend"] == "torch"
    assert manifest["model"]["resolved_revision"] == "abc123"
    assert manifest["counts"]["processed"] == 1
    assert manifest["counts"]["failed"] == 0
    assert manifest["timing"]["elapsed_seconds"] >= 0
