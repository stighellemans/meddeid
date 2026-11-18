from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from threading import Lock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from meddeid import cli
from meddeid.api import Deidentifier, redact
from meddeid.server import create_app
from meddeid_language_nl import get_profile
from meddeid_language_en import get_profile as english_profile

KEY = bytes(range(32))
PROVENANCE = {
    "contract_version": "meddeid.inference-provenance.v1",
    "software": {"name": "meddeid", "version": "test"},
    "model": {
        "name": "test",
        "version": "1",
        "resolved_revision": "abc123",
        "bundle_sha256": "a" * 64,
    },
    "language_profile": {"profile_id": "nl-BE"},
}


def engine(scope="study-a", configured=True):
    """Real public pipeline/renderer with synthetic spans instead of model inference."""
    instance = object.__new__(Deidentifier)
    instance._name_settings = (KEY, scope) if configured else None
    instance._inference_lock = Lock()
    instance.language_profiles = tuple(
        get_profile(p) for p in ("nl-BE", "nl-NL")
    ) + tuple(english_profile(p) for p in ("en-GB", "en-US"))
    instance.language_profile = instance.language_profiles[0]
    instance.inference_provenance = lambda profile_id: {
        **PROVENANCE,
        "language_profile": {"profile_id": profile_id},
    }
    instance.pipeline = SimpleNamespace(
        prepare_document=lambda **kw: SimpleNamespace(**kw),
        prepare_state=lambda index, prepared: SimpleNamespace(windows=[prepared]),
        apply_predictions=lambda state, predictions: None,
        decode_document=lambda state: SimpleNamespace(
            spans=[span(state.windows[0].text)]
        ),
    )
    instance.runtime = SimpleNamespace(
        infer_windows=lambda windows: [object() for _ in windows], close=lambda: None
    )
    instance.model_info = lambda: {
        "model": {"name": "test", "version": "1"},
        "contracts": {
            "language_profiles": [
                {"profile_id": p.profile_id} for p in instance.language_profiles
            ],
            "default_language_profile": "nl-BE",
        },
        "runtime": {"ready": True, "backend": "torch", "device": "cpu"},
        "environment": {},
    }
    return instance


def span(text, label="Name:Patient"):
    category, _, subtype = label.partition(":")
    return {
        "begin": 0,
        "end": len(text),
        "text": text,
        "label": label,
        "category": category,
        "subtype": subtype or None,
    }


@pytest.mark.parametrize("profile_id", ["nl-BE", "nl-NL", "en-GB", "en-US"])
@pytest.mark.parametrize("label", ["Name:Patient", "Name:Caregiver", "Name:Other"])
def test_boolean_applies_to_all_name_subtypes_and_locales(profile_id, label):
    instance = engine()
    profile = next(p for p in instance.language_profiles if p.profile_id == profile_id)
    for flag in (False, True):
        spans, warnings, processing = instance._apply_replacements(
            "Mark Peters",
            [span("Mark Peters", label)],
            {"name_replacement": flag},
            profile,
        )
        expected = (
            f"[{profile.replace_name('Mark Peters', key=KEY, scope='study-a')}]"
            if flag
            else f"[{label}]"
        )
        assert redact("Mark Peters", spans) == expected
        assert warnings == []
        assert processing["name_replacement"]["replaced_spans"] == int(flag)
        assert processing["name_replacement"]["placeholder_spans"] == int(not flag)
        assert KEY.hex() not in json.dumps(processing)
        assert "study-a" not in json.dumps(processing)


def test_python_batch_threads_and_retry_have_the_same_result():
    instance = engine()
    metadata = {"name_replacement": True}
    expected = "[Sue-Ann Colen]"
    assert instance("Mark Peters", metadata=metadata).deid_text == expected
    assert [
        r.deid_text for r in instance.deidentify_many([("Mark Peters", metadata)] * 4)
    ] == [expected] * 4
    with ThreadPoolExecutor(4) as workers:
        assert (
            list(
                workers.map(
                    lambda _: instance("Mark Peters", metadata=metadata).deid_text,
                    range(10),
                )
            )
            == [expected] * 10
        )
    assert (
        engine(scope="other-study")("Mark Peters", metadata=metadata).deid_text
        != expected
    )
    assert instance("Mark Peters").deid_text == "[Name:Patient]"


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false"])
def test_invalid_boolean_is_rejected_before_inference(value):
    instance = engine()
    instance.runtime.infer_windows = lambda _: pytest.fail("inference must not run")
    with pytest.raises(ValueError, match="boolean"):
        instance("Mark Peters", metadata={"name_replacement": value})


def test_missing_configuration_and_request_secrets_are_rejected_before_inference():
    instance = engine(configured=False)
    instance.runtime.infer_windows = lambda _: pytest.fail("inference must not run")
    with pytest.raises(ValueError, match="configured name key"):
        instance("Mark Peters", metadata={"name_replacement": True})
    for key in ("name_key", "name_scope"):
        with pytest.raises(ValueError, match="deployment-only"):
            instance("Mark Peters", metadata={key: "secret"})


def test_unsupported_profile_or_name_preserves_original_label():
    instance = engine()
    profile = get_profile("nl-BE")
    for text, selected, code in (
        ("<unknown>", profile, "name_parse_fallback"),
        (
            "Mark Peters",
            replace(profile, name_replacement_provider=None),
            "name_replacement_unsupported",
        ),
    ):
        spans, warnings, processing = instance._apply_replacements(
            text, [span(text)], {"name_replacement": True}, selected
        )
        assert spans[0]["replacement"] == "[Name:Patient]"
        assert {w["code"] for w in warnings} == {code}
        assert processing["name_replacement"]["placeholder_spans"] == 1


def test_names_do_not_change_dates_or_ages_and_docdate_remains_relevant():
    instance = engine()
    profile = get_profile("nl-BE")
    for source, label, docdate, expected in (
        ("14/05/2013", "Age_Birthdate", None, "[2014]"),
        ("14/09/2022", "Age_Birthdate", "14/05/2026", "[3 jaar, 7 maanden oud]"),
        ("15/01/2025", "Date", "14/05/2026", "[19/02/2026]"),
    ):
        for flag in (False, True):
            spans, _, _ = instance._apply_replacements(
                source,
                [span(source, label)],
                {
                    "name_replacement": flag,
                    "date_shift_days": 400,
                    "document_creation_date": docdate,
                },
                profile,
            )
            assert spans[0]["replacement"] == expected


def test_http_single_batch_schema_and_missing_configuration():
    with TestClient(create_app(engine=engine())) as client:
        schema = client.get("/openapi.json").json()["components"]["schemas"][
            "RecordMetadata"
        ]
        assert schema["properties"]["name_replacement"]["default"] is False
        payload = {"text": "Mark Peters", "metadata": {"name_replacement": True}}
        response = client.post("/deidentify", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["deid_text"] == "[Sue-Ann Colen]"
        assert response.json()["processing"]["name_replacement"]["replaced_spans"] == 1
        batch = client.post(
            "/deidentify-batch",
            json={"documents": [{"document_id": str(i), **payload} for i in range(3)]},
        )
        assert batch.status_code == 200, batch.text
        assert [d["deid_text"] for d in batch.json()["documents"]] == [
            "[Sue-Ann Colen]"
        ] * 3
        for flag in ("true", 1, None):
            assert (
                client.post(
                    "/deidentify",
                    json={
                        "text": "Mark Peters",
                        "metadata": {"name_replacement": flag},
                    },
                ).status_code
                == 422
            )
        assert (
            client.post("/deidentify", json={"text": "Mark Peters"}).json()["deid_text"]
            == "[Name:Patient]"
        )
    with TestClient(create_app(engine=engine(configured=False))) as client:
        response = client.post("/deidentify", json=payload)
        assert response.status_code == 422


def test_cli_uses_the_same_boolean(tmp_path, monkeypatch, capsys):
    source = tmp_path / "note.txt"
    source.write_text("Mark Peters")
    monkeypatch.setattr(cli, "_load_engine", lambda args: engine())
    assert (
        cli.main(
            [
                "deidentify",
                str(source),
                "--name-replacement",
                "--model",
                "test/model",
                "--quiet",
            ]
        )
        == 0
    )
    assert "[Sue-Ann Colen]" in capsys.readouterr().out


def test_engine_captures_deployment_settings_once(monkeypatch):
    import meddeid.api as api

    monkeypatch.setenv("MEDDEID_NAME_KEY", KEY.hex())
    monkeypatch.setenv("MEDDEID_NAME_SCOPE", "study-a")
    monkeypatch.setattr(api, "DeidentificationPipeline", lambda bundle: None)
    instance = Deidentifier(
        SimpleNamespace(
            postprocess=SimpleNamespace(profiles=[SimpleNamespace(profile_id="nl-BE")])
        ),
        SimpleNamespace(),
        model_source="synthetic/model",
        requested_revision=None,
        resolved_revision=None,
        offline=True,
    )
    monkeypatch.setenv("MEDDEID_NAME_SCOPE", "changed-after-start")
    assert instance._names_for_metadata({"name_replacement": True}) == (KEY, "study-a")
    assert instance._names_for_metadata({"name_replacement": False}) is None
