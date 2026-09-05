from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from meddeid import server
from meddeid.runtime.microbatch import MicroBatchRuntime
from meddeid.server import create_app

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


class FakeEngine:
    def __init__(
        self,
        *,
        source="test/model",
        profiles=("nl-BE",),
        default_profile="nl-BE",
    ):
        self.closed = False
        self.source = source
        self.profiles = profiles
        self.default_profile = default_profile

    def model_info(self):
        return {
            "model": {
                "source": self.source,
                "name": "test",
                "version": "1",
                "resolved_revision": "abc123",
            },
            "contracts": {
                "language_profiles": [
                    {"profile_id": profile} for profile in self.profiles
                ],
                "default_language_profile": self.default_profile,
            },
            "runtime": {"ready": True, "backend": "torch", "device": "cpu"},
            "environment": {},
        }

    def __call__(self, text, *, metadata):
        profile = metadata.get("lang") or self.default_profile or self.profiles[0]
        return SimpleNamespace(
            text=text,
            deid_text=text,
            spans=[],
            language_profile=profile,
            warnings=[],
            processing={},
            provenance={
                **PROVENANCE,
                "language_profile": {"profile_id": profile},
            },
        )

    def close(self):
        self.closed = True


def test_server_validates_input_and_reports_model_health() -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine, max_input_chars=12)) as client:
        assert client.get("/").json()["service"] == "MedDeID"
        assert client.get("/").json()["ui"] == "/ui"
        ui = client.get("/ui")
        assert ui.status_code == 200
        assert "De-identify one clinical note" in ui.text
        assert 'id="current-model"' in ui.text
        assert 'id="language-field"' in ui.text
        assert "language.disabled = profiles.length === 1" in ui.text
        assert "metadata.lang = language.value" in ui.text
        assert "launchParameters.get('api_key')" in ui.text
        assert "history.replaceState" in ui.text
        assert "default-src 'none'" in ui.headers["content-security-policy"]
        assert client.get("/live").json() == {"status": "ok"}
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "ready": True,
            "serving": {
                "profile": "latency",
                "gateway_microbatching": False,
                "max_concurrent_requests": 1,
            },
            "model": {"name": "test", "version": "1"},
            "contracts": {
                "language_profiles": [{"profile_id": "nl-BE"}],
                "default_language_profile": "nl-BE",
            },
        }

        assert client.post("/deidentify", json={}).status_code == 422
        assert client.post("/deidentify", json={"text": "   "}).status_code == 422
        assert client.post("/deidentify", json={"text": "x" * 13}).status_code == 422
        response = client.post("/deidentify", json={"text": " Jan "})
        assert response.status_code == 200
        assert list(response.json()) == [
            "deid_text",
            "spans",
            "processing",
            "warnings",
            "provenance",
        ]
        assert "text" not in response.json()
        assert response.json()["provenance"]["language_profile"] == {
            "profile_id": "nl-BE"
        }
        assert response.json()["deid_text"] == " Jan "

        invalid_label = client.post(
            "/deidentify",
            json={
                "text": "Jan",
                "metadata": {"known_values": [{"value": "Jan", "label": "Secret"}]},
            },
        )
        assert invalid_label.status_code == 422

        batch = client.post(
            "/deidentify-batch",
            json={
                "documents": [
                    {
                        "document_id": "note-1",
                        "text": "Jan",
                        "metadata": {
                            "lang": "nl-BE",
                            "patient": {"given_name": "Jan"},
                            "known_values": [{"value": "Jan", "label": "Name:Patient"}],
                        },
                    },
                    {"document_id": "note-2", "text": "Piet"},
                ]
            },
        )
        assert batch.status_code == 200
        assert [item["document_id"] for item in batch.json()["documents"]] == [
            "note-1",
            "note-2",
        ]
        assert batch.json()["documents"][0]["provenance"]["language_profile"] == {
            "profile_id": "nl-BE"
        }
        assert list(batch.json()["documents"][0]) == [
            "document_id",
            "deid_text",
            "spans",
            "processing",
            "warnings",
            "provenance",
        ]
        assert "text" not in batch.json()["documents"][0]

        retired_identity_key = client.post(
            "/deidentify",
            json={"text": "Jan", "metadata": {"patient_name": {"given_name": "Jan"}}},
        )
        assert retired_identity_key.status_code == 422
    assert engine.closed


def test_local_ui_does_not_embed_configured_api_key() -> None:
    with TestClient(
        create_app(
            engine=FakeEngine(),
            api_key="correct-horse-battery-staple",
            ui_enabled=True,
        )
    ) as client:
        assert "correct-horse-battery-staple" not in client.get("/ui").text


def test_server_passes_configured_language_profile_to_engine(monkeypatch) -> None:
    captured = {}

    def fake_load(*args, **kwargs):
        captured.update(kwargs)
        return FakeEngine()

    monkeypatch.setattr("meddeid.server.Deidentifier.from_pretrained", fake_load)
    monkeypatch.setenv("MEDDEID_LANGUAGE_PROFILE", "en-GB")
    monkeypatch.setenv("MEDDEID_MODEL", "stighellemans/meddeid-english-synth")
    monkeypatch.setenv("MEDDEID_AGE_GRANULARITY_CONFIG", "/config/age.json")
    monkeypatch.setenv("MEDDEID_MIN_RECOMMENDED_DATE_SHIFT_DAYS", "500")
    monkeypatch.setenv("MEDDEID_TRITON_TRANSPORT", "binary")
    monkeypatch.setenv("MEDDEID_TORCH_PRECISION", "fp16")
    monkeypatch.setenv("MEDDEID_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("MEDDEID_TORCH_COMPILE_DYNAMIC", "false")

    with TestClient(create_app(ui_enabled=False)):
        pass

    assert captured["language_profile"] == "en-GB"
    assert captured["age_granularity_config"] == "/config/age.json"
    assert captured["min_recommended_date_shift_days"] == 500
    assert captured["triton_transport"] == "binary"
    assert captured["torch_precision"] == "fp16"
    assert captured["torch_compile_mode"] == "reduce-overhead"
    assert captured["torch_compile_dynamic"] is False


def test_throughput_profile_wraps_runtime_with_bounded_microbatcher(
    monkeypatch,
) -> None:
    class Runtime:
        max_windows_per_batch = 32

        def infer_windows(self, windows):
            return windows

        def healthcheck(self):
            return {"ready": True, "backend": "torch"}

        def close(self):
            return None

    class ProfileEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.runtime = Runtime()
            self.bundle = SimpleNamespace(inference=SimpleNamespace(max_length=512))

        def close(self):
            self.runtime.close()
            super().close()

    engine = ProfileEngine()
    monkeypatch.setattr(
        "meddeid.server.Deidentifier.from_pretrained", lambda *_args, **_kwargs: engine
    )
    monkeypatch.setenv("MEDDEID_MODEL", "test/model")
    monkeypatch.setenv("MEDDEID_DEVICE", "cuda")
    monkeypatch.setenv("MEDDEID_SERVING_PROFILE", "throughput")
    monkeypatch.setenv("MEDDEID_MICROBATCH_MAX_WINDOWS", "16")
    monkeypatch.setenv("MEDDEID_SEQUENCE_LENGTH_BUCKETS", "128,256,512")

    with TestClient(create_app(ui_enabled=False)) as client:
        assert isinstance(engine.runtime, MicroBatchRuntime)
        assert engine.runtime.max_windows == 16
        assert engine.runtime.sequence_buckets == (128, 256, 512)
        assert client.get("/health").json()["serving"]["max_concurrent_requests"] == 16

    assert engine.closed is True


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_throughput_profile_detects_automatically_selected_gpu(
    monkeypatch, device_type: str
) -> None:
    class Runtime:
        max_windows_per_batch = 32
        device = SimpleNamespace(type=device_type)

        def infer_windows(self, windows):
            return windows

        def healthcheck(self):
            return {"ready": True, "backend": "torch"}

        def close(self):
            return None

    class ProfileEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.runtime = Runtime()
            self.bundle = SimpleNamespace(inference=SimpleNamespace(max_length=512))

        def close(self):
            self.runtime.close()
            super().close()

    engine = ProfileEngine()
    monkeypatch.setattr(
        "meddeid.server.Deidentifier.from_pretrained", lambda *_args, **_kwargs: engine
    )
    monkeypatch.setenv("MEDDEID_MODEL", "test/model")
    monkeypatch.delenv("MEDDEID_DEVICE", raising=False)
    monkeypatch.setenv("MEDDEID_SERVING_PROFILE", "throughput")

    with TestClient(create_app(ui_enabled=False)):
        assert isinstance(engine.runtime, MicroBatchRuntime)

    assert engine.closed is True


def test_throughput_profile_leaves_batching_to_triton_by_default(monkeypatch) -> None:
    class Runtime:
        max_windows_per_batch = 64

        def infer_windows(self, windows):
            return windows

        def healthcheck(self):
            return {"ready": True, "backend": "triton"}

        def close(self):
            return None

    class ProfileEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.runtime = Runtime()
            self.bundle = SimpleNamespace(inference=SimpleNamespace(max_length=512))

    engine = ProfileEngine()
    monkeypatch.setattr(
        "meddeid.server.Deidentifier.from_pretrained", lambda *_args, **_kwargs: engine
    )
    monkeypatch.setenv("MEDDEID_MODEL", "test/model")
    monkeypatch.setenv("MEDDEID_BACKEND", "triton")
    monkeypatch.setenv("MEDDEID_SERVING_PROFILE", "throughput")

    with TestClient(create_app(ui_enabled=False)) as client:
        assert engine.runtime.__class__ is Runtime
        assert client.get("/health").json()["serving"] == {
            "profile": "throughput",
            "gateway_microbatching": False,
            "max_concurrent_requests": 8,
        }


def test_triton_throughput_profile_can_enable_nested_microbatching(monkeypatch) -> None:
    class Runtime:
        max_windows_per_batch = 64

        def infer_windows(self, windows):
            return windows

        def healthcheck(self):
            return {"ready": True, "backend": "triton"}

        def close(self):
            return None

    class ProfileEngine(FakeEngine):
        def __init__(self):
            super().__init__()
            self.runtime = Runtime()
            self.bundle = SimpleNamespace(inference=SimpleNamespace(max_length=512))

        def close(self):
            self.runtime.close()
            super().close()

    engine = ProfileEngine()
    monkeypatch.setattr(
        "meddeid.server.Deidentifier.from_pretrained", lambda *_args, **_kwargs: engine
    )
    monkeypatch.setenv("MEDDEID_MODEL", "test/model")
    monkeypatch.setenv("MEDDEID_BACKEND", "triton")
    monkeypatch.setenv("MEDDEID_SERVING_PROFILE", "throughput")
    monkeypatch.setenv("MEDDEID_MICROBATCH_ENABLED", "true")

    with TestClient(create_app(ui_enabled=False)) as client:
        assert isinstance(engine.runtime, MicroBatchRuntime)
        assert client.get("/health").json()["serving"]["gateway_microbatching"] is True


def test_server_requires_explicit_model(monkeypatch) -> None:
    monkeypatch.delenv("MEDDEID_MODEL", raising=False)

    with pytest.raises(ValueError, match="no model selected"):
        create_app(ui_enabled=False)


def test_server_model_allowlist_rejects_before_loading(monkeypatch) -> None:
    called = False

    def fake_load(*_args, **_kwargs):
        nonlocal called
        called = True
        return FakeEngine()

    monkeypatch.setattr("meddeid.server.Deidentifier.from_pretrained", fake_load)
    monkeypatch.setenv("MEDDEID_MODEL", "unapproved/model")
    monkeypatch.setenv("MEDDEID_ALLOWED_MODELS", "approved/model")

    with pytest.raises(ValueError, match="not permitted"):
        create_app(ui_enabled=False)

    assert called is False


def test_server_language_profile_allowlist_limits_requests_and_health() -> None:
    engine = FakeEngine(
        profiles=("en-GB", "en-US"),
        default_profile="en-GB",
    )
    with TestClient(
        create_app(
            engine=engine,
            allowed_language_profiles={"en-GB"},
            ui_enabled=False,
        )
    ) as client:
        health = client.get("/health").json()
        assert health["contracts"] == {
            "language_profiles": [{"profile_id": "en-GB"}],
            "default_language_profile": "en-GB",
        }
        allowed = client.post(
            "/deidentify",
            json={"text": "Alex", "metadata": {"lang": "en_GB"}},
        )
        assert allowed.status_code == 200
        denied = client.post(
            "/deidentify",
            json={"text": "Alex", "metadata": {"lang": "en-US"}},
        )
        assert denied.status_code == 422
        assert denied.json()["detail"]["code"] == "language_profile_not_allowed"


def test_server_language_profile_allowlist_rejects_configuration_typos() -> None:
    engine = FakeEngine()
    with pytest.raises(ValueError, match="not declared"):
        create_app(
            engine=engine,
            allowed_language_profiles={"nl-NL"},
            ui_enabled=False,
        )
    assert engine.closed is True


def test_single_allowed_profile_becomes_server_fallback(monkeypatch) -> None:
    captured = {}

    def fake_load(model, **kwargs):
        captured.update(kwargs)
        return FakeEngine(
            source=model,
            profiles=("en-GB", "en-US"),
            default_profile="en-GB",
        )

    monkeypatch.setattr("meddeid.server.Deidentifier.from_pretrained", fake_load)
    monkeypatch.setenv("MEDDEID_MODEL", "example/english-model")
    monkeypatch.delenv("MEDDEID_LANGUAGE_PROFILE", raising=False)
    monkeypatch.setenv("MEDDEID_ALLOWED_LANGUAGE_PROFILES", "en-GB")

    with TestClient(create_app(ui_enabled=False)):
        pass

    assert captured["language_profile"] == "en-GB"


def test_server_command_requires_explicit_model(monkeypatch, capsys) -> None:
    monkeypatch.delenv("MEDDEID_MODEL", raising=False)

    with pytest.raises(SystemExit) as error:
        server.main([])

    assert error.value.code == 2
    assert "MEDDEID_MODEL is required" in capsys.readouterr().err


def test_server_command_accepts_serving_profile_flag(monkeypatch) -> None:
    monkeypatch.setenv("MEDDEID_MODEL", "approved/dutch-model")
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", fake_run)

    server.main(["--serving-profile", "throughput"])

    assert os.environ["MEDDEID_SERVING_PROFILE"] == "throughput"
    assert captured["args"] == ("meddeid.server:create_app",)


def test_server_command_loads_reproducible_environment_file(
    monkeypatch, tmp_path
) -> None:
    environment_file = tmp_path / "server.env"
    environment_file.write_text(
        "\n".join(
            [
                "# One model and locale are approved for this service.",
                "MEDDEID_MODEL=approved/dutch-model",
                "MEDDEID_ALLOWED_MODELS=approved/dutch-model",
                "MEDDEID_ALLOWED_LANGUAGE_PROFILES=nl-BE",
                "MEDDEID_BIND_ADDRESS=127.0.0.2",
                "MEDDEID_PORT=8123",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MEDDEID_MODEL", raising=False)
    monkeypatch.delenv("MEDDEID_ALLOWED_MODELS", raising=False)
    monkeypatch.delenv("MEDDEID_ALLOWED_LANGUAGE_PROFILES", raising=False)
    monkeypatch.delenv("MEDDEID_BIND_ADDRESS", raising=False)
    monkeypatch.delenv("MEDDEID_PORT", raising=False)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("uvicorn.run", fake_run)
    loaded_keys = (
        "MEDDEID_MODEL",
        "MEDDEID_ALLOWED_MODELS",
        "MEDDEID_ALLOWED_LANGUAGE_PROFILES",
        "MEDDEID_BIND_ADDRESS",
        "MEDDEID_PORT",
    )
    try:
        server.main(["--env-file", str(environment_file)])

        assert captured["kwargs"]["host"] == "127.0.0.2"
        assert captured["kwargs"]["port"] == 8123
        assert os.environ["MEDDEID_MODEL"] == "approved/dutch-model"
        assert os.environ["MEDDEID_ALLOWED_MODELS"] == "approved/dutch-model"
        assert os.environ["MEDDEID_ALLOWED_LANGUAGE_PROFILES"] == "nl-BE"
    finally:
        for key in loaded_keys:
            os.environ.pop(key, None)


def test_server_environment_file_does_not_override_injected_secret(
    monkeypatch, tmp_path
) -> None:
    environment_file = tmp_path / "server.env"
    environment_file.write_text(
        "MEDDEID_API_KEY=file-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDDEID_API_KEY", "injected-secret")

    server._load_server_environment(environment_file)

    assert os.environ["MEDDEID_API_KEY"] == "injected-secret"


def test_server_environment_file_rejects_unknown_settings(tmp_path) -> None:
    environment_file = tmp_path / "server.env"
    environment_file.write_text("MEDDEID_MODLE=typo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown server setting"):
        server._load_server_environment(environment_file)


def test_server_environment_template_contains_only_supported_settings() -> None:
    template = Path(__file__).parents[1] / "server.env.example"
    template_keys = {
        line.split("=", 1)[0].strip()
        for line in template.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert template_keys == server.SERVER_ENVIRONMENT_KEYS


def test_openapi_describes_the_current_typed_result_contract() -> None:
    schema = create_app(engine=FakeEngine(), ui_enabled=False).openapi()
    for request_schema_name in (
        "DeidentifyRequest",
        "BatchRequest",
        "BatchDocumentRequest",
        "RecordMetadata",
        "PatientMetadata",
        "PersonMetadata",
        "KnownValueMetadata",
    ):
        assert (
            schema["components"]["schemas"][request_schema_name]["additionalProperties"]
            is False
        )

    response_ref = schema["paths"]["/deidentify"]["post"]["responses"]["200"][
        "content"
    ]["application/json"]["schema"]["$ref"]
    response_name = response_ref.rsplit("/", 1)[-1]
    response_schema = schema["components"]["schemas"][response_name]

    assert list(response_schema["properties"]) == [
        "deid_text",
        "spans",
        "processing",
        "warnings",
        "provenance",
    ]
    assert "language_profile" not in response_schema["properties"]

    span_ref = response_schema["properties"]["spans"]["items"]["$ref"]
    span_schema = schema["components"]["schemas"][span_ref.rsplit("/", 1)[-1]]
    assert span_schema["additionalProperties"] is False
    assert list(span_schema["properties"]) == [
        "begin",
        "end",
        "label",
        "text",
        "category",
        "subtype",
        "score",
        "replacement",
    ]

    provenance_ref = response_schema["properties"]["provenance"]["$ref"]
    provenance_schema = schema["components"]["schemas"][
        provenance_ref.rsplit("/", 1)[-1]
    ]
    assert list(provenance_schema["properties"]) == [
        "contract_version",
        "software",
        "model",
        "language_profile",
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "Jan", "unexpected": True},
        {"text": "Jan", "metadata": {"unexpected": True}},
        {
            "text": "Jan",
            "metadata": {"patient": {"given_name": "Jan", "unexpected": True}},
        },
        {
            "text": "Dr. Jan",
            "metadata": {"caregivers": [{"given_name": "Jan", "unexpected": True}]},
        },
        {
            "text": "Jan",
            "metadata": {
                "known_values": [
                    {
                        "value": "Jan",
                        "label": "Name:Patient",
                        "unexpected": True,
                    }
                ]
            },
        },
    ],
)
def test_server_rejects_unknown_request_properties(payload) -> None:
    with TestClient(create_app(engine=FakeEngine(), ui_enabled=False)) as client:
        response = client.post("/deidentify", json=payload)

    assert response.status_code == 422


def test_server_rejects_unknown_batch_properties() -> None:
    with TestClient(create_app(engine=FakeEngine(), ui_enabled=False)) as client:
        unknown_batch_field = client.post(
            "/deidentify-batch",
            json={
                "documents": [{"document_id": "one", "text": "Jan"}],
                "unexpected": True,
            },
        )
        unknown_document_field = client.post(
            "/deidentify-batch",
            json={
                "documents": [
                    {
                        "document_id": "one",
                        "text": "Jan",
                        "unexpected": True,
                    }
                ]
            },
        )

    assert unknown_batch_field.status_code == 422
    assert unknown_document_field.status_code == 422


def test_server_rejects_boolean_date_shift() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        response = client.post(
            "/deidentify",
            json={"text": "15/01/2025", "metadata": {"date_shift_days": True}},
        )
    assert response.status_code == 422


def test_server_rejects_request_level_age_policy_selection() -> None:
    with TestClient(create_app(engine=FakeEngine())) as client:
        response = client.post(
            "/deidentify",
            json={
                "text": "42 jaar",
                "metadata": {"age_granularity_policy": "request-choice"},
            },
        )
    assert response.status_code == 422


def test_server_rejects_oversized_batches() -> None:
    engine = FakeEngine()
    with TestClient(
        create_app(
            engine=engine,
            max_input_chars=12,
            max_batch_documents=2,
            max_batch_chars=5,
        )
    ) as client:
        too_many = client.post(
            "/deidentify-batch",
            json={
                "documents": [
                    {"document_id": "1", "text": "a"},
                    {"document_id": "2", "text": "b"},
                    {"document_id": "3", "text": "c"},
                ]
            },
        )
        assert too_many.status_code == 422
        too_large = client.post(
            "/deidentify-batch",
            json={"documents": [{"document_id": "1", "text": "abcdef"}]},
        )
        assert too_large.status_code == 413


def test_server_api_key_request_limits_and_security_headers() -> None:
    engine = FakeEngine()
    with TestClient(
        create_app(
            engine=engine,
            api_key="correct-horse-battery-staple",
            max_request_bytes=80,
            docs_enabled=False,
            ui_enabled=False,
        )
    ) as client:
        health = client.get("/health", headers={"X-Request-ID": "health-check-1"})
        assert health.status_code == 200
        assert health.headers["x-request-id"] == "health-check-1"
        assert health.headers["cache-control"] == "no-store"
        assert health.headers["x-content-type-options"] == "nosniff"

        assert client.get("/docs").status_code == 404
        assert client.get("/ui").status_code == 404

        unauthorized = client.post("/deidentify", json={"text": "Jan"})
        assert unauthorized.status_code == 401
        assert unauthorized.json()["detail"]["code"] == "unauthorized"

        bearer = client.post(
            "/deidentify",
            json={"text": "Jan"},
            headers={"Authorization": "Bearer correct-horse-battery-staple"},
        )
        assert bearer.status_code == 200

        header = client.post(
            "/deidentify",
            json={"text": "Jan"},
            headers={"X-API-Key": "correct-horse-battery-staple"},
        )
        assert header.status_code == 200

        oversized = client.post(
            "/deidentify",
            content=b"x" * 81,
            headers={
                "content-type": "application/json",
                "X-API-Key": "correct-horse-battery-staple",
            },
        )
        assert oversized.status_code == 413
        assert oversized.json()["detail"]["code"] == "request_too_large"
        assert oversized.headers["cache-control"] == "no-store"
        assert oversized.headers["x-frame-options"] == "DENY"


def test_server_can_require_an_api_key_at_startup(monkeypatch) -> None:
    monkeypatch.delenv("MEDDEID_API_KEY", raising=False)
    monkeypatch.setenv("MEDDEID_REQUIRE_API_KEY", "true")
    try:
        create_app(engine=FakeEngine())
    except ValueError as exc:
        assert "MEDDEID_API_KEY is empty" in str(exc)
    else:
        raise AssertionError("create_app should reject an empty required API key")
