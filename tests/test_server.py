from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from meddeid.server import create_app


class FakeEngine:
    def __init__(self):
        self.closed = False

    def model_info(self):
        return {
            "model": {"name": "test", "resolved_revision": "abc123"},
            "contracts": {"language_profile": "nl-BE"},
            "runtime": {"ready": True, "backend": "torch", "device": "cpu"},
            "environment": {},
        }

    def __call__(self, text, *, metadata):
        return SimpleNamespace(text=text, deid_text=text, spans=[])

    def close(self):
        self.closed = True


def test_server_validates_input_and_reports_model_health() -> None:
    engine = FakeEngine()
    with TestClient(create_app(engine=engine, max_input_chars=12)) as client:
        assert client.get("/").json()["service"] == "MedDeID"
        assert client.get("/").json()["ui"] == "/ui"
        ui = client.get("/ui")
        assert ui.status_code == 200
        assert "De-identify one Dutch clinical note" in ui.text
        assert "default-src 'none'" in ui.headers["content-security-policy"]
        assert client.get("/live").json() == {"status": "ok"}
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"]["resolved_revision"] == "abc123"

        assert client.post("/deidentify", json={}).status_code == 422
        assert client.post("/deidentify", json={"text": "   "}).status_code == 422
        assert client.post("/deidentify", json={"text": "x" * 13}).status_code == 422
        response = client.post("/deidentify", json={"text": " Jan "})
        assert response.status_code == 200
        assert response.json()["text"] == " Jan "

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
                            "known_values": [
                                {"value": "Jan", "label": "Name:Patient"}
                            ],
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

        retired_identity_key = client.post(
            "/deidentify",
            json={"text": "Jan", "metadata": {"patient_name": {"given_name": "Jan"}}},
        )
        assert retired_identity_key.status_code == 422
    assert engine.closed


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
