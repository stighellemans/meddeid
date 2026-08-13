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
