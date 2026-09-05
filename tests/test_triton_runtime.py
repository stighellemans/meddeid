from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from types import ModuleType

from meddeid.pipeline.types import PreparedWindow
from meddeid.runtime.triton import TritonRuntime


class FakeResponse:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.body


def test_triton_runtime_uses_v2_tensor_contract_and_restores_lengths(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        payload = json.loads(request.data)
        assert payload["inputs"][0]["datatype"] == "INT32"
        assert payload["inputs"][0]["shape"] == [1, 8]
        body = {
            "outputs": [
                {"name": "bio_logits", "shape": [1, 8, 3], "data": [0.0] * 24},
                {"name": "label_logits", "shape": [1, 8, 14], "data": [0.0] * 112},
            ]
        }
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr("meddeid.runtime.triton.urlopen", fake_urlopen)
    runtime = TritonRuntime(
        SimpleNamespace(name="meddeid-dutch-synth", model_version="1"),
        base_url="http://triton:8000",
    )
    window = PreparedWindow(
        doc_index=0,
        begin=0,
        end=3,
        input_ids=[0, 10, 2],
        attention_mask=[1, 1, 1],
        special_tokens_mask=[1, 0, 1],
    )
    result = runtime.infer_windows([window])
    assert result[0].bio_logits.shape == (3, 3)
    assert result[0].label_logits.shape == (3, 14)
    assert requests[0][0].full_url.endswith(
        "/v2/models/meddeid-dutch-synth/versions/1/infer"
    )


def test_binary_transport_uses_binary_tensor_extension(monkeypatch) -> None:
    recorded = {}

    class InferInput:
        def __init__(self, name, shape, datatype):
            self.name = name
            self.shape = tuple(shape)
            self.datatype = datatype

        def set_data_from_numpy(self, value, *, binary_data):
            assert binary_data is True
            self.value = value

    class InferRequestedOutput:
        def __init__(self, name, *, binary_data):
            assert binary_data is True
            self.name = name

    class Response:
        def as_numpy(self, name):
            dimensions = 3 if name == "bio_logits" else 14
            return __import__("numpy").zeros((1, 8, dimensions), dtype="float32")

    class Client:
        def __init__(self, **kwargs):
            recorded["client"] = kwargs

        def infer(self, **kwargs):
            recorded["infer"] = kwargs
            return Response()

    package = ModuleType("tritonclient")
    package.__path__ = []
    http = ModuleType("tritonclient.http")
    http.InferInput = InferInput
    http.InferRequestedOutput = InferRequestedOutput
    http.InferenceServerClient = Client
    package.http = http
    monkeypatch.setitem(sys.modules, "tritonclient", package)
    monkeypatch.setitem(sys.modules, "tritonclient.http", http)

    runtime = TritonRuntime(
        SimpleNamespace(name="meddeid-dutch-synth", model_version="1"),
        base_url="http://triton:8000",
        transport="binary",
    )
    window = PreparedWindow(
        doc_index=0,
        begin=0,
        end=3,
        input_ids=[0, 10, 2],
        attention_mask=[1, 1, 1],
        special_tokens_mask=[1, 0, 1],
    )

    result = runtime.infer_windows([window])

    assert runtime.supports_concurrent_requests is True
    assert recorded["client"]["url"] == "triton:8000"
    assert recorded["infer"]["inputs"][0].datatype == "INT32"
    assert recorded["infer"]["inputs"][0].shape == (1, 8)
    assert recorded["infer"]["timeout"] == 30_000_000
    assert result[0].bio_logits.shape == (3, 3)
    assert result[0].label_logits.shape == (3, 14)
