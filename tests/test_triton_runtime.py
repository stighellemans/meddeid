from __future__ import annotations

import json
from types import SimpleNamespace

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
