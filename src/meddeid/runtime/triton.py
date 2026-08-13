from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from meddeid.bundle import ModelBundle
from meddeid.pipeline.types import PreparedWindow, WindowPrediction


class TritonRuntime:
    """NVIDIA Triton V2 HTTP client for a TensorRT-backed MedDeID model."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_windows_per_batch: int = 16,
        min_sequence_length: int = 8,
    ) -> None:
        if not base_url.strip():
            raise ValueError("the Triton backend requires --triton-url or MEDDEID_TRITON_URL")
        self.bundle = bundle
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_windows_per_batch = max(1, int(max_windows_per_batch))
        self.min_sequence_length = max(1, int(min_sequence_length))
        self.infer_url = (
            f"{self.base_url}/v2/models/{bundle.name}/versions/{bundle.model_version}/infer"
        )

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        results: list[WindowPrediction] = []
        for begin in range(0, len(windows), self.max_windows_per_batch):
            results.extend(self._infer_batch(windows[begin : begin + self.max_windows_per_batch]))
        return results

    def _infer_batch(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        if not windows:
            return []
        sequence_lengths = [window.sequence_length for window in windows]
        max_length = max(max(sequence_lengths), self.min_sequence_length)
        input_ids = [
            window.input_ids + [0] * (max_length - window.sequence_length)
            for window in windows
        ]
        attention_mask = [
            window.attention_mask + [0] * (max_length - window.sequence_length)
            for window in windows
        ]
        payload = {
            "inputs": [
                {
                    "name": "input_ids",
                    "datatype": "INT32",
                    "shape": [len(windows), max_length],
                    "data": input_ids,
                },
                {
                    "name": "attention_mask",
                    "datatype": "INT32",
                    "shape": [len(windows), max_length],
                    "data": attention_mask,
                },
            ],
            "outputs": [{"name": "bio_logits"}, {"name": "label_logits"}],
        }
        body = self._request_json(self.infer_url, method="POST", payload=payload)
        outputs = {
            str(item["name"]): np.asarray(item["data"], dtype=np.float64).reshape(item["shape"])
            for item in body.get("outputs", [])
        }
        if "bio_logits" not in outputs or "label_logits" not in outputs:
            raise RuntimeError("Triton did not return bio_logits and label_logits")
        return [
            WindowPrediction(
                bio_logits=outputs["bio_logits"][index, :length],
                label_logits=outputs["label_logits"][index, :length],
            )
            for index, length in enumerate(sequence_lengths)
        ]

    def healthcheck(self) -> dict[str, Any]:
        checks = {
            "server_live": f"{self.base_url}/v2/health/live",
            "server_ready": f"{self.base_url}/v2/health/ready",
            "model_ready": f"{self.base_url}/v2/models/{self.bundle.name}/ready",
        }
        status: dict[str, Any] = {
            "backend": "triton",
            "device": "tensorrt",
            "url": self.base_url,
            "max_windows_per_batch": self.max_windows_per_batch,
        }
        for name, url in checks.items():
            try:
                self._request_json(url, method="GET", allow_empty=True)
                status[name] = True
            except RuntimeError as exc:
                status[name] = False
                status.setdefault("error", str(exc))
        status["ready"] = all(bool(status[name]) for name in checks)
        status["status"] = "ok" if status["ready"] else "unavailable"
        return status

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content = response.read()
        except HTTPError as exc:
            raise RuntimeError(f"Triton returned HTTP {exc.code} for {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"cannot reach Triton at {self.base_url}: {exc.reason}") from exc
        if not content and allow_empty:
            return {}
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Triton returned invalid JSON for {url}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Triton returned an unexpected response for {url}")
        return parsed

    def close(self) -> None:
        return None
