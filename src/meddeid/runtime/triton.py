from __future__ import annotations

import json
from threading import local
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from meddeid.bundle import ModelBundle
from meddeid.pipeline.types import PreparedWindow, WindowPrediction


class TritonRuntime:
    """NVIDIA Triton V2 HTTP client for a TensorRT-backed MedDeID model."""

    supports_concurrent_requests = True

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_windows_per_batch: int = 16,
        min_sequence_length: int = 8,
        transport: str = "json",
    ) -> None:
        if not base_url.strip():
            raise ValueError("the Triton backend requires --triton-url or MEDDEID_TRITON_URL")
        self.bundle = bundle
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_windows_per_batch = max(1, int(max_windows_per_batch))
        self.min_sequence_length = max(1, int(min_sequence_length))
        self.transport = transport.strip().lower()
        if self.transport not in {"json", "binary"}:
            raise ValueError("Triton transport must be 'json' or 'binary'")
        self._client_state = local()
        if self.transport == "binary":
            try:
                import tritonclient.http as triton_http
            except ImportError as exc:
                raise RuntimeError(
                    "binary Triton transport requires tritonclient[http]"
                ) from exc
            parsed = urlsplit(self.base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("binary Triton transport requires an http(s) URL")
            self._triton_http = triton_http
            self._client_url = parsed.netloc + parsed.path.rstrip("/")
            self._client_ssl = parsed.scheme == "https"
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
        if self.transport == "binary":
            return self._infer_batch_binary(
                windows,
                sequence_lengths=sequence_lengths,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
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

    def _binary_client(self):
        client = getattr(self._client_state, "client", None)
        if client is None:
            client = self._triton_http.InferenceServerClient(
                url=self._client_url,
                ssl=self._client_ssl,
                connection_timeout=self.timeout_seconds,
                network_timeout=self.timeout_seconds,
            )
            self._client_state.client = client
        return client

    def _infer_batch_binary(
        self,
        windows: list[PreparedWindow],
        *,
        sequence_lengths: list[int],
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
    ) -> list[WindowPrediction]:
        triton_http = self._triton_http
        inputs = []
        for name, values in (
            ("input_ids", input_ids),
            ("attention_mask", attention_mask),
        ):
            array = np.asarray(values, dtype=np.int32)
            tensor = triton_http.InferInput(name, array.shape, "INT32")
            tensor.set_data_from_numpy(array, binary_data=True)
            inputs.append(tensor)
        requested = [
            triton_http.InferRequestedOutput(name, binary_data=True)
            for name in ("bio_logits", "label_logits")
        ]
        try:
            response = self._binary_client().infer(
                model_name=self.bundle.name,
                model_version=self.bundle.model_version,
                inputs=inputs,
                outputs=requested,
                # Triton's HTTP inference parameter is an unsigned integer in
                # microseconds; the client connection timeouts above use seconds.
                timeout=max(1, round(self.timeout_seconds * 1_000_000)),
            )
        except Exception as exc:
            raise RuntimeError(f"binary Triton inference failed: {exc}") from exc
        outputs = {
            name: response.as_numpy(name)
            for name in ("bio_logits", "label_logits")
        }
        if any(output is None for output in outputs.values()):
            raise RuntimeError("Triton did not return bio_logits and label_logits")
        return [
            WindowPrediction(
                bio_logits=np.asarray(outputs["bio_logits"][index, :length], dtype=np.float64),
                label_logits=np.asarray(outputs["label_logits"][index, :length], dtype=np.float64),
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
            "transport": self.transport,
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
            raise TypeError(f"Triton returned an unexpected response for {url}")
        return parsed

    def close(self) -> None:
        return None
