from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from meddeid.pipeline.types import PreparedWindow, WindowPrediction


class InferenceRuntime(ABC):
    supports_concurrent_requests = False

    @abstractmethod
    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        raise NotImplementedError

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
