from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from meddeid.bundle import ModelBundle
from meddeid.model import DualHeadTokenClassifier, load_checkpoint
from meddeid.pipeline.types import PreparedWindow, WindowPrediction


def _automatic_device(torch: Any) -> str:
    """Prefer an available local GPU while retaining a portable CPU fallback."""

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _normalize_precision(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"fp32", "fp16"}:
        raise ValueError("torch precision must be 'fp32' or 'fp16'")
    return normalized


def _normalize_compile_mode(value: str) -> str:
    normalized = value.strip().lower()
    allowed = {"off", "default", "reduce-overhead", "max-autotune"}
    if normalized not in allowed:
        raise ValueError(
            "torch compile mode must be off, default, reduce-overhead, or max-autotune"
        )
    return normalized


class TorchRuntime:
    supports_concurrent_requests = False

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        device: str | None = None,
        max_windows_per_batch: int = 32,
        precision: str = "fp32",
        compile_mode: str = "off",
        compile_dynamic: bool | None = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("the meddeid installation is missing PyTorch") from exc
        self.torch = torch
        self.device = torch.device(device or _automatic_device(torch))
        self.precision = _normalize_precision(precision)
        self.compile_mode = _normalize_compile_mode(compile_mode)
        self.compile_dynamic = compile_dynamic
        if self.precision == "fp16" and self.device.type != "cuda":
            raise ValueError("torch fp16 inference is supported only on CUDA")
        self.max_windows_per_batch = max(1, int(max_windows_per_batch))
        model_source = (
            str(bundle.encoder_config_path.parent)
            if bundle.encoder_config_path is not None
            else bundle.base_encoder
        )
        self.model = DualHeadTokenClassifier(
            model_source,
            num_bio_labels=len(bundle.bio_labels),
            num_entity_labels=len(bundle.entity_labels),
            initialize_from_pretrained=bundle.encoder_config_path is None,
            local_files_only=bundle.encoder_config_path is not None,
        )
        _, state = load_checkpoint(bundle.checkpoint_path)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        if self.compile_mode != "off":
            compile_kwargs: dict[str, Any] = {}
            if self.compile_dynamic is not None:
                compile_kwargs["dynamic"] = self.compile_dynamic
            if self.compile_mode != "default":
                compile_kwargs["mode"] = self.compile_mode
            self.model = torch.compile(self.model, **compile_kwargs)

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        if not windows:
            return []
        results: list[WindowPrediction] = []
        for begin in range(0, len(windows), self.max_windows_per_batch):
            results.extend(
                self._infer_batch(windows[begin : begin + self.max_windows_per_batch])
            )
        return results

    def _infer_batch(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        torch = self.torch
        max_length = max(window.sequence_length for window in windows)
        input_ids = []
        attention = []
        for window in windows:
            padding = max_length - window.sequence_length
            input_ids.append(window.input_ids + [0] * padding)
            attention.append(window.attention_mask + [0] * padding)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.precision == "fp16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            output = self.model(
                torch.tensor(input_ids, dtype=torch.long, device=self.device),
                torch.tensor(attention, dtype=torch.long, device=self.device),
            )
        results: list[WindowPrediction] = []
        for index, window in enumerate(windows):
            length = window.sequence_length
            results.append(
                WindowPrediction(
                    bio_logits=np.asarray(
                        output.bio_logits[index, :length].cpu(), dtype=np.float64
                    ),
                    label_logits=np.asarray(
                        output.label_logits[index, :length].cpu(), dtype=np.float64
                    ),
                )
            )
        return results

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "backend": "torch",
            "device": str(self.device),
            "precision": self.precision,
            "compile_mode": self.compile_mode,
            "compile_dynamic": self.compile_dynamic,
            "max_windows_per_batch": self.max_windows_per_batch,
        }

    def close(self) -> None:
        return None
