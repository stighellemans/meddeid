from __future__ import annotations

from typing import Any

import numpy as np

from meddeid.bundle import ModelBundle
from meddeid.model import DualHeadTokenClassifier, load_checkpoint
from meddeid.pipeline.types import PreparedWindow, WindowPrediction


class TorchRuntime:
    def __init__(
        self,
        bundle: ModelBundle,
        *,
        device: str | None = None,
        max_windows_per_batch: int = 32,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("the meddeid installation is missing PyTorch") from exc
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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

    def infer_windows(self, windows: list[PreparedWindow]) -> list[WindowPrediction]:
        if not windows:
            return []
        results: list[WindowPrediction] = []
        for begin in range(0, len(windows), self.max_windows_per_batch):
            results.extend(self._infer_batch(windows[begin : begin + self.max_windows_per_batch]))
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
        with torch.inference_mode():
            output = self.model(
                torch.tensor(input_ids, dtype=torch.long, device=self.device),
                torch.tensor(attention, dtype=torch.long, device=self.device),
            )
        results: list[WindowPrediction] = []
        for index, window in enumerate(windows):
            length = window.sequence_length
            results.append(
                WindowPrediction(
                    bio_logits=np.asarray(output.bio_logits[index, :length].cpu(), dtype=np.float64),
                    label_logits=np.asarray(output.label_logits[index, :length].cpu(), dtype=np.float64),
                )
            )
        return results

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "ready": True,
            "backend": "torch",
            "device": str(self.device),
            "max_windows_per_batch": self.max_windows_per_batch,
        }

    def close(self) -> None:
        return None
