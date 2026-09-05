from pathlib import Path
import sys
from types import ModuleType

from meddeid.model_source import resolve_model_source


def test_remote_gateway_resolution_omits_checkpoint_weights(monkeypatch, tmp_path) -> None:
    calls = []

    def snapshot_download(
        *,
        repo_id,
        revision,
        cache_dir,
        token,
        local_files_only,
        allow_patterns,
    ):
        calls.append(allow_patterns)
        return str(tmp_path)

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    resolved = resolve_model_source(
        "example/gateway-model",
        revision="abc123",
        local_files_only=True,
        include_weights=False,
    )

    assert resolved.root == tmp_path
    assert "model.safetensors" not in calls[0]
    assert "model.pt" not in calls[0]
    assert "bundle.json" in calls[0]
