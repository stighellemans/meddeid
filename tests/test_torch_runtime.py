from types import SimpleNamespace

import pytest

from meddeid.runtime.torch import (
    _automatic_device,
    _normalize_compile_mode,
    _normalize_precision,
)


def _fake_torch(*, cuda: bool, mps: bool):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_automatic_device_prefers_cuda() -> None:
    assert _automatic_device(_fake_torch(cuda=True, mps=True)) == "cuda"


def test_automatic_device_uses_mps_before_cpu() -> None:
    assert _automatic_device(_fake_torch(cuda=False, mps=True)) == "mps"


def test_automatic_device_falls_back_to_cpu() -> None:
    assert _automatic_device(_fake_torch(cuda=False, mps=False)) == "cpu"


def test_torch_optimization_settings_are_bounded() -> None:
    assert _normalize_precision(" FP16 ") == "fp16"
    assert _normalize_compile_mode("reduce-overhead") == "reduce-overhead"
    with pytest.raises(ValueError, match="precision"):
        _normalize_precision("int8")
    with pytest.raises(ValueError, match="compile mode"):
        _normalize_compile_mode("fastest")
