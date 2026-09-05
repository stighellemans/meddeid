from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import InferenceRuntime

if TYPE_CHECKING:
    from .microbatch import MicroBatchRuntime
    from .torch import TorchRuntime
    from .triton import TritonRuntime

__all__ = ["InferenceRuntime", "MicroBatchRuntime", "TorchRuntime", "TritonRuntime"]


def __getattr__(name: str) -> Any:
    """Keep the public runtime imports while avoiding optional eager imports."""

    if name == "TorchRuntime":
        from .torch import TorchRuntime

        return TorchRuntime
    if name == "MicroBatchRuntime":
        from .microbatch import MicroBatchRuntime

        return MicroBatchRuntime
    if name == "TritonRuntime":
        from .triton import TritonRuntime

        return TritonRuntime
    raise AttributeError(name)
