from typing import TYPE_CHECKING, Any

from .bundle import ModelBundle, load_model_bundle

if TYPE_CHECKING:
    from .api import Deidentifier, DeidentificationResult

__all__ = ["Deidentifier", "DeidentificationResult", "ModelBundle", "load_model_bundle"]
__version__ = "0.3.0"


def __getattr__(name: str) -> Any:
    """Keep public inference imports compatible without loading PyTorch for CLI help."""

    if name in {"Deidentifier", "DeidentificationResult"}:
        from .api import Deidentifier, DeidentificationResult

        exports = {
            "Deidentifier": Deidentifier,
            "DeidentificationResult": DeidentificationResult,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
