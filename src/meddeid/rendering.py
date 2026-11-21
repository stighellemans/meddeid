"""Apply the shared CPU renderer to detections or structured values."""
from meddeid_core.rendering import (
    PseudonymizationScope,
    apply_spans,
    derive_date_offset,
    pseudonymize_name,
)

__all__ = ["PseudonymizationScope", "apply_spans", "derive_date_offset", "pseudonymize_name"]
