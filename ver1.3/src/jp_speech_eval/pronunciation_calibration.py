"""Future human-validated pronunciation calibration contract.

This module intentionally contains no score mapping.  It gives experiments a
stable provenance payload while ProductScore v2 remains the production path
and ProductScore v3 remains candidate-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PronunciationCalibration:
    """Frozen metadata for a future, human-validated mapping.

    ``mapping`` must stay ``None`` until a pre-registered held-out validation
    has been completed.  This type deliberately provides no mapping method.
    """

    model_version: str = "unconfigured"
    ssl_model: str = "microsoft/wavlm-large"
    layer: str = "layer12"
    aggregation: str = "median"
    normalizer: Optional[str] = None
    mapping: None = None
    mapping_version: Optional[str] = None
    human_dataset_version: str = "pronunciation_listener_manifest_v1"
    valid_target_scope: str = "same_target_multi_reference_only"
    calibration_confidence: Optional[float] = None
    production_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_pronunciation_calibration() -> PronunciationCalibration:
    """Return the disabled, non-mapping default used by candidate telemetry."""
    return PronunciationCalibration()
