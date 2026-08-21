"""Core utilities for the cattle arched-back posture PoC.

The CLI scripts in ``scripts/`` are batch backends over these modules; the
notebooks in ``notebooks/`` drive the same functions interactively.
"""

from .geometry import (
    DORSAL_KEYPOINTS,
    LEGACY_DORSAL_KEYPOINTS,
    MEASUREMENT_KEYPOINTS,
    anchored_topline,
    anchored_topline_features,
    auto_topline_features,
    keypoint_features,
    measurement_geometry,
)
from .prepare import PrepareConfig, blank_record, detect_frame_cows, process_frame

__all__ = [
    "DORSAL_KEYPOINTS",
    "LEGACY_DORSAL_KEYPOINTS",
    "MEASUREMENT_KEYPOINTS",
    "anchored_topline",
    "anchored_topline_features",
    "auto_topline_features",
    "keypoint_features",
    "measurement_geometry",
    "PrepareConfig",
    "blank_record",
    "detect_frame_cows",
    "process_frame",
]
__version__ = "0.3.0"
