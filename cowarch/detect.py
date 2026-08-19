"""Cow detection/segmentation wrappers.

The pretrained detector supplies *pre-annotation only*. Its boxes and masks are
inputs to human inspection, never posture ground truth.
"""
from __future__ import annotations

import math

import numpy as np

from .frames import require_cv2


def find_cow_class(names) -> int:
    """Resolve the ``cow`` class by name; never trust a hard-coded index."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    matches = [int(index) for index, name in items if str(name).strip().lower() == "cow"]
    if not matches:
        raise RuntimeError("The selected checkpoint has no class named 'cow'")
    return matches[0]


def load_detector(model_name: str):
    if str(model_name).lower() == "none":
        return None, None
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Ultralytics is required unless the detector is disabled") from exc
    model = YOLO(model_name)
    return model, find_cow_class(model.names)


def predict_cows(model, cow_class: int, frame: np.ndarray, confidence: float):
    """Return ``[(box_xyxy, score, mask_or_None), ...]`` for cows in one frame."""
    result = model.predict(frame, classes=[cow_class], conf=confidence, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    masks = None
    if result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy()
    return [(boxes[i], float(scores[i]), None if masks is None else masks[i]) for i in range(len(boxes))]


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    cv2 = require_cv2()
    height, width = shape
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5


def bbox_crop(
    frame: np.ndarray, box: np.ndarray, padding: float
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(math.floor(x1 - pad_x)))
    top = max(0, int(math.floor(y1 - pad_y)))
    right = min(width, int(math.ceil(x2 + pad_x)))
    bottom = min(height, int(math.ceil(y2 + pad_y)))
    return frame[top:bottom, left:right], (left, top, right, bottom)
