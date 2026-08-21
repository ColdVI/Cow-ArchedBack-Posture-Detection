"""Single production interface for the three-point keypoint model."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .geometry import MEASUREMENT_KEYPOINTS


def _normalize_prediction(value: Any) -> dict[str, dict[str, float]] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        rows = []
        for name in MEASUREMENT_KEYPOINTS:
            point = value.get(name)
            if point is None:
                return None
            if isinstance(point, dict):
                rows.append((point.get("x"), point.get("y"), point.get("confidence", 1.0)))
            else:
                values = list(point)
                rows.append((*values[:2], values[2] if len(values) > 2 else 1.0))
    else:
        array = np.asarray(value, dtype=float)
        if array.shape == (3, 2):
            array = np.column_stack((array, np.ones(3)))
        if array.shape != (3, 3):
            return None
        rows = array.tolist()

    array = np.asarray(rows, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        return None
    if np.any((array[:, 2] < 0) | (array[:, 2] > 1)):
        return None
    return {
        name: {
            "x": float(array[index, 0]),
            "y": float(array[index, 1]),
            "confidence": float(array[index, 2]),
        }
        for index, name in enumerate(MEASUREMENT_KEYPOINTS)
    }


def predict_anchors(
    crop: np.ndarray,
    predictor: Callable[[np.ndarray], Any] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Predict ``withers, sacrum, head`` and per-point confidence.

    The callable boundary deliberately hides the selected framework.  A
    production caller supplies an initialized predictor (for example
    :class:`UltralyticsAnchorPredictor`); tests and notebooks can supply a
    lightweight callable.  Failure to produce exactly three finite points is
    represented as ``None``.
    """
    image = np.asarray(crop)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.size == 0:
        raise ValueError("crop must be a non-empty HxWx3/4 image")
    if predictor is None:
        raise RuntimeError(
            "No anchor predictor configured. Supply the selected T1 model wrapper."
        )
    return _normalize_prediction(predictor(image))


def anchor_array(
    prediction: dict[str, dict[str, float]],
    *,
    min_confidence: float,
) -> np.ndarray | None:
    """Convert a prediction to the ordered three-point array after confidence gating."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between zero and one")
    normalized = _normalize_prediction(prediction)
    if normalized is None:
        return None
    if any(normalized[name]["confidence"] < min_confidence for name in MEASUREMENT_KEYPOINTS):
        return None
    return np.asarray(
        [[normalized[name]["x"], normalized[name]["y"]] for name in MEASUREMENT_KEYPOINTS],
        dtype=float,
    )


class UltralyticsAnchorPredictor:
    """Lazy YOLO-pose adapter with fixed keypoint order.

    The checkpoint must have ``kpt_shape[0] == 3`` and must have been trained
    in ``withers, sacrum, head`` order.  COCO human-pose weights are rejected.
    """

    def __init__(self, checkpoint: str, *, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional heavy dependency
            raise RuntimeError("ultralytics is required for the YOLO pose backend") from exc
        self.model = YOLO(checkpoint)
        self.device = device
        shape = getattr(getattr(self.model, "model", None), "yaml", {}).get("kpt_shape")
        if not shape or int(shape[0]) != 3:
            raise ValueError(
                "Anchor checkpoint must contain exactly 3 keypoints ordered as "
                "withers, sacrum, head; COCO human-pose weights are not valid."
            )

    def __call__(self, crop: np.ndarray) -> np.ndarray | None:
        results = self.model.predict(crop, verbose=False, device=self.device)
        if not results or results[0].keypoints is None or results[0].keypoints.xy is None:
            return None
        xy = results[0].keypoints.xy.detach().cpu().numpy()
        if xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1:] != (3, 2):
            return None
        box_conf = results[0].boxes.conf.detach().cpu().numpy()
        selected = int(np.argmax(box_conf)) if len(box_conf) else 0
        confidence = results[0].keypoints.conf
        if confidence is None:
            point_conf = np.ones(3, dtype=float)
        else:
            point_conf = confidence.detach().cpu().numpy()[selected]
        return np.column_stack((xy[selected], point_conf))


def evaluate_pckh(
    predictions: pd.DataFrame,
    *,
    threshold: float = 0.2,
    required_anchor_pckh: float = 0.85,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate the three production points with explicit head-size scaling.

    Input is long-form with one row per sample/keypoint and columns
    ``sample_id,keypoint,pred_x,pred_y,true_x,true_y,head_scale_px``.
    """
    required = {
        "sample_id", "keypoint", "pred_x", "pred_y", "true_x", "true_y", "head_scale_px"
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"PCKh table is missing columns: {missing}")
    if not 0 < threshold <= 1 or not 0 < required_anchor_pckh <= 1:
        raise ValueError("PCKh thresholds must be in (0, 1]")
    frame = predictions.copy()
    frame["keypoint"] = frame["keypoint"].astype(str).str.strip()
    invalid = sorted(set(frame["keypoint"]) - set(MEASUREMENT_KEYPOINTS))
    if invalid:
        raise ValueError(f"Unsupported keypoints in PCKh table: {invalid}")
    counts = frame.groupby("sample_id")["keypoint"].agg(lambda values: set(values))
    expected = set(MEASUREMENT_KEYPOINTS)
    if any(value != expected for value in counts):
        raise ValueError("Every sample must contain withers, sacrum and head exactly once")
    if frame.duplicated(["sample_id", "keypoint"]).any():
        raise ValueError("PCKh table repeats a sample/keypoint pair")
    numeric_columns = ["pred_x", "pred_y", "true_x", "true_y", "head_scale_px"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all()
        or (frame["head_scale_px"] <= 0).any()
    ):
        raise ValueError("PCKh coordinates must be finite and head_scale_px positive")
    frame["normalized_error"] = np.hypot(
        frame["pred_x"] - frame["true_x"], frame["pred_y"] - frame["true_y"]
    ) / frame["head_scale_px"]
    frame["correct_at_threshold"] = frame["normalized_error"] <= threshold
    by_point = (
        frame.groupby("keypoint", sort=False)["correct_at_threshold"]
        .mean()
        .reindex(MEASUREMENT_KEYPOINTS)
    )
    summary = {
        "n_samples": int(frame["sample_id"].nunique()),
        "threshold": threshold,
        "pckh_overall": float(frame["correct_at_threshold"].mean()),
        "pckh_withers": float(by_point["withers"]),
        "pckh_sacrum": float(by_point["sacrum"]),
        "pckh_head": float(by_point["head"]),
        "required_anchor_pckh": required_anchor_pckh,
        "anchor_gate_pass": bool(
            by_point["withers"] >= required_anchor_pckh
            and by_point["sacrum"] >= required_anchor_pckh
        ),
    }
    return summary, frame
