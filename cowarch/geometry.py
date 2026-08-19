from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np


DORSAL_KEYPOINTS = (
    "withers",
    "thoracic",
    "thoracolumbar",
    "lumbar",
    "sacrum",
)


def extract_topline(
    mask: np.ndarray,
    trim: float = 0.20,
    min_columns: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the upper silhouette from a binary mask.

    This is intentionally named a silhouette feature, not an anatomical spine
    extractor. A fixed trim cannot guarantee withers/sacrum localization.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2-D array")
    if not 0.0 <= trim < 0.45:
        raise ValueError("trim must be in [0, 0.45)")

    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0:
        return None

    order = np.argsort(xs, kind="stable")
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    x_unique, starts = np.unique(xs_sorted, return_index=True)
    y_top = np.minimum.reduceat(ys_sorted, starts)

    lo = int(len(x_unique) * trim)
    hi = int(len(x_unique) * (1.0 - trim))
    x_unique = x_unique[lo:hi]
    y_top = y_top[lo:hi]
    if len(x_unique) < min_columns or x_unique[-1] <= x_unique[0]:
        return None
    return x_unique.astype(float), y_top.astype(float)


def chord_values(px: np.ndarray, py: np.ndarray, endpoint_window: int = 5) -> np.ndarray:
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if px.ndim != 1 or py.ndim != 1 or len(px) != len(py) or len(px) < 3:
        raise ValueError("px and py must be equal-length 1-D arrays")
    if not np.all(np.diff(px) > 0):
        raise ValueError("px must be strictly increasing")
    k = max(1, min(endpoint_window, len(py) // 3))
    y0 = float(np.median(py[:k]))
    y1 = float(np.median(py[-k:]))
    return np.interp(px, [px[0], px[-1]], [y0, y1])


def normalized_sagitta(px: np.ndarray, py: np.ndarray, endpoint_window: int = 5) -> float:
    chord = chord_values(px, py, endpoint_window=endpoint_window)
    width = float(px[-1] - px[0])
    if width <= 0:
        raise ValueError("topline width must be positive")
    # Image y grows downward: an upward arch has y_top < y_chord.
    return float(np.max(chord - py) / width)


def menger_curvature(points: np.ndarray) -> float:
    """Return 1/R for three 2-D points; zero for a degenerate/straight triple."""
    p = np.asarray(points, dtype=float)
    if p.shape != (3, 2):
        raise ValueError("points must have shape (3, 2)")
    a = float(np.linalg.norm(p[1] - p[0]))
    b = float(np.linalg.norm(p[2] - p[1]))
    c = float(np.linalg.norm(p[2] - p[0]))
    denom = a * b * c
    if denom <= 1e-12:
        return 0.0
    first = p[1] - p[0]
    second = p[2] - p[0]
    twice_area = abs(float(first[0] * second[1] - first[1] * second[0]))
    return float(2.0 * twice_area / denom)


def auto_topline_features(mask: np.ndarray, trim: float = 0.20) -> dict[str, float]:
    """Experimental fixed-trim silhouette features.

    Names are prefixed with ``auto_`` to prevent accidental presentation as
    anatomically anchored back measurements.
    """
    extracted = extract_topline(mask, trim=trim)
    if extracted is None:
        return {
            "auto_sagitta": np.nan,
            "auto_chord_rmse": np.nan,
            "auto_circle_curvature_norm": np.nan,
        }
    px, py = extracted
    chord = chord_values(px, py)
    width = float(px[-1] - px[0])
    deviation = chord - py
    peak = int(np.argmax(deviation))
    points = np.array(
        [[px[0], chord[0]], [px[peak], py[peak]], [px[-1], chord[-1]]],
        dtype=float,
    )
    return {
        "auto_sagitta": float(np.max(deviation) / width),
        "auto_chord_rmse": float(np.sqrt(np.mean(np.square(deviation))) / width),
        "auto_circle_curvature_norm": float(menger_curvature(points) * width),
    }


def decode_keypoints(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = json.loads(value)

    if isinstance(value, dict):
        if not all(name in value for name in DORSAL_KEYPOINTS):
            return None
        points = [[value[name]["x"], value[name]["y"]] for name in DORSAL_KEYPOINTS]
    elif isinstance(value, Iterable):
        rows = list(value)
        if len(rows) != len(DORSAL_KEYPOINTS):
            return None
        if rows and isinstance(rows[0], dict):
            by_name = {row.get("name"): row for row in rows}
            if not all(name in by_name for name in DORSAL_KEYPOINTS):
                return None
            points = [[by_name[name]["x"], by_name[name]["y"]] for name in DORSAL_KEYPOINTS]
        else:
            points = rows
    else:
        return None

    arr = np.asarray(points, dtype=float)
    if arr.shape != (5, 2) or not np.all(np.isfinite(arr)):
        return None
    return arr


def keypoint_features(points: np.ndarray) -> dict[str, float]:
    """Compute orientation-invariant features from five ordered dorsal points."""
    p = np.asarray(points, dtype=float)
    if p.shape != (5, 2) or not np.all(np.isfinite(p)):
        raise ValueError("five finite 2-D dorsal points are required")

    baseline = p[-1] - p[0]
    length = float(np.linalg.norm(baseline))
    if length <= 1e-6:
        raise ValueError("withers and sacrum cannot coincide")
    unit = baseline / length
    relative = p - p[0]
    along = relative @ unit / length
    cross = unit[0] * relative[:, 1] - unit[1] * relative[:, 0]
    deviation = np.abs(cross) / length

    if np.any(np.diff(along) <= 0):
        raise ValueError("dorsal keypoints must progress monotonically from withers to sacrum")

    coeff = np.polyfit(along, deviation, deg=2)
    grid = np.linspace(0.0, 1.0, 101)
    quad_peak = float(max(0.0, np.max(np.polyval(coeff, grid))))

    v1 = p[2] - p[0]
    v2 = p[4] - p[2]
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-12:
        angle = 0.0
    else:
        cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cos_angle)))

    return {
        "kp_sagitta_norm": float(np.max(deviation)),
        "kp_mean_deviation_norm": float(np.mean(deviation[1:-1])),
        "kp_quad_peak_norm": quad_peak,
        "kp_back_deflection_deg": angle,
        "kp_menger_curvature_norm": float(menger_curvature(p[[0, 2, 4]]) * length),
        "kp_length_px": length,
    }
