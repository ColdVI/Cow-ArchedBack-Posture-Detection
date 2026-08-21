from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np


MEASUREMENT_KEYPOINTS = (
    "withers",
    "sacrum",
    "head",
)

# The five-point geometry path is retained only for reproducibility of the
# original supervised PoC.  New measurement annotations use
# ``MEASUREMENT_KEYPOINTS``.
LEGACY_DORSAL_KEYPOINTS = (
    "withers",
    "thoracic",
    "thoracolumbar",
    "lumbar",
    "sacrum",
)

# Public compatibility name used by the annotation frontends.  In v3 it means
# the production three-point protocol, not the historical five-point curve.
DORSAL_KEYPOINTS = MEASUREMENT_KEYPOINTS


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
            "body_length_px": np.nan,
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
        "body_length_px": width,
    }


def _canonical_chord(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a left-to-right chord and its image-upward unit normal."""
    if first[0] < second[0]:
        left, right = first, second
    elif second[0] < first[0]:
        left, right = second, first
    else:
        raise ValueError("chord endpoints must have distinct horizontal positions")

    chord = right - left
    length = float(np.linalg.norm(chord))
    if length <= 1e-6:
        raise ValueError("chord endpoints cannot coincide")
    unit = chord / length
    # Image y increases downwards. Since ``unit`` points towards image-right,
    # this perpendicular always has a negative y component and therefore
    # points towards the dorsal (image-top) side of a lateral cow image.
    dorsal_normal = np.array([unit[1], -unit[0]], dtype=float)
    return left, right, unit, dorsal_normal, length


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered, reversal-invariant moving average."""
    if not isinstance(window, (int, np.integer)) or window < 1 or window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
    values = np.asarray(values, dtype=float)
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, np.full(window, 1.0 / window), mode="valid")


def _regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope, returning zero for a one-point/flat domain."""
    if len(x) < 2:
        return 0.0
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def anchored_topline(
    mask: np.ndarray,
    anchors: np.ndarray,
    *,
    smoothing_window: int = 5,
    samples: int = 101,
    min_columns: int = 5,
) -> np.ndarray:
    """Return a smoothed dense topline between anatomical anchors.

    ``anchors`` must contain ``(withers, sacrum)`` in crop coordinates. The
    upper silhouette is restricted to the horizontal and chord-projection
    interval covered by those anchors, resampled on a fixed anatomical grid,
    and smoothed with a centered moving average. The returned ``(samples, 2)``
    array is ordered anatomically from withers to sacrum, uses crop coordinates,
    and contains the exact anchors as its endpoints.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2-D array")

    anchor_points = np.asarray(anchors, dtype=float)
    if anchor_points.shape != (2, 2):
        raise ValueError("anchors must have shape (2, 2), ordered as (withers, sacrum)")
    if not np.all(np.isfinite(anchor_points)):
        raise ValueError("anchors must contain only finite coordinates")
    if not isinstance(samples, (int, np.integer)) or samples < 5:
        raise ValueError("samples must be an integer of at least 5")
    if not isinstance(min_columns, (int, np.integer)) or min_columns < 3:
        raise ValueError("min_columns must be an integer of at least 3")

    height, width = binary.shape
    if height == 0 or width == 0:
        raise ValueError("mask must not be empty")
    x = anchor_points[:, 0]
    y = anchor_points[:, 1]
    if np.any(x < 0) or np.any(x > width - 1) or np.any(y < 0) or np.any(y > height - 1):
        raise ValueError("anchors must lie within the mask bounds")
    if float(np.linalg.norm(anchor_points[1] - anchor_points[0])) <= 1e-6:
        raise ValueError("withers and sacrum anchors cannot coincide")

    withers, sacrum = anchor_points
    left, _, canonical_unit, dorsal_normal, chord_length = _canonical_chord(
        withers,
        sacrum,
    )

    ys, xs = np.where(binary.astype(bool))
    if xs.size == 0:
        raise ValueError("mask contains no foreground pixels")
    order = np.argsort(xs, kind="stable")
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    x_unique, starts = np.unique(xs_sorted, return_index=True)
    y_top = np.minimum.reduceat(ys_sorted, starts)

    x_low = float(np.min(anchor_points[:, 0]))
    x_high = float(np.max(anchor_points[:, 0]))
    in_horizontal_span = (x_unique >= x_low) & (x_unique <= x_high)
    contour = np.column_stack(
        (x_unique[in_horizontal_span], y_top[in_horizontal_span])
    ).astype(float)
    if len(contour) < min_columns:
        raise ValueError("mask has insufficient topline support between anchors")

    # A horizontal-span check excludes head/tail components outside the
    # anatomical endpoints. The projection check handles sloped chords.
    canonical_position = (contour - left) @ canonical_unit / chord_length
    in_chord_span = (canonical_position >= -1e-9) & (canonical_position <= 1.0 + 1e-9)
    contour = contour[in_chord_span]
    if len(contour) < min_columns:
        raise ValueError("mask has insufficient topline support along the anchor chord")

    anatomical_unit = (sacrum - withers) / chord_length
    anatomical_position = (contour - withers) @ anatomical_unit / chord_length
    signed_distance = (contour - left) @ dorsal_normal
    order = np.argsort(anatomical_position, kind="stable")
    anatomical_position = anatomical_position[order]
    signed_distance = signed_distance[order]

    # The clicked anchors define the chord endpoints. Including their exact
    # zero deviations avoids extrapolating a nearby integer mask column to an
    # annotation that can have subpixel coordinates.
    anatomical_position = np.concatenate(([0.0], anatomical_position, [1.0]))
    signed_distance = np.concatenate(([0.0], signed_distance, [0.0]))
    unique_position, inverse = np.unique(anatomical_position, return_inverse=True)
    distance_sum = np.zeros(len(unique_position), dtype=float)
    distance_count = np.zeros(len(unique_position), dtype=float)
    np.add.at(distance_sum, inverse, signed_distance)
    np.add.at(distance_count, inverse, 1.0)
    unique_distance = distance_sum / distance_count

    grid = np.linspace(0.0, 1.0, int(samples))
    dense_distance = np.interp(grid, unique_position, unique_distance)
    smooth_distance = _moving_average(dense_distance, smoothing_window)
    # Preserve the anchor/chord identity after smoothing.
    smooth_distance[0] = 0.0
    smooth_distance[-1] = 0.0
    dense_chord = withers + grid[:, np.newaxis] * (sacrum - withers)
    return dense_chord + smooth_distance[:, np.newaxis] * dorsal_normal


def anchored_topline_features(
    mask: np.ndarray,
    anchors: np.ndarray,
    *,
    smoothing_window: int = 5,
    samples: int = 101,
    min_columns: int = 5,
) -> dict[str, float]:
    """Return normalized features for the anchor-bounded dense topline.

    ``anchors`` are ordered as ``(withers, sacrum)``. Signed sagitta is
    positive towards image-top, peak position runs anatomically from zero at
    the withers to one at the sacrum, and slope features follow that same
    anatomical direction. The returned dictionary contains numeric scalars
    only; use :func:`anchored_topline` when sampled plot coordinates are needed.
    """
    anchor_points = np.asarray(anchors, dtype=float)
    topline = anchored_topline(
        mask,
        anchor_points,
        smoothing_window=smoothing_window,
        samples=samples,
        min_columns=min_columns,
    )
    withers, sacrum = anchor_points
    _, _, _, dorsal_normal, chord_length = _canonical_chord(withers, sacrum)
    grid = np.linspace(0.0, 1.0, int(samples))
    dense_chord = withers + grid[:, np.newaxis] * (sacrum - withers)
    normalized = ((topline - dense_chord) @ dorsal_normal) / chord_length

    magnitude = np.abs(normalized)
    peak_candidates = np.flatnonzero(
        np.isclose(magnitude, float(np.max(magnitude)), rtol=1e-9, atol=1e-12)
    )
    peak = int(peak_candidates[np.argmin(np.abs(grid[peak_candidates] - 0.5))])
    signed_sagitta = float(normalized[peak])
    return {
        "anchored_sagitta_signed_norm": signed_sagitta,
        "anchored_sagitta_abs_norm": float(abs(signed_sagitta)),
        "anchored_chord_rmse_norm": float(np.sqrt(np.mean(np.square(normalized)))),
        "anchored_peak_position": float(grid[peak]),
        "anchored_cranial_slope": _regression_slope(grid[: peak + 1], normalized[: peak + 1]),
        "anchored_caudal_slope": _regression_slope(grid[peak:], normalized[peak:]),
    }


def decode_keypoints(value: Any, *, allow_legacy: bool = False) -> np.ndarray | None:
    """Decode the production three-point protocol.

    Five-point annotations are accepted only when ``allow_legacy=True``.  This
    makes legacy supervised experiments explicit while keeping new measurement
    code on ``withers, sacrum, head``.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        value = json.loads(value)

    names = LEGACY_DORSAL_KEYPOINTS if allow_legacy else MEASUREMENT_KEYPOINTS
    if isinstance(value, dict):
        if not all(name in value for name in names):
            return None
        points = [[value[name]["x"], value[name]["y"]] for name in names]
    elif isinstance(value, Iterable):
        rows = list(value)
        if len(rows) != len(names):
            return None
        if rows and isinstance(rows[0], dict):
            by_name = {row.get("name"): row for row in rows}
            if not all(name in by_name for name in names):
                return None
            points = [[by_name[name]["x"], by_name[name]["y"]] for name in names]
        else:
            points = rows
    else:
        return None

    arr = np.asarray(points, dtype=float)
    if arr.shape != (len(names), 2) or not np.all(np.isfinite(arr)):
        return None
    return arr


def measurement_geometry(
    points: np.ndarray,
    *,
    head_drop_max_norm: float,
) -> tuple[np.ndarray, float]:
    """Return withers/sacrum anchors and normalized head drop.

    Image ``y`` grows downwards, so positive head drop means that the head is
    below the withers.  Frames beyond the configured limit are rejected before
    sagitta aggregation.
    """
    p = np.asarray(points, dtype=float)
    if p.shape != (3, 2) or not np.all(np.isfinite(p)):
        raise ValueError("three finite points ordered as withers, sacrum, head are required")
    if not np.isfinite(head_drop_max_norm) or head_drop_max_norm < 0:
        raise ValueError("head_drop_max_norm must be a finite non-negative value")
    anchors = p[:2]
    chord_length = float(np.linalg.norm(anchors[1] - anchors[0]))
    if chord_length <= 1e-6:
        raise ValueError("withers and sacrum cannot coincide")
    head_drop = float((p[2, 1] - p[0, 1]) / chord_length)
    if head_drop > head_drop_max_norm:
        raise ValueError("head_down")
    return anchors, head_drop


def keypoint_features(points: np.ndarray) -> dict[str, float]:
    """Compute direction-invariant features from five ordered dorsal points.

    Signed deviation is positive towards image-top (an upward arch) and is
    independent of whether the cow faces image-left or image-right.
    ``kp_arch_height_*`` values are pixel distances; deviation fields carrying
    a ``_norm`` suffix are divided by Euclidean withers-sacrum chord length.
    """
    p = np.asarray(points, dtype=float)
    if p.shape != (5, 2) or not np.all(np.isfinite(p)):
        raise ValueError(
            "five finite legacy dorsal points are required; this path is not the v3 protocol"
        )

    if float(np.linalg.norm(p[-1] - p[0])) <= 1e-6:
        raise ValueError("withers and sacrum cannot coincide")
    left, _, unit, dorsal_normal, length = _canonical_chord(p[0], p[-1])
    relative = p - left
    along = relative @ unit / length
    signed_distance = relative @ dorsal_normal
    signed_deviation = signed_distance / length
    deviation = np.abs(signed_deviation)

    progress = np.diff(along)
    if not (np.all(progress > 0) or np.all(progress < 0)):
        raise ValueError("dorsal keypoints must progress monotonically from withers to sacrum")

    internal_signed = signed_deviation[1:-1]
    signed_peak_index = int(np.argmax(np.abs(internal_signed)))
    signed_arch_height = float(signed_distance[1:-1][signed_peak_index])
    if abs(signed_arch_height) <= 1e-12:
        signed_arch_height = 0.0

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
        "kp_arch_height_signed": signed_arch_height,
        "kp_arch_height_abs": float(abs(signed_arch_height)),
        "kp_mean_signed_deviation_norm": float(np.mean(internal_signed)),
        "kp_mean_abs_deviation_norm": float(np.mean(np.abs(internal_signed))),
        "kp_quad_peak_norm": quad_peak,
        "kp_back_deflection_deg": angle,
        "kp_menger_curvature_norm": float(menger_curvature(p[[0, 2, 4]]) * length),
        "kp_length_px": length,
    }
