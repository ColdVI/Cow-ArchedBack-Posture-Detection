"""Camera-bias mitigations for absolute dorsal-arch measurement."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlumbLineCalibration:
    image_width: int
    image_height: int
    camera_matrix: tuple[tuple[float, float, float], ...]
    distortion: tuple[float, float, float, float, float]
    rms_line_error_before_px: float
    rms_line_error_after_px: float

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "PlumbLineCalibration":
        return cls(
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            camera_matrix=tuple(tuple(float(v) for v in row) for row in payload["camera_matrix"]),
            distortion=tuple(float(v) for v in payload["distortion"]),
            rms_line_error_before_px=float(payload["rms_line_error_before_px"]),
            rms_line_error_after_px=float(payload["rms_line_error_after_px"]),
        )


def _line_residuals(lines: list[np.ndarray]) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for points in lines:
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[-1]
        residuals.append(centered @ normal)
    return np.concatenate(residuals)


def _undistorted_lines(
    lines: list[np.ndarray], camera_matrix: np.ndarray, distortion: np.ndarray
) -> list[np.ndarray]:
    import cv2

    return [
        cv2.undistortPoints(
            points.reshape(-1, 1, 2).astype(np.float64),
            camera_matrix,
            distortion,
            P=camera_matrix,
        ).reshape(-1, 2)
        for points in lines
    ]


def fit_plumb_line_calibration(
    lines: Iterable[Iterable[Iterable[float]]],
    *,
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
) -> PlumbLineCalibration:
    """Estimate radial ``k1,k2`` by making known-straight image lines straight.

    This is a plumb-line calibration, so it does not claim metric camera
    calibration.  The optical centre is fixed to the image centre and focal
    length is supplied or set to the larger image dimension.  At least three
    lines spanning different image regions are required.
    """
    if image_width < 2 or image_height < 2:
        raise ValueError("image dimensions must be at least two pixels")
    normalized = [np.asarray(points, dtype=float) for points in lines]
    if len(normalized) < 3:
        raise ValueError("at least three plumb lines are required")
    for points in normalized:
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
            raise ValueError("each plumb line needs at least four 2-D points")
        if not np.all(np.isfinite(points)):
            raise ValueError("plumb-line points must be finite")

    focal = float(focal_length_px or max(image_width, image_height))
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal_length_px must be positive")
    camera_matrix = np.array(
        [[focal, 0.0, image_width / 2.0], [0.0, focal, image_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError("scipy is required for plumb-line calibration") from exc

    def objective(params: np.ndarray) -> np.ndarray:
        distortion = np.array([params[0], params[1], 0.0, 0.0, 0.0], dtype=float)
        residual = _line_residuals(_undistorted_lines(normalized, camera_matrix, distortion))
        # Mild regularization prevents an implausible high-order solution when
        # clicks cover only a small part of the frame.
        return np.concatenate((residual, np.asarray(params) * 0.1))

    result = least_squares(
        objective,
        x0=np.zeros(2, dtype=float),
        bounds=(np.array([-1.5, -1.5]), np.array([1.5, 1.5])),
        method="trf",
    )
    if not result.success:
        raise RuntimeError(f"plumb-line optimization failed: {result.message}")
    distortion = np.array([result.x[0], result.x[1], 0.0, 0.0, 0.0], dtype=float)
    before = _line_residuals(normalized)
    after = _line_residuals(_undistorted_lines(normalized, camera_matrix, distortion))
    return PlumbLineCalibration(
        image_width=image_width,
        image_height=image_height,
        camera_matrix=tuple(tuple(float(v) for v in row) for row in camera_matrix),
        distortion=tuple(float(v) for v in distortion),
        rms_line_error_before_px=float(np.sqrt(np.mean(np.square(before)))),
        rms_line_error_after_px=float(np.sqrt(np.mean(np.square(after)))),
    )


def save_calibration(calibration: PlumbLineCalibration, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(calibration.as_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def load_calibration(path: str | Path) -> PlumbLineCalibration:
    return PlumbLineCalibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def undistort_image(image: np.ndarray, calibration: PlumbLineCalibration) -> np.ndarray:
    """Undistort an image without changing its dimensions."""
    import cv2

    frame = np.asarray(image)
    if frame.shape[:2] != (calibration.image_height, calibration.image_width):
        raise ValueError(
            "image dimensions do not match the plumb-line calibration: "
            f"{frame.shape[1]}x{frame.shape[0]} vs "
            f"{calibration.image_width}x{calibration.image_height}"
        )
    return cv2.undistort(
        frame,
        np.asarray(calibration.camera_matrix, dtype=float),
        np.asarray(calibration.distortion, dtype=float),
        None,
        np.asarray(calibration.camera_matrix, dtype=float),
    )
