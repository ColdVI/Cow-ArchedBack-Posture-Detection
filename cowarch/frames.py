"""Frame sourcing helpers shared by the CLI backend and the inspection notebooks."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterator

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("OpenCV is required. Install requirements.txt") from exc
    return cv2


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def iter_frames(path: Path, kind: str, target_fps: float) -> Iterator[tuple[int, np.ndarray, str]]:
    """Yield ``(frame_index, BGR frame, display name)`` for an image, directory or video."""
    cv2 = require_cv2()
    path = Path(path)
    if kind in {"image", "direct"} and path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        yield 0, image, path.name
        return
    if kind == "image_dir" or path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        for index, image_path in enumerate(files):
            image = cv2.imread(str(image_path))
            if image is not None:
                yield index, image, image_path.name
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25.0
    step = max(1, int(round(source_fps / target_fps)))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % step == 0:
            yield frame_index, frame, path.name
        frame_index += 1
    capture.release()


def probe_source(path: Path, kind: str) -> dict:
    """Cheap metadata for the dataset browser: never decodes a whole video."""
    cv2 = require_cv2()
    path = Path(path)
    info = {"exists": path.exists(), "kind": kind, "path": str(path)}
    if not path.exists():
        return info
    if kind == "image_dir" or path.is_dir():
        files = [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        info.update({"n_images": len(files), "duration_s": float("nan"), "fps": float("nan")})
        return info
    if path.suffix.lower() in IMAGE_SUFFIXES:
        info.update({"n_images": 1, "duration_s": float("nan"), "fps": float("nan")})
        return info
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        info["error"] = "unreadable"
        return info
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    n_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    info.update(
        {
            "fps": fps if math.isfinite(fps) and fps > 0 else float("nan"),
            "n_frames": int(n_frames) if math.isfinite(n_frames) else 0,
            "duration_s": float(n_frames / fps) if fps > 0 and math.isfinite(n_frames) else float("nan"),
            "width": width,
            "height": height,
        }
    )
    return info


def sample_frames(path: Path, kind: str, target_fps: float, max_samples: int) -> list[tuple[int, np.ndarray, str]]:
    """Take at most ``max_samples`` frames. Used by preview-first notebook cells."""
    out: list[tuple[int, np.ndarray, str]] = []
    for item in iter_frames(path, kind, target_fps):
        out.append(item)
        if max_samples and len(out) >= max_samples:
            break
    return out


def dhash(image: np.ndarray) -> int:
    cv2 = require_cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())
