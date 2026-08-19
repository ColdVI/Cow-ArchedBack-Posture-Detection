"""Occlusion sensitivity for the frozen-embedding classifier.

Answers the question that decides whether a result is worth anything: is the
model responding to the animal's back, or to the barn behind it?

Grad-CAM needs gradients through a trained backbone. Here the backbone is frozen
and only a linear head is fitted, so occlusion is both simpler and more honest:
grey out a patch, re-embed, and see how much the prediction moves.
"""
from __future__ import annotations

import numpy as np

from .embeddings import extract_resnet18_embeddings


def occlusion_map(
    crop_bgr: np.ndarray,
    score_fn,
    grid: tuple[int, int] = (6, 8),
    fill: int = 128,
    batch_size: int = 32,
) -> np.ndarray:
    """Return a ``grid``-shaped array of ``baseline - occluded`` scores.

    ``score_fn`` maps a list of BGR crops to P(arched). High cell values mean the
    prediction depended on that patch.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("crop is empty")
    rows, columns = grid
    height, width = crop_bgr.shape[:2]
    step_y = max(1, height // rows)
    step_x = max(1, width // columns)

    variants = []
    for row in range(rows):
        for column in range(columns):
            occluded = crop_bgr.copy()
            y0, x0 = row * step_y, column * step_x
            y1 = height if row == rows - 1 else min(height, y0 + step_y)
            x1 = width if column == columns - 1 else min(width, x0 + step_x)
            occluded[y0:y1, x0:x1] = fill
            variants.append(occluded)

    baseline = float(score_fn([crop_bgr])[0])
    scores = np.asarray(score_fn(variants), dtype=float)
    return (baseline - scores).reshape(rows, columns)


def embedding_score_fn(pipeline, device: str = "auto", batch_size: int = 32):
    """Wrap a fitted sklearn pipeline into a ``list[BGR crop] -> P(arched)`` callable."""
    import tempfile
    from pathlib import Path

    import cv2

    def score(crops):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, crop in enumerate(crops):
                path = Path(directory) / f"{index:05d}.png"
                cv2.imwrite(str(path), crop)
                paths.append(path)
            embeddings = extract_resnet18_embeddings(paths, batch_size=batch_size, device=device)
        return pipeline.predict_proba(embeddings)[:, 1]

    return score


def upsample_map(sensitivity: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    import cv2

    height, width = shape
    return cv2.resize(sensitivity.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def topline_attention_ratio(sensitivity_map: np.ndarray, mask: np.ndarray, band: int = 40) -> float:
    """Fraction of positive sensitivity that falls in a band under the topline.

    Near 1 means the evidence sits on the back. Near 0 means it does not, and the
    result is about the background.
    """
    if mask is None:
        raise ValueError("a segmentation mask is required")
    height, width = sensitivity_map.shape[:2]
    band_mask = np.zeros((height, width), dtype=bool)
    for column in range(width):
        rows = np.flatnonzero(mask[:, column])
        if rows.size:
            top = int(rows.min())
            band_mask[top : min(height, top + band), column] = True
    positive = np.clip(sensitivity_map, 0, None)
    total = float(positive.sum())
    if total <= 1e-12:
        return float("nan")
    return float(positive[band_mask].sum() / total)
