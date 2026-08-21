"""Inspection visuals for the notebook-first workflow.

Every transformation the pipeline performs has a panel here. The point is that
no step is accepted because a script printed "ok" - it is accepted because the
overlay looked right.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .geometry import (
    DORSAL_KEYPOINTS,
    LEGACY_DORSAL_KEYPOINTS,
    chord_values,
    extract_topline,
)

ARCHED_COLOR = "#d1495b"
NORMAL_COLOR = "#2e86ab"
ACCENT = "#f4a259"


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        return None
    if image.ndim == 2:
        return image
    return image[:, :, ::-1]


def _show(ax, image, title: str) -> None:
    if image is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", fontsize=9, color="grey")
    elif image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(bgr_to_rgb(image))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def panel_detections(ax, frame, detections, box=None) -> None:
    _show(ax, frame, f"2. detections (n={len(detections)})")
    if frame is None:
        return
    for index, (candidate_box, score, _) in enumerate(detections):
        x1, y1, x2, y2 = [float(v) for v in candidate_box]
        chosen = box is not None and np.allclose(candidate_box, box)
        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=ACCENT if chosen else "white",
                linewidth=2.0 if chosen else 1.0,
                linestyle="-" if chosen else "--",
            )
        )
        ax.text(x1, max(y1 - 4, 8), f"[{index}] cow {score:.2f}", color=ACCENT, fontsize=8)


def panel_mask(ax, crop, mask) -> None:
    if crop is None or mask is None:
        _show(ax, crop, "4. mask overlay (no mask)")
        return
    overlay = bgr_to_rgb(crop).astype(np.float32) / 255.0
    tint = np.zeros_like(overlay)
    tint[:, :, 1] = 1.0
    alpha = (mask.astype(np.float32) * 0.45)[:, :, None]
    ax.imshow(np.clip(overlay * (1 - alpha) + tint * alpha, 0, 1))
    ax.set_title(f"4. mask overlay (cover={mask.mean():.2f})", fontsize=9)
    ax.axis("off")


def panel_topline(ax, crop, mask, trim: float = 0.20) -> dict:
    """Draw the trimmed silhouette topline. Returns what was drawable."""
    _show(ax, crop, "5. silhouette topline")
    info = {"available": False}
    if mask is None:
        ax.set_title("5. topline (no mask)", fontsize=9)
        return info
    extracted = extract_topline(mask, trim=trim)
    if extracted is None:
        ax.set_title("5. topline (degenerate mask)", fontsize=9)
        return info
    px, py = extracted
    chord = chord_values(px, py)
    ax.plot(px, py, color=ACCENT, linewidth=2.0, label="topline")
    ax.plot(px, chord, color="white", linewidth=1.2, linestyle="--", label="chord")
    peak = int(np.argmax(chord - py))
    ax.plot([px[peak], px[peak]], [py[peak], chord[peak]], color=ARCHED_COLOR, linewidth=2.0)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.4)
    ax.set_title(f"5. topline (trim={trim:.0%} each end)", fontsize=9)
    info.update({"available": True, "px": px, "py": py, "chord": chord, "peak": peak})
    return info


def panel_keypoint_geometry(ax, crop, points) -> dict:
    """Render the production three-point protocol or legacy five-point curve."""
    _show(ax, crop, "6. dorsal keypoint geometry")
    if points is None:
        ax.set_title("6. keypoint geometry (not annotated)", fontsize=9)
        return {"available": False}

    p = np.asarray(points, dtype=float)
    if p.shape == (3, 2):
        baseline = p[1] - p[0]
        length = float(np.linalg.norm(baseline))
        if length <= 1e-6:
            ax.set_title("6. keypoint geometry (degenerate)", fontsize=9)
            return {"available": False}
        ax.plot(
            [p[0, 0], p[1, 0]], [p[0, 1], p[1, 1]],
            color="white", linewidth=1.4, linestyle="--",
        )
        ax.scatter(p[:, 0], p[:, 1], color=ACCENT, s=28)
        for index, name in enumerate(DORSAL_KEYPOINTS):
            ax.annotate(
                f"{index + 1}.{name}", p[index], fontsize=6.5, color="white",
                xytext=(2, 6), textcoords="offset points",
            )
        head_drop = float((p[2, 1] - p[0, 1]) / length)
        ax.set_title(f"6. 3-point protocol (head drop={head_drop:.3f})", fontsize=9)
        return {"available": True, "head_drop_norm": head_drop, "length_px": length}
    if p.shape != (5, 2):
        ax.set_title("6. keypoint geometry (invalid point count)", fontsize=9)
        return {"available": False}

    baseline = p[-1] - p[0]
    length = float(np.linalg.norm(baseline))
    if length <= 1e-6:
        ax.set_title("6. keypoint geometry (degenerate)", fontsize=9)
        return {"available": False}
    unit = baseline / length
    relative = p - p[0]
    cross = unit[0] * relative[:, 1] - unit[1] * relative[:, 0]
    deviation = np.abs(cross) / length
    peak = int(np.argmax(deviation))
    foot = p[0] + float(relative[peak] @ unit) * unit

    ax.plot([p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]], color="white", linewidth=1.4, linestyle="--")
    ax.plot(p[:, 0], p[:, 1], color=ACCENT, linewidth=1.6, marker="o", markersize=5)
    ax.plot([p[peak, 0], foot[0]], [p[peak, 1], foot[1]], color=ARCHED_COLOR, linewidth=2.4)
    for index, name in enumerate(LEGACY_DORSAL_KEYPOINTS):
        ax.annotate(f"{index + 1}.{name}", p[index], fontsize=6.5, color="white",
                    xytext=(2, 6), textcoords="offset points")
    sagitta = float(np.max(deviation))
    ax.set_title(f"6. keypoint geometry (S_n = {sagitta:.3f})", fontsize=9)
    return {"available": True, "sagitta": sagitta, "peak_index": peak, "length_px": length}


def inspect_frame(outcome, keypoints=None, trim: float = 0.20, figsize=(15, 8)):
    """The six-panel view: raw -> detection -> crop -> mask -> topline -> geometry."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    _show(axes[0], outcome.frame, "1. raw frame")
    panel_detections(axes[1], outcome.frame, outcome.detections, outcome.box)
    _show(axes[2], outcome.crop, "3. cow crop")
    panel_mask(axes[3], outcome.crop, outcome.mask)
    panel_topline(axes[4], outcome.crop, outcome.mask, trim=trim)
    panel_keypoint_geometry(axes[5], outcome.crop, keypoints)

    status = "ACCEPTED" if outcome.accepted else f"REJECTED: {outcome.reject_reason}"
    color = "#2e7d32" if outcome.accepted else ARCHED_COLOR
    fig.suptitle(f"{outcome.record.get('sample_id', '?')}   [{status}]", color=color, fontsize=11)
    fig.tight_layout()
    return fig


def thumbnail_grid(images, titles=None, columns: int = 5, size: float = 2.4, colors=None):
    images = list(images)
    if not images:
        raise ValueError("nothing to show")
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * size, rows * size * 0.85))
    axes = np.atleast_1d(axes).ravel()
    for index, ax in enumerate(axes):
        if index >= len(images):
            ax.axis("off")
            continue
        image = images[index]
        if isinstance(image, (str, Path)):
            import cv2

            image = cv2.imread(str(image))
        _show(ax, image, "" if titles is None else str(titles[index]))
        if colors is not None and colors[index]:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(colors[index])
                spine.set_linewidth(3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("on")
    fig.tight_layout()
    return fig
