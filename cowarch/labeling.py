"""In-notebook posture labeling with optional dorsal keypoint capture.

Design rules this module enforces, so they cannot be forgotten in a notebook:

* A model may reorder the queue; only a human writes a label.
* Ordering hints are refused outside the training split, so validation and test
  are always labeled blind and in random order.
* When keypoint capture is on, ``arched``/``normal`` requires all five points.
* The manifest is written atomically after every single decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import DORSAL_KEYPOINTS
from .io import as_bool, atomic_write_csv, read_manifest, resolve_data_path

POSTURE_LABELS = ("arched", "normal", "uncertain", "invalid")
LABEL_COLORS = {
    "arched": "#d1495b",
    "normal": "#2e86ab",
    "uncertain": "#f4a259",
    "invalid": "#6c757d",
}
EDITABLE_COLUMNS = ("label", "keypoints_json", "reviewed_by")


def _require_widgets():
    try:
        import ipywidgets
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ipywidgets is required for the notebook labeler") from exc
    return ipywidgets


def live_sagitta(points: np.ndarray) -> float | None:
    """Normalized sagitta from partially or fully clicked points."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 3:
        return None
    baseline = p[-1] - p[0]
    length = float(np.linalg.norm(baseline))
    if length <= 1e-6:
        return None
    unit = baseline / length
    relative = p - p[0]
    cross = unit[0] * relative[:, 1] - unit[1] * relative[:, 0]
    return float(np.max(np.abs(cross)) / length)


class PostureLabeler:
    """Button-and-click labeling UI over a prepared manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        reviewer: str,
        order: str = "random",
        seed: int = 42,
        collect_keypoints: bool = False,
        max_samples: int | None = None,
        priority: pd.Series | None = None,
        relabel: bool = False,
        figsize: tuple[float, float] = (9.0, 5.5),
    ) -> None:
        if order not in {"random", "sequential", "priority"}:
            raise ValueError("order must be random, sequential or priority")
        if order == "priority" and split != "train":
            raise ValueError(
                "priority ordering is a training-split tool. Validation and test "
                "must be labeled blind, in random order."
            )
        if not str(reviewer).strip():
            raise ValueError("reviewer must be set so annotations stay attributable")

        self.manifest_path = Path(manifest_path)
        self.frame = read_manifest(self.manifest_path)
        for column in EDITABLE_COLUMNS:
            if column not in self.frame.columns:
                self.frame[column] = ""
        self.frame[list(EDITABLE_COLUMNS)] = self.frame[list(EDITABLE_COLUMNS)].astype(object)

        self.split = split
        self.reviewer = str(reviewer).strip()
        self.collect_keypoints = bool(collect_keypoints)
        self.figsize = figsize

        accepted = as_bool(self.frame["accepted"]).to_numpy()
        in_split = self.frame["split"].astype(str).eq(split).to_numpy()
        unlabeled = self.frame["label"].astype(str).str.strip().eq("").to_numpy()
        candidates = np.flatnonzero(accepted & in_split & (unlabeled | relabel))
        if candidates.size == 0:
            raise ValueError(f"No accepted, unlabeled rows in split '{split}'")

        if order == "random":
            rng = np.random.default_rng(seed)
            candidates = candidates[rng.permutation(candidates.size)]
        elif order == "priority":
            if priority is None:
                raise ValueError("priority ordering needs a priority series")
            values = pd.Series(priority).reindex(self.frame.index).to_numpy(dtype=float)
            scores = values[candidates]
            scores = np.where(np.isfinite(scores), scores, -np.inf)
            candidates = candidates[np.argsort(-scores, kind="stable")]
        if max_samples:
            candidates = candidates[:max_samples]

        self.queue = list(candidates)
        self.position = 0
        self.points: list[tuple[float, float]] = []
        self.decisions = 0
        self._build_ui()

    # ---------------------------------------------------------------- UI ---
    def _build_ui(self) -> None:
        widgets = _require_widgets()
        self.output = widgets.Output()
        self.status = widgets.HTML()
        self.progress = widgets.IntProgress(min=0, max=len(self.queue), description="queue")

        buttons = []
        for label in POSTURE_LABELS:
            button = widgets.Button(description=label, layout=widgets.Layout(width="110px"))
            button.style.button_color = LABEL_COLORS[label]
            button.on_click(lambda _b, value=label: self._on_label(value))
            buttons.append(button)

        skip = widgets.Button(description="skip", layout=widgets.Layout(width="90px"))
        skip.on_click(lambda _b: self._advance())
        back = widgets.Button(description="back", layout=widgets.Layout(width="90px"))
        back.on_click(lambda _b: self._advance(-1))
        clear = widgets.Button(description="clear points", layout=widgets.Layout(width="120px"))
        clear.on_click(lambda _b: self._clear_points())
        undo = widgets.Button(description="undo point", layout=widgets.Layout(width="110px"))
        undo.on_click(lambda _b: self._undo_point())

        controls = [*buttons, skip, back]
        if self.collect_keypoints:
            controls += [undo, clear]
        self.ui = widgets.VBox(
            [widgets.HBox(controls), widgets.HBox([self.progress, self.status]), self.output]
        )

        with self.output:
            self.figure, self.axis = plt.subplots(figsize=self.figsize)
            plt.show()
        if self.collect_keypoints:
            self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def display(self):
        from IPython.display import display

        self._render()
        display(self.ui)
        return self.ui

    # ------------------------------------------------------------- state ---
    @property
    def current_index(self) -> int | None:
        if 0 <= self.position < len(self.queue):
            return int(self.queue[self.position])
        return None

    def _clear_points(self) -> None:
        self.points = []
        self._render()

    def _undo_point(self) -> None:
        if self.points:
            self.points.pop()
        self._render()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axis or event.xdata is None:
            return
        if len(self.points) >= len(DORSAL_KEYPOINTS):
            return
        self.points.append((float(event.xdata), float(event.ydata)))
        self._render()

    def _on_label(self, label: str) -> None:
        index = self.current_index
        if index is None:
            return
        needs_points = self.collect_keypoints and label in {"arched", "normal"}
        if needs_points and len(self.points) != len(DORSAL_KEYPOINTS):
            self.status.value = (
                f"<b style='color:#d1495b'>{len(self.points)}/5 keypoints. "
                "Click the remaining dorsal points before labeling.</b>"
            )
            return

        self.frame.at[index, "label"] = label
        self.frame.at[index, "reviewed_by"] = self.reviewer
        if self.collect_keypoints and self.points:
            self.frame.at[index, "keypoints_json"] = json.dumps(
                [[round(x, 2), round(y, 2)] for x, y in self.points]
            )
        atomic_write_csv(self.frame, self.manifest_path)
        self.decisions += 1
        self._advance()

    def _advance(self, step: int = 1) -> None:
        self.position = max(0, min(len(self.queue), self.position + step))
        self.points = []
        self._render()

    # ------------------------------------------------------------ render ---
    def _render(self) -> None:
        import cv2

        self.progress.value = min(self.position, len(self.queue))
        index = self.current_index
        self.axis.clear()

        if index is None:
            self.axis.text(0.5, 0.5, "queue finished", ha="center", va="center", fontsize=14)
            self.axis.axis("off")
            self.status.value = (
                f"<b>done</b> - {self.decisions} decisions written to {self.manifest_path.name}"
            )
            self.figure.canvas.draw_idle()
            return

        row = self.frame.loc[index]
        path = resolve_data_path(str(row["crop_path"]), self.manifest_path)
        image = cv2.imread(str(path))
        if image is None:
            self.axis.text(0.5, 0.5, f"unreadable crop:\n{path}", ha="center", va="center")
            self.axis.axis("off")
        else:
            self.axis.imshow(image[:, :, ::-1])
            self.axis.axis("off")

        if self.points:
            p = np.asarray(self.points, dtype=float)
            self.axis.plot(p[:, 0], p[:, 1], color="#f4a259", marker="o", markersize=6, linewidth=1.6)
            for order, (x, y) in enumerate(self.points):
                self.axis.annotate(
                    f"{order + 1}.{DORSAL_KEYPOINTS[order]}",
                    (x, y),
                    color="white",
                    fontsize=7,
                    xytext=(3, 6),
                    textcoords="offset points",
                )
            if len(self.points) == len(DORSAL_KEYPOINTS):
                self.axis.plot(
                    [p[0, 0], p[-1, 0]], [p[0, 1], p[-1, 1]],
                    color="white", linestyle="--", linewidth=1.2,
                )

        title = f"{row['sample_id']}   ({self.position + 1}/{len(self.queue)})   split={self.split}"
        self.axis.set_title(title, fontsize=10)
        self.figure.canvas.draw_idle()

        hint = ""
        if self.collect_keypoints:
            sagitta = live_sagitta(np.asarray(self.points)) if len(self.points) >= 3 else None
            target = DORSAL_KEYPOINTS[len(self.points)] if len(self.points) < 5 else "complete"
            hint = f" | points {len(self.points)}/5 -> next: <code>{target}</code>"
            if sagitta is not None:
                hint += f" | live S_n={sagitta:.3f} <i>(feature, not a label)</i>"
        self.status.value = f"reviewer <b>{self.reviewer}</b> | written {self.decisions}{hint}"


def label_summary(manifest_path: str | Path) -> pd.DataFrame:
    frame = read_manifest(manifest_path)
    accepted = as_bool(frame["accepted"])
    subset = frame.loc[accepted].copy()
    subset["label"] = subset["label"].astype(str).replace("", "unlabeled")
    return (
        pd.crosstab(subset["split"].replace("", "unassigned"), subset["label"])
        .reindex(columns=[*POSTURE_LABELS, "unlabeled"], fill_value=0)
    )
