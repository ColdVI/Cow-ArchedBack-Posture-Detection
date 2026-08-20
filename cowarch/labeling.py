"""Two-pass, in-notebook annotation for posture and dorsal geometry.

Pass A deliberately shows only the crop and posture controls. Pass B runs only
after posture is complete and stores geometry without touching the posture
label. Both passes write the manifest atomically after each saved annotation.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .annotations import (
    ANNOTATION_COLUMNS,
    LABEL_COLORS,
    POSTURE_LABELS,
    ensure_annotation_columns,
)
from .geometry import DORSAL_KEYPOINTS, keypoint_features
from .io import as_bool, atomic_write_csv, read_manifest, resolve_data_path

# Kept as an alias for callers which imported the old module-level constant.
EDITABLE_COLUMNS = ANNOTATION_COLUMNS
GEOMETRY_MODES = {
    "keypoints": DORSAL_KEYPOINTS,
    "anchors": ("withers", "sacrum"),
}


def _require_widgets():
    try:
        import ipywidgets
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ipywidgets is required for the notebook labeler") from exc
    return ipywidgets


def live_sagitta(points: np.ndarray) -> float | None:
    """Compatibility helper for old callers; the annotation UIs do not call it."""
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


def _load_crop(row: pd.Series, manifest_path: Path) -> tuple[np.ndarray | None, Path]:
    import cv2

    path = resolve_data_path(str(row["crop_path"]), manifest_path)
    return cv2.imread(str(path)), path


class PostureLabeler:
    """Pass A: blind posture decisions with no geometry or score display."""

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
        if split != "train" and order != "random":
            raise ValueError(
                "Validation and test posture labels must be collected blind, in random order; "
                "priority ordering is a training-split tool."
            )
        if not str(reviewer).strip():
            raise ValueError("reviewer must be set so annotations stay attributable")
        if collect_keypoints:
            warnings.warn(
                "PostureLabeler no longer captures keypoints. Use "
                "DorsalGeometryLabeler after Pass A; collect_keypoints is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.manifest_path = Path(manifest_path)
        self.frame = ensure_annotation_columns(read_manifest(self.manifest_path))
        self.split = str(split)
        self.reviewer = str(reviewer).strip()
        # The argument remains readable for compatibility, but Pass A always
        # keeps geometry capture off.
        self.collect_keypoints = False
        self.figsize = figsize

        accepted = as_bool(self.frame["accepted"]).to_numpy()
        in_split = self.frame["split"].astype(str).eq(self.split).to_numpy()
        unlabeled = self.frame["label"].astype(str).str.strip().eq("").to_numpy()
        candidates = np.flatnonzero(accepted & in_split & (unlabeled | relabel))
        if candidates.size == 0:
            raise ValueError(f"No accepted, unlabeled rows in split '{self.split}'")

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
        if max_samples is not None:
            candidates = candidates[:max_samples]

        self.queue = list(candidates)
        self.position = 0
        self.decisions = 0
        self._build_ui()

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
        self.ui = widgets.VBox(
            [
                widgets.HBox([*buttons, skip, back]),
                widgets.HBox([self.progress, self.status]),
                self.output,
            ]
        )

        with self.output:
            self.figure, self.axis = plt.subplots(figsize=self.figsize)
            plt.show()

    def display(self):
        from IPython.display import display

        self._render()
        display(self.ui)
        return self.ui

    @property
    def current_index(self) -> int | None:
        if 0 <= self.position < len(self.queue):
            return int(self.queue[self.position])
        return None

    def _on_label(self, label: str) -> None:
        if label not in POSTURE_LABELS:
            raise ValueError(f"unknown posture label: {label}")
        index = self.current_index
        if index is None:
            return

        self.frame.at[index, "label"] = label
        self.frame.at[index, "posture_reviewed_by"] = self.reviewer
        # Preserve compatibility for consumers which still read reviewed_by.
        self.frame.at[index, "reviewed_by"] = self.reviewer
        self.frame.at[index, "annotation_pass"] = "posture"
        atomic_write_csv(self.frame, self.manifest_path)
        self.decisions += 1
        self._advance()

    def _advance(self, step: int = 1) -> None:
        self.position = max(0, min(len(self.queue), self.position + step))
        self._render()

    def _render(self) -> None:
        self.progress.value = min(self.position, len(self.queue))
        index = self.current_index
        self.axis.clear()

        if index is None:
            self.axis.text(0.5, 0.5, "queue finished", ha="center", va="center", fontsize=14)
            self.axis.axis("off")
            self.status.value = (
                f"<b>done</b> - {self.decisions} decisions written to "
                f"{self.manifest_path.name}"
            )
            self.figure.canvas.draw_idle()
            return

        row = self.frame.loc[index]
        image, _path = _load_crop(row, self.manifest_path)
        if image is None:
            self.axis.text(0.5, 0.5, "unreadable crop", ha="center", va="center")
            self.axis.axis("off")
        else:
            self.axis.imshow(image[:, :, ::-1])
            self.axis.axis("off")

        # Pass A intentionally withholds identifiers, split metadata, source
        # fields, predictions, scores, and every geometry overlay.
        self.axis.set_title(f"Pass A   ({self.position + 1}/{len(self.queue)})", fontsize=10)
        self.figure.canvas.draw_idle()
        self.status.value = f"reviewer <b>{self.reviewer}</b> | written {self.decisions}"


class DorsalGeometryLabeler:
    """Pass B: capture five dorsal points or two anchors after posture labeling.

    This class has no posture controls and never assigns to ``label`` or
    ``posture_reviewed_by``. Its queue contains only accepted rows whose Pass A
    label is ``arched`` or ``normal``.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        reviewer: str,
        mode: str = "keypoints",
        order: str = "random",
        seed: int = 42,
        max_samples: int | None = None,
        relabel: bool = False,
        figsize: tuple[float, float] = (9.0, 5.5),
    ) -> None:
        if mode not in GEOMETRY_MODES:
            raise ValueError("mode must be 'keypoints' or 'anchors'")
        if order not in {"random", "sequential"}:
            raise ValueError("geometry order must be random or sequential")
        if not str(reviewer).strip():
            raise ValueError("reviewer must be set so annotations stay attributable")

        self.manifest_path = Path(manifest_path)
        self.frame = ensure_annotation_columns(read_manifest(self.manifest_path))
        self.split = str(split)
        self.reviewer = str(reviewer).strip()
        self.mode = mode
        self.point_names = GEOMETRY_MODES[mode]
        self.storage_column = "keypoints_json" if mode == "keypoints" else "anchors_json"
        self.figsize = figsize

        accepted = as_bool(self.frame["accepted"]).to_numpy()
        in_split = self.frame["split"].astype(str).eq(self.split).to_numpy()
        posture_complete = self.frame["label"].astype(str).str.strip().isin(
            ["arched", "normal"]
        ).to_numpy()
        geometry_missing = (
            self.frame[self.storage_column].astype(str).str.strip().eq("").to_numpy()
        )
        candidates = np.flatnonzero(
            accepted & in_split & posture_complete & (geometry_missing | relabel)
        )
        if candidates.size == 0:
            raise ValueError(
                f"No eligible Pass B rows in split '{self.split}': posture must be "
                f"arched/normal and {self.storage_column} must be empty"
            )

        if order == "random":
            rng = np.random.default_rng(seed)
            candidates = candidates[rng.permutation(candidates.size)]
        if max_samples is not None:
            candidates = candidates[:max_samples]

        self.queue = list(candidates)
        self.position = 0
        self.points: list[tuple[float, float]] = []
        self.decisions = 0
        self.last_saved_features: dict[str, float] | None = None
        self._last_saved_message = ""
        self._build_ui()

    def _build_ui(self) -> None:
        widgets = _require_widgets()
        self.output = widgets.Output()
        self.status = widgets.HTML()
        self.progress = widgets.IntProgress(min=0, max=len(self.queue), description="queue")

        save = widgets.Button(description="save geometry", layout=widgets.Layout(width="130px"))
        save.style.button_color = "#2e86ab"
        save.on_click(lambda _b: self._on_save())
        skip = widgets.Button(description="skip", layout=widgets.Layout(width="90px"))
        skip.on_click(lambda _b: self._advance())
        back = widgets.Button(description="back", layout=widgets.Layout(width="90px"))
        back.on_click(lambda _b: self._advance(-1))
        undo = widgets.Button(description="undo point", layout=widgets.Layout(width="110px"))
        undo.on_click(lambda _b: self._undo_point())
        clear = widgets.Button(description="clear points", layout=widgets.Layout(width="120px"))
        clear.on_click(lambda _b: self._clear_points())

        self.ui = widgets.VBox(
            [
                widgets.HBox([save, skip, back, undo, clear]),
                widgets.HBox([self.progress, self.status]),
                self.output,
            ]
        )
        with self.output:
            self.figure, self.axis = plt.subplots(figsize=self.figsize)
            plt.show()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

    def display(self):
        from IPython.display import display

        self._render()
        display(self.ui)
        return self.ui

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
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return
        if len(self.points) >= len(self.point_names):
            return
        self.points.append((float(event.xdata), float(event.ydata)))
        self._render()

    def _on_save(self) -> None:
        index = self.current_index
        if index is None:
            return
        if len(self.points) != len(self.point_names):
            self.status.value = (
                f"<b style='color:#d1495b'>{len(self.points)}/{len(self.point_names)} "
                "points. Click every required point before saving.</b>"
            )
            return

        posture_before = str(self.frame.at[index, "label"])
        if posture_before not in {"arched", "normal"}:
            self.status.value = "<b style='color:#d1495b'>Pass A posture is not complete.</b>"
            return

        points = np.asarray(self.points, dtype=float)
        features = None
        if self.mode == "keypoints":
            try:
                features = keypoint_features(points)
            except ValueError as exc:
                self.status.value = f"<b style='color:#d1495b'>{exc}</b>"
                return
        elif float(np.linalg.norm(points[1] - points[0])) <= 1e-6:
            self.status.value = "<b style='color:#d1495b'>withers and sacrum cannot coincide</b>"
            return

        self.frame.at[index, self.storage_column] = json.dumps(
            [[round(x, 2), round(y, 2)] for x, y in self.points]
        )
        self.frame.at[index, "geometry_reviewed_by"] = self.reviewer
        self.frame.at[index, "annotation_pass"] = "geometry"
        # Deliberately do not assign label, reviewed_by, or posture_reviewed_by.
        atomic_write_csv(self.frame, self.manifest_path)
        if str(self.frame.at[index, "label"]) != posture_before:  # pragma: no cover - invariant
            raise RuntimeError("geometry annotation changed the posture label")

        self.last_saved_features = features
        self._last_saved_message = f"saved {self.mode} for {self.frame.at[index, 'sample_id']}"
        if features:
            score_name = (
                "kp_arch_height_signed"
                if "kp_arch_height_signed" in features
                else "kp_sagitta_norm"
            )
            self._last_saved_message += f" | {score_name}={features[score_name]:.3f}"
        self.decisions += 1
        self._advance()

    def _advance(self, step: int = 1) -> None:
        self.position = max(0, min(len(self.queue), self.position + step))
        self.points = []
        self._render()

    def _render(self) -> None:
        self.progress.value = min(self.position, len(self.queue))
        index = self.current_index
        self.axis.clear()

        if index is None:
            self.axis.text(0.5, 0.5, "queue finished", ha="center", va="center", fontsize=14)
            self.axis.axis("off")
            self.status.value = (
                f"<b>done</b> - {self.decisions} geometry annotations written to "
                f"{self.manifest_path.name}"
            )
            self.figure.canvas.draw_idle()
            return

        row = self.frame.loc[index]
        image, path = _load_crop(row, self.manifest_path)
        if image is None:
            self.axis.text(0.5, 0.5, f"unreadable crop:\n{path}", ha="center", va="center")
            self.axis.axis("off")
        else:
            self.axis.imshow(image[:, :, ::-1])
            self.axis.axis("off")

        if self.points:
            points = np.asarray(self.points, dtype=float)
            self.axis.plot(
                points[:, 0],
                points[:, 1],
                color="#f4a259",
                marker="o",
                markersize=6,
                linewidth=1.6,
            )
            for point_order, (x, y) in enumerate(self.points):
                self.axis.annotate(
                    f"{point_order + 1}.{self.point_names[point_order]}",
                    (x, y),
                    color="white",
                    fontsize=7,
                    xytext=(3, 6),
                    textcoords="offset points",
                )
            if len(self.points) == len(self.point_names):
                self.axis.plot(
                    [points[0, 0], points[-1, 0]],
                    [points[0, 1], points[-1, 1]],
                    color="white",
                    linestyle="--",
                    linewidth=1.2,
                )

        self.axis.set_title(
            f"{row['sample_id']}   ({self.position + 1}/{len(self.queue)})   "
            f"geometry={self.mode}",
            fontsize=10,
        )
        self.figure.canvas.draw_idle()
        next_name = (
            self.point_names[len(self.points)]
            if len(self.points) < len(self.point_names)
            else "ready to save"
        )
        saved = f" | {self._last_saved_message}" if self._last_saved_message else ""
        self.status.value = (
            f"reviewer <b>{self.reviewer}</b> | written {self.decisions} | points "
            f"{len(self.points)}/{len(self.point_names)} -> <code>{next_name}</code>{saved}"
        )


def label_summary(manifest_path: str | Path) -> pd.DataFrame:
    frame = read_manifest(manifest_path)
    accepted = as_bool(frame["accepted"])
    subset = frame.loc[accepted].copy()
    subset["label"] = subset["label"].astype(str).replace("", "unlabeled")
    return pd.crosstab(
        subset["split"].replace("", "unassigned"), subset["label"]
    ).reindex(columns=[*POSTURE_LABELS, "unlabeled"], fill_value=0)
