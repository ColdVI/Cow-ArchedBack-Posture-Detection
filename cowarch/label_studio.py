"""Safe bridge between the CowArch manifest and Label Studio JSON exports.

The two annotation passes stay separate by design. Posture tasks never expose
geometry or source scores, while geometry tasks are generated only after a
binary posture decision exists. Imports are written to a caller-selected
manifest so the original CSV is never overwritten accidentally.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import pandas as pd

from .annotations import POSTURE_LABELS, ensure_annotation_columns
from .geometry import DORSAL_KEYPOINTS
from .io import as_bool, atomic_write_csv, read_manifest, resolve_data_path

PHASES = ("posture", "geometry")
BINARY_POSTURE_LABELS = ("arched", "normal")


def _atomic_write_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _local_image_url(
    crop_path: str,
    manifest_path: str | Path,
    data_root: str | Path,
) -> tuple[str, Path]:
    image_path = resolve_data_path(crop_path, manifest_path).resolve()
    root = Path(data_root).resolve()
    try:
        relative = image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Crop is outside Label Studio data root: {image_path} (root: {root})"
        ) from exc
    encoded = quote(relative.as_posix(), safe="/")
    return f"/data/local-files/?d={encoded}", image_path


def build_label_studio_tasks(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    phase: str,
    data_root: str | Path,
    split: str | None = None,
    include_completed: bool = False,
    approved_sources_path: str | Path | None = None,
) -> dict[str, int]:
    """Create Label Studio import JSON for one blinded annotation phase."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    manifest_path = Path(manifest_path)
    frame = ensure_annotation_columns(read_manifest(manifest_path).copy())
    selected = as_bool(frame["accepted"])
    accepted_before_source_gate = int(selected.sum())
    if approved_sources_path is not None:
        sources = pd.read_csv(approved_sources_path, keep_default_na=False)
        required = {"source_id", "license_status"}
        missing_columns = sorted(required - set(sources.columns))
        if missing_columns:
            raise ValueError(
                f"Approved sources file is missing columns: {missing_columns}"
            )
        statuses = sources["license_status"].astype(str).str.strip().str.lower()
        invalid_statuses = sorted(set(statuses) - {"approved"})
        if invalid_statuses:
            raise ValueError(
                "Approved sources file contains non-approved license statuses: "
                f"{invalid_statuses}"
            )
        approved_source_ids = set(sources["source_id"].astype(str))
        selected &= frame["source_id"].astype(str).isin(approved_source_ids)
    accepted_after_source_gate = int(selected.sum())
    if split is not None:
        selected &= frame["split"].astype(str).eq(str(split))

    if phase == "posture":
        if not include_completed:
            selected &= frame["label"].astype(str).str.strip().eq("")
    else:
        selected &= frame["label"].astype(str).str.strip().isin(BINARY_POSTURE_LABELS)
        if not include_completed:
            selected &= frame["keypoints_json"].astype(str).str.strip().eq("")

    tasks: list[dict[str, Any]] = []
    missing: list[Path] = []
    for _, row in frame.loc[selected].iterrows():
        image_url, image_path = _local_image_url(
            str(row["crop_path"]), manifest_path, data_root
        )
        if not image_path.is_file():
            missing.append(image_path)
            continue

        # Only ``image`` is rendered by the labeling configs. Provenance stays
        # in task data for a lossless round trip but remains hidden in Pass A.
        data: dict[str, Any] = {
            "image": image_url,
            "sample_id": str(row["sample_id"]),
        }
        for column in (
            "source_id",
            "video_id",
            "cow_id",
            "passage_id",
            "farm_id",
            "split",
            "license_status",
            "license_name",
        ):
            if column in frame.columns:
                data[column] = str(row.get(column, ""))
        tasks.append({"data": data})

    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"{len(missing)} selected crops are missing: {preview}{suffix}"
        )

    sample_ids = [task["data"]["sample_id"] for task in tasks]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Selected manifest rows contain duplicate sample_id values")

    _atomic_write_json(tasks, output_path)
    return {
        "manifest_rows": len(frame),
        "accepted_rows": accepted_before_source_gate,
        "excluded_by_source_gate": (
            accepted_before_source_gate - accepted_after_source_gate
        ),
        "tasks_written": len(tasks),
    }


def _load_tasks(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Label Studio export must be a JSON list of tasks")
    return payload


def _single_completed_annotation(task: dict[str, Any], sample_id: str) -> dict[str, Any] | None:
    annotations = task.get("annotations") or []
    completed = [
        annotation
        for annotation in annotations
        if not annotation.get("was_cancelled") and annotation.get("result")
    ]
    if not completed:
        return None
    if len(completed) > 1:
        raise ValueError(
            f"Task {sample_id!r} has {len(completed)} completed annotations; "
            "adjudicate them before importing a single ground truth"
        )
    return completed[0]


def _reviewer_name(annotation: dict[str, Any], fallback: str | None) -> str:
    if fallback and fallback.strip():
        return fallback.strip()
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        for key in ("email", "username", "first_name", "id"):
            value = completed_by.get(key)
            if value not in (None, ""):
                return str(value).strip()
    elif completed_by not in (None, ""):
        return str(completed_by).strip()
    raise ValueError(
        "Reviewer identity is missing from the export; pass an explicit reviewer"
    )


def _posture_value(annotation: dict[str, Any], sample_id: str) -> str:
    matches = [
        result
        for result in annotation.get("result", [])
        if result.get("from_name") == "posture" and result.get("type") == "choices"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Task {sample_id!r} must contain exactly one posture result"
        )
    choices = matches[0].get("value", {}).get("choices", [])
    if len(choices) != 1 or choices[0] not in POSTURE_LABELS:
        raise ValueError(
            f"Task {sample_id!r} has invalid posture choice: {choices!r}"
        )
    return str(choices[0])


def _geometry_value(annotation: dict[str, Any], sample_id: str) -> list[list[float]]:
    matches = [
        result
        for result in annotation.get("result", [])
        if result.get("from_name") == "dorsal_keypoints"
        and result.get("type") == "keypointlabels"
    ]
    by_name: dict[str, list[float]] = {}
    for result in matches:
        labels = result.get("value", {}).get("keypointlabels", [])
        if len(labels) != 1 or labels[0] not in DORSAL_KEYPOINTS:
            raise ValueError(
                f"Task {sample_id!r} has invalid dorsal keypoint label: {labels!r}"
            )
        name = str(labels[0])
        if name in by_name:
            raise ValueError(f"Task {sample_id!r} has duplicate keypoint {name!r}")
        width = result.get("original_width")
        height = result.get("original_height")
        rotation = result.get("image_rotation", 0)
        x_percent = result.get("value", {}).get("x")
        y_percent = result.get("value", {}).get("y")
        if not all(isinstance(value, (int, float)) for value in (width, height, x_percent, y_percent)):
            raise ValueError(
                f"Task {sample_id!r} keypoint {name!r} is missing numeric coordinates"
            )
        if rotation not in (0, 0.0, None):
            raise ValueError(
                f"Task {sample_id!r} uses image rotation; annotate unrotated crops"
            )
        if width <= 0 or height <= 0 or not (0 <= x_percent <= 100 and 0 <= y_percent <= 100):
            raise ValueError(
                f"Task {sample_id!r} keypoint {name!r} has out-of-range coordinates"
            )
        by_name[name] = [
            round(float(x_percent) * float(width) / 100.0, 3),
            round(float(y_percent) * float(height) / 100.0, 3),
        ]

    missing = [name for name in DORSAL_KEYPOINTS if name not in by_name]
    if missing:
        raise ValueError(
            f"Task {sample_id!r} is missing dorsal keypoints: {missing}"
        )
    return [by_name[name] for name in DORSAL_KEYPOINTS]


def import_label_studio_results(
    manifest_path: str | Path,
    export_path: str | Path,
    output_manifest: str | Path,
    *,
    phase: str,
    reviewer: str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, int]:
    """Merge one Label Studio phase into a new, atomically written manifest."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")

    manifest_path = Path(manifest_path)
    output_manifest = Path(output_manifest)
    if output_manifest.resolve() == manifest_path.resolve():
        raise ValueError(
            "Refusing to overwrite the source manifest; choose a new output path"
        )

    frame = ensure_annotation_columns(read_manifest(manifest_path).copy())
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("Manifest contains duplicate sample_id values")
    row_by_sample = {
        str(sample_id): int(index)
        for index, sample_id in frame["sample_id"].items()
    }

    tasks = _load_tasks(export_path)
    seen: set[str] = set()
    imported = 0
    skipped_unannotated = 0
    for task in tasks:
        data = task.get("data") or {}
        sample_id = str(data.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("A Label Studio task is missing data.sample_id")
        if sample_id in seen:
            raise ValueError(f"Label Studio export repeats sample_id {sample_id!r}")
        seen.add(sample_id)
        if sample_id not in row_by_sample:
            raise ValueError(f"Unknown sample_id in Label Studio export: {sample_id!r}")

        annotation = _single_completed_annotation(task, sample_id)
        if annotation is None:
            skipped_unannotated += 1
            continue
        identity = _reviewer_name(annotation, reviewer)
        index = row_by_sample[sample_id]

        if phase == "posture":
            new_value = _posture_value(annotation, sample_id)
            old_value = str(frame.at[index, "label"]).strip()
            if old_value and old_value != new_value and not allow_overwrite:
                raise ValueError(
                    f"Task {sample_id!r} would overwrite posture {old_value!r} "
                    f"with {new_value!r}; use allow_overwrite only after adjudication"
                )
            frame.at[index, "label"] = new_value
            frame.at[index, "posture_reviewed_by"] = identity
            frame.at[index, "reviewed_by"] = identity
            frame.at[index, "annotation_pass"] = "posture"
        else:
            posture = str(frame.at[index, "label"]).strip()
            if posture not in BINARY_POSTURE_LABELS:
                raise ValueError(
                    f"Task {sample_id!r} has no binary posture decision in the manifest"
                )
            points = _geometry_value(annotation, sample_id)
            new_value = json.dumps(points, separators=(",", ":"))
            old_value = str(frame.at[index, "keypoints_json"]).strip()
            if old_value and old_value != new_value and not allow_overwrite:
                raise ValueError(
                    f"Task {sample_id!r} would overwrite existing geometry; "
                    "use allow_overwrite only after adjudication"
                )
            frame.at[index, "keypoints_json"] = new_value
            frame.at[index, "geometry_reviewed_by"] = identity
            frame.at[index, "annotation_pass"] = "geometry"
        imported += 1

    atomic_write_csv(frame, output_manifest)
    return {
        "tasks_read": len(tasks),
        "annotations_imported": imported,
        "unannotated_skipped": skipped_unannotated,
        "manifest_rows": len(frame),
    }
