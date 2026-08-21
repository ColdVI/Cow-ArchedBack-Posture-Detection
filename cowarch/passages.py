"""Robust frame-to-passage aggregation for longitudinal posture monitoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .io import as_bool


REQUIRED_COLUMNS = {
    "passage_id",
    "cow_id",
    "camera_id",
    "timestamp_utc",
    "accepted",
    "reject_reason",
    "auto_sagitta",
    "auto_chord_rmse",
    "auto_circle_curvature_norm",
}


def _one_value(group: pd.DataFrame, column: str, passage_id: str, *, blank_ok: bool = False):
    values = group[column].dropna().astype(str).str.strip()
    if blank_ok:
        values = values[values != ""]
    elif (values == "").any() or values.empty:
        raise ValueError(f"Passage {passage_id!r} has blank {column}")
    unique = values.unique()
    if len(unique) > 1:
        raise ValueError(
            f"Passage {passage_id!r} maps to multiple {column} values: {unique.tolist()}"
        )
    return unique[0] if len(unique) else ""


def _numeric(group: pd.DataFrame, column: str, valid: pd.Series) -> pd.Series:
    return pd.to_numeric(group.loc[valid, column], errors="coerce").dropna()


def _median(values: pd.Series) -> float:
    return float(values.median()) if not values.empty else float("nan")


def aggregate_passages(
    frames: pd.DataFrame,
    *,
    min_quality: float = 0.5,
    min_valid_frames: int = 3,
) -> pd.DataFrame:
    """Aggregate an auditable frame manifest into one row per passage.

    No passage is silently discarded. Operational quality thresholds only set
    ``baseline_eligible``; raw counts and reject reasons remain in the output.
    """
    if not 0.0 <= min_quality <= 1.0:
        raise ValueError("min_quality must be between zero and one")
    if min_valid_frames < 1:
        raise ValueError("min_valid_frames must be positive")
    if missing := sorted(REQUIRED_COLUMNS - set(frames.columns)):
        raise ValueError(f"Frame manifest is missing columns: {missing}")
    if frames.empty:
        raise ValueError("Frame manifest is empty")

    passage_ids = frames["passage_id"].astype(str).str.strip()
    if (passage_ids == "").any():
        raise ValueError("Blank values found in passage_id")

    source = frames.copy()
    source["passage_id"] = passage_ids
    accepted = as_bool(source["accepted"])
    rows: list[dict] = []
    for passage_id, group in source.groupby("passage_id", sort=True):
        group_accepted = accepted.loc[group.index]
        n_total = int(len(group))
        n_valid = int(group_accepted.sum())

        rejected = group.loc[~group_accepted, "reject_reason"].astype(str).str.strip()
        rejected = rejected[rejected != ""]
        reject_breakdown = {
            str(reason): int(count)
            for reason, count in rejected.value_counts().sort_index().items()
        }

        sagitta = _numeric(group, "auto_sagitta", group_accepted)
        chord_rmse = _numeric(group, "auto_chord_rmse", group_accepted)
        curvature = _numeric(group, "auto_circle_curvature_norm", group_accepted)

        timestamps = pd.to_datetime(group["timestamp_utc"], utc=True, errors="coerce").dropna()
        if len(timestamps) >= 2:
            transit_seconds = float((timestamps.max() - timestamps.min()).total_seconds())
        elif len(timestamps) == 1:
            transit_seconds = 0.0
        else:
            transit_seconds = float("nan")

        confidence = pd.to_numeric(
            group.loc[group_accepted, "detection_conf"]
            if "detection_conf" in group.columns
            else pd.Series(1.0, index=group.index[group_accepted]),
            errors="coerce",
        ).dropna()
        mean_confidence = float(confidence.mean()) if not confidence.empty else 0.0
        quality = float((n_valid / n_total) * mean_confidence)

        is_ir_values = (
            as_bool(group["is_ir"]).unique().tolist()
            if "is_ir" in group.columns
            else [False]
        )
        if len(is_ir_values) != 1:
            raise ValueError(f"Passage {passage_id!r} mixes IR and non-IR frames")

        pipeline_version = (
            _one_value(group, "pipeline_version", passage_id, blank_ok=True)
            if "pipeline_version" in group.columns
            else "legacy"
        )
        timestamp_utc = (
            timestamps.min().isoformat().replace("+00:00", "Z")
            if len(timestamps)
            else ""
        )
        rows.append(
            {
                "passage_id": passage_id,
                "cow_id": _one_value(group, "cow_id", passage_id),
                "camera_id": _one_value(group, "camera_id", passage_id),
                "timestamp_utc": timestamp_utc,
                "n_frames_total": n_total,
                "n_frames_valid": n_valid,
                "reject_breakdown": json.dumps(
                    reject_breakdown, sort_keys=True, separators=(",", ":")
                ),
                "sagitta_median": _median(sagitta),
                "sagitta_iqr": (
                    float(sagitta.quantile(0.75) - sagitta.quantile(0.25))
                    if not sagitta.empty
                    else float("nan")
                ),
                "sagitta_p90": float(sagitta.quantile(0.90)) if not sagitta.empty else float("nan"),
                "chord_rmse_median": _median(chord_rmse),
                "curvature_median": _median(curvature),
                "transit_seconds": transit_seconds,
                "passage_quality": quality,
                "baseline_eligible": bool(
                    n_valid >= min_valid_frames and quality >= min_quality
                ),
                "is_ir": bool(is_ir_values[0]),
                "pipeline_version": pipeline_version or "legacy",
            }
        )
    return pd.DataFrame(rows)
