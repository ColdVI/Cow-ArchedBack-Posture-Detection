"""Variance decomposition and GO/NO-GO gate for longitudinal measurements."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .io import as_bool


def _pooled_sd(groups: list[np.ndarray]) -> float:
    usable = [values[np.isfinite(values)] for values in groups]
    usable = [values for values in usable if len(values) >= 2]
    degrees = sum(len(values) - 1 for values in usable)
    if degrees == 0:
        return float("nan")
    variance = sum((len(values) - 1) * float(np.var(values, ddof=1)) for values in usable)
    return float(math.sqrt(variance / degrees))


def variance_study(
    passages: pd.DataFrame,
    arched_cow_ids: set[str],
    *,
    signal_ratio_min: float = 2.0,
    camera_drift_ratio: float = 2.0,
    measurement_column: str = "sagitta_median",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return the v3 absolute-score variance gate for one measurement path."""
    required = {"cow_id", "camera_id", "timestamp_utc", measurement_column}
    if missing := sorted(required - set(passages.columns)):
        raise ValueError(f"Passages table is missing columns: {missing}")
    if signal_ratio_min <= 0 or camera_drift_ratio <= 0:
        raise ValueError("ratio thresholds must be positive")

    frame = passages.copy()
    if "score_eligible" in frame.columns:
        frame = frame.loc[as_bool(frame["score_eligible"])].copy()
    elif "baseline_eligible" in frame.columns:
        frame = frame.loc[as_bool(frame["baseline_eligible"])].copy()
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    if frame.empty:
        raise ValueError("No score-eligible passages were found")
    if (frame["cow_id"] == "").any():
        raise ValueError("Blank cow_id found in eligible passages")

    cameras = frame["camera_id"].astype(str).str.strip().unique()
    if len(cameras) != 1:
        raise ValueError(
            "Variance study must be run per camera; filter passages to one camera_id"
        )
    if "pipeline_version" in frame.columns:
        versions = frame["pipeline_version"].astype(str).str.strip().unique()
        if len(versions) != 1:
            raise ValueError(
                "Variance study cannot mix pipeline versions; select one version"
            )
    else:
        versions = np.array(["legacy"])

    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["sagitta"] = pd.to_numeric(frame[measurement_column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "sagitta"]).copy()
    if frame.empty:
        raise ValueError("Eligible passages have no finite timestamped sagitta values")
    frame["date"] = frame["timestamp"].dt.date.astype(str)

    normalized_arched = {str(value).strip() for value in arched_cow_ids if str(value).strip()}
    frame["known_arched"] = frame["cow_id"].isin(normalized_arched)
    observed = set(frame["cow_id"])
    missing_arched = sorted(normalized_arched - observed)

    cow_rows = []
    for cow_id, group in frame.groupby("cow_id", sort=True):
        daily = group.groupby("date")["sagitta"].median()
        cow_rows.append(
            {
                "cow_id": cow_id,
                "known_arched": bool(cow_id in normalized_arched),
                "n_passages": int(len(group)),
                "n_days": int(group["date"].nunique()),
                "sagitta_median": float(group["sagitta"].median()),
                "within_cow_sd": float(group["sagitta"].std(ddof=1)) if len(group) >= 2 else float("nan"),
                "between_day_sd": float(daily.std(ddof=1)) if len(daily) >= 2 else float("nan"),
            }
        )
    per_cow = pd.DataFrame(cow_rows)
    if not per_cow["known_arched"].any() or per_cow["known_arched"].all():
        raise ValueError("Observed data must contain known-arched and comparison cows")

    within_day_groups = [
        group["sagitta"].to_numpy(dtype=float)
        for _, group in frame.groupby(["cow_id", "date"], sort=False)
    ]
    daily_medians = (
        frame.groupby(["cow_id", "date"], as_index=False)["sagitta"].median()
    )
    between_day_groups = [
        group["sagitta"].to_numpy(dtype=float)
        for _, group in daily_medians.groupby("cow_id", sort=False)
    ]
    within_cow_groups = [
        group["sagitta"].to_numpy(dtype=float)
        for _, group in frame.groupby("cow_id", sort=False)
    ]
    sigma_within_day = _pooled_sd(within_day_groups)
    sigma_between_day = _pooled_sd(between_day_groups)
    sigma_within_cow = _pooled_sd(within_cow_groups)

    arched_median = float(per_cow.loc[per_cow["known_arched"], "sagitta_median"].median())
    normal_median = float(per_cow.loc[~per_cow["known_arched"], "sagitta_median"].median())
    delta_signed = arched_median - normal_median
    healthy_medians = per_cow.loc[~per_cow["known_arched"], "sagitta_median"]
    sigma_between_healthy = (
        float(healthy_medians.std(ddof=1)) if len(healthy_medians) >= 2 else float("nan")
    )
    if np.isfinite(sigma_within_cow) and sigma_within_cow > 0:
        within_signal_ratio = abs(delta_signed) / sigma_within_cow
    elif abs(delta_signed) > 0 and sigma_within_cow == 0:
        within_signal_ratio = float("inf")
    else:
        within_signal_ratio = float("nan")
    if np.isfinite(sigma_between_healthy) and sigma_between_healthy > 0:
        signal_ratio = abs(delta_signed) / sigma_between_healthy
    elif abs(delta_signed) > 0 and sigma_between_healthy == 0:
        signal_ratio = float("inf")
    else:
        signal_ratio = float("nan")

    if np.isfinite(sigma_within_day) and sigma_within_day > 0:
        day_ratio = sigma_between_day / sigma_within_day
    elif sigma_between_day > 0 and sigma_within_day == 0:
        day_ratio = float("inf")
    else:
        day_ratio = float("nan")

    signal_pass = bool(np.isfinite(signal_ratio) and signal_ratio >= signal_ratio_min) or (
        np.isinf(signal_ratio) and signal_ratio > 0
    )
    camera_warning = bool(
        (np.isfinite(day_ratio) and day_ratio > camera_drift_ratio)
        or np.isinf(day_ratio)
    )
    if not signal_pass:
        decision = "NO_GO_SIGNAL"
    elif camera_warning:
        decision = "NO_GO_CAMERA_ENVIRONMENT"
    else:
        decision = "GO"

    summary = {
        "decision": decision,
        "camera_id": str(cameras[0]),
        "pipeline_version": str(versions[0]),
        "n_passages": int(len(frame)),
        "n_cows": int(frame["cow_id"].nunique()),
        "n_days": int(frame["date"].nunique()),
        "n_known_arched_cows": int(per_cow["known_arched"].sum()),
        "missing_known_arched_cow_ids": missing_arched,
        "sigma_within_day": sigma_within_day,
        "sigma_between_day": sigma_between_day,
        "sigma_within_cow": sigma_within_cow,
        "sigma_between_healthy": sigma_between_healthy,
        "arched_cow_median": arched_median,
        "comparison_cow_median": normal_median,
        "delta_signed": delta_signed,
        "delta_absolute": abs(delta_signed),
        "delta_over_sigma_within_cow": within_signal_ratio,
        "delta_over_sigma_between_healthy": signal_ratio,
        "between_over_within_day": day_ratio,
        "signal_ratio_min": signal_ratio_min,
        "camera_drift_ratio": camera_drift_ratio,
        "signal_pass": signal_pass,
        "camera_environment_warning": camera_warning,
        "measurement_column": measurement_column,
    }
    return summary, per_cow, frame
