"""Causal per-cow baselines and capacity-calibrated change detection."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .io import as_bool


def herd_daily_drift(passages: pd.DataFrame) -> pd.DataFrame:
    """Return daily herd medians and offsets, stratified by camera/version."""
    frame = passages.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["sagitta"] = pd.to_numeric(frame["sagitta_median"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "sagitta"])
    frame["date"] = frame["timestamp"].dt.floor("D")
    keys = ["camera_id", "pipeline_version"]
    daily = (
        frame.groupby(keys + ["date"], as_index=False)
        .agg(herd_sagitta_median=("sagitta", "median"), n_cows=("cow_id", "nunique"))
        .sort_values(keys + ["date"])
    )
    references = daily.groupby(keys)["herd_sagitta_median"].transform("median")
    daily["herd_reference_median"] = references
    daily["camera_drift_offset"] = daily["herd_sagitta_median"] - references
    return daily


def _calving_dates(calvings: pd.DataFrame | None) -> dict[str, list[pd.Timestamp]]:
    if calvings is None or calvings.empty:
        return {}
    required = {"cow_id", "date"}
    if missing := sorted(required - set(calvings.columns)):
        raise ValueError(f"Calvings table is missing columns: {missing}")
    frame = calvings.copy()
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.floor("D")
    if frame["date"].isna().any() or (frame["cow_id"] == "").any():
        raise ValueError("Calvings require non-blank cow_id and parseable date")
    return {
        cow_id: sorted(group["date"].tolist())
        for cow_id, group in frame.groupby("cow_id", sort=False)
    }


def score_changes(
    passages: pd.DataFrame,
    *,
    sigma_floor: float,
    min_history: int = 10,
    window_days: int = 28,
    method: str = "ewma",
    ewma_alpha: float = 0.3,
    cusum_k: float = 0.5,
    calvings: pd.DataFrame | None = None,
    peripartum_days_before: int = 7,
    peripartum_days_after: int = 14,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute causal scores; thresholds are deliberately calibrated elsewhere."""
    required = {
        "passage_id", "cow_id", "camera_id", "timestamp_utc", "sagitta_median",
        "passage_quality", "pipeline_version",
    }
    if missing := sorted(required - set(passages.columns)):
        raise ValueError(f"Passages table is missing columns: {missing}")
    if sigma_floor <= 0:
        raise ValueError("sigma_floor must be positive")
    if min_history < 1 or window_days < 1:
        raise ValueError("min_history and window_days must be positive")
    if method not in {"ewma", "cusum"}:
        raise ValueError("method must be ewma or cusum")
    if not 0 < ewma_alpha <= 1:
        raise ValueError("ewma_alpha must be in (0, 1]")
    if cusum_k < 0:
        raise ValueError("cusum_k must be non-negative")

    frame = passages.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    frame["sagitta"] = pd.to_numeric(frame["sagitta_median"], errors="coerce")
    if frame["timestamp"].isna().any() or frame["sagitta"].isna().any():
        raise ValueError("Every passage needs a parseable timestamp and finite sagitta_median")
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    if (frame["cow_id"] == "").any():
        raise ValueError("Blank cow_id found in passages")
    frame["baseline_eligible"] = (
        as_bool(frame["baseline_eligible"])
        if "baseline_eligible" in frame.columns
        else True
    )
    frame["date"] = frame["timestamp"].dt.floor("D")
    drift = herd_daily_drift(frame.loc[frame["baseline_eligible"]])
    join_keys = ["camera_id", "pipeline_version", "date"]
    frame = frame.merge(
        drift[join_keys + ["camera_drift_offset"]],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    frame["corrected_sagitta"] = frame["sagitta"] - frame["camera_drift_offset"]
    frame = frame.sort_values(["cow_id", "timestamp", "passage_id"]).reset_index(drop=True)

    calving_map = _calving_dates(calvings)
    output_rows = []
    state: dict[tuple[str, str, str, str], float] = {}
    for _, current in frame.iterrows():
        cow_id = current["cow_id"]
        timestamp = current["timestamp"]
        cow_calvings = calving_map.get(cow_id, [])
        previous_calvings = [date for date in cow_calvings if date <= current["date"]]
        most_recent_calving = max(previous_calvings) if previous_calvings else None
        # A calving starts a new baseline/change-detector episode. Including it
        # in the state key prevents pre-calving EWMA/CUSUM state leaking across
        # the reset even after enough post-calving history has accumulated.
        episode = most_recent_calving.isoformat() if most_recent_calving is not None else "initial"
        key = (
            cow_id,
            str(current["camera_id"]),
            str(current["pipeline_version"]),
            episode,
        )
        peripartum = any(
            date - pd.Timedelta(days=peripartum_days_before)
            <= current["date"]
            <= date + pd.Timedelta(days=peripartum_days_after)
            for date in cow_calvings
        )

        history = frame[
            (frame["cow_id"] == cow_id)
            & (frame["camera_id"].astype(str) == str(current["camera_id"]))
            & (frame["pipeline_version"].astype(str) == str(current["pipeline_version"]))
            & (frame["timestamp"] < timestamp)
            & (frame["timestamp"] >= timestamp - pd.Timedelta(days=window_days))
            & frame["baseline_eligible"]
        ]
        if most_recent_calving is not None:
            history = history.loc[history["timestamp"] > most_recent_calving]

        row = current.to_dict()
        row.update(
            {
                "history_count": int(len(history)),
                "baseline_median": float("nan"),
                "baseline_mad": float("nan"),
                "baseline_sigma": float("nan"),
                "standardized_deviation": float("nan"),
                "change_score": float("nan"),
                "peripartum_suppressed": bool(peripartum),
                "status": "scored",
            }
        )
        if not bool(current["baseline_eligible"]):
            row["status"] = "low_quality"
        elif len(history) < min_history:
            row["status"] = "insufficient_history"
        else:
            values = history["corrected_sagitta"].to_numpy(dtype=float)
            baseline = float(np.median(values))
            mad = float(np.median(np.abs(values - baseline)))
            sigma = max(1.4826 * mad, float(sigma_floor))
            deviation = float((current["corrected_sagitta"] - baseline) / sigma)
            positive = max(0.0, deviation)
            previous = state.get(key, 0.0)
            if method == "ewma":
                score = ewma_alpha * positive + (1.0 - ewma_alpha) * previous
            else:
                score = max(0.0, previous + deviation - cusum_k)
            state[key] = score
            row.update(
                {
                    "baseline_median": baseline,
                    "baseline_mad": mad,
                    "baseline_sigma": sigma,
                    "standardized_deviation": deviation,
                    "change_score": float(score),
                    "status": "peripartum_suppressed" if peripartum else "scored",
                }
            )
        output_rows.append(row)
    return pd.DataFrame(output_rows), drift


def capacity_threshold(scored: pd.DataFrame, daily_capacity: int) -> dict:
    """Derive a global score threshold from inspectable cows/day capacity."""
    if daily_capacity < 1:
        raise ValueError("daily_capacity must be positive")
    eligible = scored.loc[
        scored["status"].eq("scored") & pd.to_numeric(scored["change_score"], errors="coerce").notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No scored, unsuppressed passages are available for calibration")
    eligible["date"] = pd.to_datetime(eligible["timestamp"], utc=True).dt.floor("D")
    cow_day = eligible.groupby(["date", "cow_id"], as_index=False)["change_score"].max()
    active_per_day = cow_day.groupby("date")["cow_id"].nunique()
    typical_active = float(active_per_day.median())
    fraction = min(1.0, daily_capacity / typical_active)
    quantile = max(0.0, 1.0 - fraction)

    # Select the least restrictive observed-score boundary whose calibration
    # days never exceed the stated inspection capacity. A strict comparison
    # handles score ties conservatively: when ten cows share the same boundary
    # score, we do not arbitrarily choose identities to fit the quota.
    scores = cow_day["change_score"].to_numpy(dtype=float)
    candidates = np.r_[np.nextafter(float(np.min(scores)), -np.inf), np.unique(scores)]
    selected_counts = None
    threshold = float(candidates[-1])
    for candidate in candidates:
        counts = (
            cow_day.assign(flag=cow_day["change_score"] > candidate)
            .groupby("date")["flag"]
            .sum()
        )
        if int(counts.max()) <= daily_capacity:
            threshold = float(candidate)
            selected_counts = counts
            break
    assert selected_counts is not None
    daily_counts = selected_counts
    return {
        "threshold": threshold,
        "daily_capacity": int(daily_capacity),
        "typical_active_cows_per_day": typical_active,
        "target_capacity_quantile": quantile,
        "threshold_rule": "change_score > threshold; calibration max flags/day <= capacity",
        "mean_flags_per_day": float(daily_counts.mean()),
        "max_flags_per_day": int(daily_counts.max()),
        "n_calibration_cow_days": int(len(cow_day)),
    }


def apply_threshold(scored: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach passage alerts and one auditable row per cow/day."""
    result = scored.copy()
    result["alert"] = (
        result["status"].eq("scored")
        & (pd.to_numeric(result["change_score"], errors="coerce") > threshold)
    )
    result["threshold"] = float(threshold)
    result["date"] = pd.to_datetime(result["timestamp"], utc=True).dt.date.astype(str)
    daily = (
        result.groupby(["date", "cow_id", "camera_id", "pipeline_version"], as_index=False)
        .agg(
            change_score=("change_score", "max"),
            alert=("alert", "max"),
            n_passages=("passage_id", "size"),
            any_low_quality=("status", lambda values: bool((values == "low_quality").any())),
            any_peripartum_suppressed=(
                "peripartum_suppressed", lambda values: bool(pd.Series(values).any())
            ),
        )
    )
    daily["threshold"] = float(threshold)
    return result, daily
