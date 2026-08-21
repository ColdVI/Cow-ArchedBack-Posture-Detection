"""Retrospective validation against sparse treatment records."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import as_bool


def normalize_treatments(
    treatments: pd.DataFrame,
    *,
    routine_count_threshold: int = 5,
) -> pd.DataFrame:
    """Validate treatment records and infer only missing trigger values."""
    required = {"date", "cow_id"}
    if missing := sorted(required - set(treatments.columns)):
        raise ValueError(f"Treatments table is missing columns: {missing}")
    if routine_count_threshold < 2:
        raise ValueError("routine_count_threshold must be at least two")
    frame = treatments.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.floor("D")
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    if frame["date"].isna().any() or (frame["cow_id"] == "").any():
        raise ValueError("Treatments require non-blank cow_id and parseable date")
    if "trigger" not in frame.columns:
        frame["trigger"] = ""
    frame["trigger"] = frame["trigger"].astype(str).str.strip().str.lower()
    explicit = frame["trigger"] != ""
    invalid = sorted(set(frame.loc[explicit, "trigger"]) - {"routine", "observed"})
    if invalid:
        raise ValueError(f"Unsupported treatment trigger values: {invalid}")
    counts = frame.groupby("date")["cow_id"].transform("nunique")
    frame["trigger_inferred"] = ~explicit
    frame.loc[~explicit, "trigger"] = np.where(
        counts.loc[~explicit] >= routine_count_threshold, "routine", "observed"
    )
    return frame


def normalize_calvings(calvings: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "cow_id"}
    if missing := sorted(required - set(calvings.columns)):
        raise ValueError(f"Calvings table is missing columns: {missing}")
    frame = calvings.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.floor("D")
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    if frame["date"].isna().any() or (frame["cow_id"] == "").any():
        raise ValueError("Calvings require non-blank cow_id and parseable date")
    return frame


def validate_absolute_scores(
    passages: pd.DataFrame,
    treatments: pd.DataFrame,
    calvings: pd.DataFrame,
    *,
    score_column: str = "sagitta_median",
    routine_count_threshold: int = 5,
    event_horizon_days: int = 30,
    peripartum_days_before: int = 7,
    peripartum_days_after: int = 21,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Correlate absolute passage score with subsequent observed treatment.

    Routine treatments are retained in the normalized audit table but never
    treated as positive clinical events. Passages inside the configured
    calving window are marked and excluded from the correlation.
    """
    required = {"passage_id", "cow_id", "timestamp_utc", score_column}
    if missing := sorted(required - set(passages.columns)):
        raise ValueError(f"Passages table is missing columns: {missing}")
    if event_horizon_days < 1 or min(peripartum_days_before, peripartum_days_after) < 0:
        raise ValueError("validation windows must be non-negative and event horizon positive")

    frame = passages.copy()
    if "score_eligible" in frame.columns:
        frame = frame.loc[as_bool(frame["score_eligible"])].copy()
    frame["cow_id"] = frame["cow_id"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(
        frame["timestamp_utc"], utc=True, errors="coerce"
    ).dt.floor("D")
    frame["absolute_score"] = pd.to_numeric(frame[score_column], errors="coerce")
    frame = frame.dropna(subset=["date", "absolute_score"]).copy()
    if frame.empty or (frame["cow_id"] == "").any():
        raise ValueError("No eligible passages with cow_id, timestamp and finite score")

    treatment_frame = normalize_treatments(
        treatments, routine_count_threshold=routine_count_threshold
    )
    calving_frame = normalize_calvings(calvings)
    observed = treatment_frame.loc[treatment_frame["trigger"].eq("observed")].copy()

    frame["peripartum_suppressed"] = False
    for event in calving_frame.itertuples(index=False):
        lower = event.date - pd.Timedelta(days=peripartum_days_before)
        upper = event.date + pd.Timedelta(days=peripartum_days_after)
        mask = frame["cow_id"].eq(event.cow_id) & frame["date"].between(lower, upper)
        frame.loc[mask, "peripartum_suppressed"] = True

    frame["observed_treatment_within_horizon"] = False
    frame["next_observed_treatment_date"] = ""
    for index, row in frame.iterrows():
        future = observed.loc[
            observed["cow_id"].eq(row["cow_id"])
            & observed["date"].between(
                row["date"], row["date"] + pd.Timedelta(days=event_horizon_days)
            )
        ].sort_values("date")
        if not future.empty:
            frame.at[index, "observed_treatment_within_horizon"] = True
            frame.at[index, "next_observed_treatment_date"] = future.iloc[0]["date"].date().isoformat()

    analysis = frame.loc[~frame["peripartum_suppressed"]].copy()
    outcome = analysis["observed_treatment_within_horizon"].astype(int)
    if outcome.nunique() < 2:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(analysis["absolute_score"].corr(outcome, method="pearson"))
        spearman = float(
            analysis["absolute_score"].rank(method="average").corr(
                outcome.rank(method="average"), method="pearson"
            )
        )

    event_rows = []
    for event_index, event in observed.iterrows():
        preceding = analysis.loc[
            analysis["cow_id"].eq(event["cow_id"])
            & analysis["date"].between(
                event["date"] - pd.Timedelta(days=event_horizon_days), event["date"]
            )
        ]
        event_rows.append(
            {
                "treatment_index": int(event_index),
                "cow_id": event["cow_id"],
                "treatment_date": event["date"].date().isoformat(),
                "has_preceding_score": bool(len(preceding)),
                "max_preceding_score": (
                    float(preceding["absolute_score"].max()) if len(preceding) else float("nan")
                ),
            }
        )
    events = pd.DataFrame(event_rows)
    positive_scores = analysis.loc[outcome.eq(1), "absolute_score"]
    comparison_scores = analysis.loc[outcome.eq(0), "absolute_score"]
    metrics = {
        "n_passages_total": int(len(frame)),
        "n_passages_analyzed": int(len(analysis)),
        "n_passages_peripartum_suppressed": int(frame["peripartum_suppressed"].sum()),
        "n_observed_treatment_events": int(len(observed)),
        "n_observed_events_with_preceding_score": (
            int(events["has_preceding_score"].sum()) if not events.empty else 0
        ),
        "n_passages_before_observed_treatment": int(outcome.sum()),
        "absolute_score_treatment_pearson": pearson,
        "absolute_score_treatment_spearman": spearman,
        "median_score_before_observed_treatment": (
            float(positive_scores.median()) if len(positive_scores) else float("nan")
        ),
        "median_score_without_observed_treatment": (
            float(comparison_scores.median()) if len(comparison_scores) else float("nan")
        ),
        "n_inferred_triggers": int(treatment_frame["trigger_inferred"].sum()),
        "event_horizon_days": event_horizon_days,
        "peripartum_days_before": peripartum_days_before,
        "peripartum_days_after": peripartum_days_after,
        "score_column": score_column,
    }
    treatment_frame["date"] = treatment_frame["date"].dt.date.astype(str)
    frame["date"] = frame["date"].dt.date.astype(str)
    return metrics, frame, treatment_frame


def _persistent_start(
    cow_signals: pd.DataFrame,
    event_date: pd.Timestamp,
    *,
    lookback_days: int,
    min_persistent_days: int,
) -> pd.Timestamp | None:
    lower = event_date - pd.Timedelta(days=lookback_days)
    frame = cow_signals.loc[
        cow_signals["date"].between(lower, event_date)
    ].sort_values("date")
    if frame.empty:
        return None
    daily = frame.groupby("date", as_index=False)["alert"].max().sort_values("date")
    if not bool(daily.iloc[-1]["alert"]):
        return None
    run = [daily.iloc[-1]["date"]]
    for index in range(len(daily) - 2, -1, -1):
        current = daily.iloc[index]
        if not bool(current["alert"]) or run[-1] - current["date"] > pd.Timedelta(days=1):
            break
        run.append(current["date"])
    if len(run) < min_persistent_days:
        return None
    return min(run)


def validate_retrospective(
    daily_signals: pd.DataFrame,
    treatments: pd.DataFrame,
    *,
    routine_count_threshold: int = 5,
    lookback_days: int = 30,
    min_persistent_days: int = 2,
    alert_match_days: int = 30,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Calculate latency and false-alert metrics without binary accuracy."""
    required = {"date", "cow_id", "alert"}
    if missing := sorted(required - set(daily_signals.columns)):
        raise ValueError(f"Daily signals table is missing columns: {missing}")
    if min(lookback_days, min_persistent_days, alert_match_days) < 1:
        raise ValueError("validation windows must be positive")

    signals = daily_signals.copy()
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce").dt.floor("D")
    signals["cow_id"] = signals["cow_id"].astype(str).str.strip()
    signals["alert"] = as_bool(signals["alert"])
    if signals["date"].isna().any() or (signals["cow_id"] == "").any():
        raise ValueError("Daily signals require non-blank cow_id and parseable date")
    treatment_frame = normalize_treatments(
        treatments, routine_count_threshold=routine_count_threshold
    )

    latency_rows = []
    observed = treatment_frame[treatment_frame["trigger"].eq("observed")]
    for event_index, event in observed.iterrows():
        cow_signals = signals[signals["cow_id"].eq(event["cow_id"])]
        start = _persistent_start(
            cow_signals,
            event["date"],
            lookback_days=lookback_days,
            min_persistent_days=min_persistent_days,
        )
        row = {
            "treatment_index": int(event_index),
            "cow_id": event["cow_id"],
            "treatment_date": event["date"].date().isoformat(),
            "trigger_inferred": bool(event["trigger_inferred"]),
            "persistent_signal_date": start.date().isoformat() if start is not None else "",
            "detected": start is not None,
            "detection_latency_days": (
                int((event["date"] - start).days) if start is not None else float("nan")
            ),
        }
        for column in ("lesion_type", "foot"):
            row[column] = event.get(column, "")
        latency_rows.append(row)
    latency = pd.DataFrame(latency_rows)

    study_start = signals["date"].min()
    study_end = signals["date"].max()
    treated_in_study = set(
        treatment_frame.loc[
            treatment_frame["date"].between(study_start, study_end), "cow_id"
        ]
    )
    untreated = signals.loc[~signals["cow_id"].isin(treated_in_study)].copy()
    false_alerts = int(untreated.loc[untreated["alert"]].drop_duplicates(["cow_id", "date"]).shape[0])
    exposure_months = 0.0
    for _, group in untreated.groupby("cow_id"):
        exposure_months += (int((group["date"].max() - group["date"].min()).days) + 1) / 30.4375
    false_alert_rate = false_alerts / exposure_months if exposure_months > 0 else float("nan")

    alert_days = signals.loc[signals["alert"], ["cow_id", "date"]].drop_duplicates()
    matched = 0
    for _, alert in alert_days.iterrows():
        future = treatment_frame[
            treatment_frame["cow_id"].eq(alert["cow_id"])
            & treatment_frame["date"].between(
                alert["date"], alert["date"] + pd.Timedelta(days=alert_match_days)
            )
        ]
        matched += int(not future.empty)
    measured_precision_lower_bound = matched / len(alert_days) if len(alert_days) else float("nan")

    detected_events = int(latency["detected"].sum()) if not latency.empty else 0
    finite_latency = pd.to_numeric(
        latency.get("detection_latency_days", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    metrics = {
        "n_observed_treatment_events": int(len(observed)),
        "n_detected_observed_events": detected_events,
        "median_detection_latency_days": (
            float(finite_latency.median()) if not finite_latency.empty else float("nan")
        ),
        "false_alerts_untreated_cows": false_alerts,
        "untreated_cow_months": exposure_months,
        "false_alerts_per_cow_month": false_alert_rate,
        "n_alert_cow_days": int(len(alert_days)),
        "alerts_followed_by_treatment": int(matched),
        "measured_precision_lower_bound": measured_precision_lower_bound,
        "n_inferred_triggers": int(treatment_frame["trigger_inferred"].sum()),
        "lookback_days": lookback_days,
        "min_persistent_days": min_persistent_days,
        "alert_match_days": alert_match_days,
    }
    treatment_frame["date"] = treatment_frame["date"].dt.date.astype(str)
    return metrics, latency, treatment_frame
