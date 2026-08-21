"""Data-derived calibration of absolute sagitta into ordinal posture bands."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score


def _spearman(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average")))


def calibrate_scores(
    passages: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    sagitta_column: str = "sagitta_median",
) -> tuple[dict, pd.DataFrame]:
    """Measure observer agreement and derive monotone ordinal score bands.

    ``observations`` is long-form: ``passage_id, observer_id, score``. Exactly
    two observers must score every included passage. Thresholds are obtained
    from an isotonic fit; no posture cutoff is supplied by the caller.
    """
    passage_required = {"passage_id", sagitta_column}
    observation_required = {"passage_id", "observer_id", "score"}
    if missing := sorted(passage_required - set(passages.columns)):
        raise ValueError(f"Passages table is missing columns: {missing}")
    if missing := sorted(observation_required - set(observations.columns)):
        raise ValueError(f"Observations table is missing columns: {missing}")

    posture = observations.copy()
    posture["passage_id"] = posture["passage_id"].astype(str).str.strip()
    posture["observer_id"] = posture["observer_id"].astype(str).str.strip()
    posture["score"] = pd.to_numeric(posture["score"], errors="coerce")
    if (
        (posture["passage_id"] == "").any()
        or (posture["observer_id"] == "").any()
        or posture["score"].isna().any()
        or not np.isfinite(posture["score"].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Observations require passage_id, observer_id and numeric score")
    if posture.duplicated(["passage_id", "observer_id"]).any():
        raise ValueError("Each observer may score a passage only once")
    observers = sorted(posture["observer_id"].unique())
    if len(observers) != 2:
        raise ValueError("Score calibration requires exactly two observers")
    wide = posture.pivot(index="passage_id", columns="observer_id", values="score")
    if wide.isna().any(axis=None):
        raise ValueError("Both observers must score every calibration passage")
    if len(wide) < 3:
        raise ValueError("At least three jointly scored passages are required")

    first = wide[observers[0]].to_numpy(dtype=float)
    second = wide[observers[1]].to_numpy(dtype=float)
    weighted_kappa = float(cohen_kappa_score(first, second, weights="quadratic"))
    if not math.isfinite(weighted_kappa):
        raise ValueError("Weighted kappa is undefined; observer scores need variation")

    consensus = wide.median(axis=1).rename("consensus_score").reset_index()
    source = passages.copy()
    source["passage_id"] = source["passage_id"].astype(str).str.strip()
    if source["passage_id"].duplicated().any():
        raise ValueError("Passages table must contain one row per passage_id")
    source["sagitta"] = pd.to_numeric(source[sagitta_column], errors="coerce")
    pairs = consensus.merge(
        source[["passage_id", "sagitta"]], on="passage_id", how="left", validate="one_to_one"
    )
    invalid_sagitta = ~np.isfinite(pairs["sagitta"].to_numpy(dtype=float))
    if invalid_sagitta.any():
        missing_ids = pairs.loc[invalid_sagitta, "passage_id"].tolist()[:5]
        raise ValueError(f"Calibration passages lack finite sagitta: {missing_ids}")

    correlation = _spearman(pairs["sagitta"], pairs["consensus_score"])
    if not math.isfinite(correlation) or abs(correlation) <= 1e-12:
        raise ValueError("Sagitta-score direction is not identifiable from these observations")
    direction = 1 if correlation > 0 else -1
    x = direction * pairs["sagitta"].to_numpy(dtype=float)
    y = pairs["consensus_score"].to_numpy(dtype=float)
    categories = sorted(float(value) for value in np.unique(posture["score"]))
    if len(categories) < 2:
        raise ValueError("At least two ordinal score categories are required")

    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    fitted = model.fit_transform(x, y)
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    sorted_fitted = fitted[order]
    thresholds = []
    for lower, upper in zip(categories[:-1], categories[1:]):
        target = (lower + upper) / 2.0
        above = np.flatnonzero(sorted_fitted >= target)
        if len(above):
            right = int(above[0])
            if right == 0:
                oriented = float(sorted_x[0])
            else:
                x0, x1 = float(sorted_x[right - 1]), float(sorted_x[right])
                y0, y1 = float(sorted_fitted[right - 1]), float(sorted_fitted[right])
                oriented = (x0 + x1) / 2.0 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)
        else:
            below_values = x[y <= target]
            above_values = x[y > target]
            if len(below_values) == 0 or len(above_values) == 0:
                raise ValueError(
                    f"Band boundary {lower:g}->{upper:g} is not identifiable from the data"
                )
            oriented = float((np.max(below_values) + np.min(above_values)) / 2.0)
        thresholds.append(
            {
                "lower_score": lower,
                "upper_score": upper,
                "oriented_threshold": float(oriented),
                "sagitta_threshold": float(direction * oriented),
            }
        )
    oriented_values = [row["oriented_threshold"] for row in thresholds]
    if np.any(np.diff(oriented_values) < 0):
        raise RuntimeError("Derived ordinal thresholds are not monotone")

    pairs["isotonic_score"] = fitted
    summary = {
        "n_passages": int(len(pairs)),
        "observers": observers,
        "weighted_kappa_quadratic": weighted_kappa,
        "sagitta_score_spearman": correlation,
        "sagitta_direction": direction,
        "score_categories": categories,
        "thresholds": thresholds,
        "threshold_rule": (
            "multiply sagitta by sagitta_direction, then apply oriented_thresholds "
            "in ascending order"
        ),
        "sagitta_column": sagitta_column,
    }
    return summary, pairs


def apply_score_bands(values: pd.Series, calibration: dict) -> pd.Series:
    """Apply persisted, data-derived calibration bands to sagitta values."""
    numeric = pd.to_numeric(values, errors="coerce")
    direction = int(calibration["sagitta_direction"])
    categories = np.asarray(calibration["score_categories"], dtype=float)
    thresholds = np.asarray(
        [row["oriented_threshold"] for row in calibration["thresholds"]], dtype=float
    )
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    finite = numeric.notna()
    indices = np.searchsorted(thresholds, direction * numeric.loc[finite].to_numpy(), side="right")
    output.loc[finite] = categories[indices]
    return output
