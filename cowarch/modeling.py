from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedBinaryModel:
    pipeline: Pipeline
    c_value: float
    threshold: float
    validation_pr_auc: float


@dataclass(frozen=True)
class GroupedPredictions:
    """One binary target and aggregated probability per independent group."""

    group_ids: np.ndarray
    y_true: np.ndarray
    probability: np.ndarray
    frame_counts: np.ndarray

    def __len__(self) -> int:
        return int(len(self.group_ids))


SUPPORTED_PROBABILITY_AGGREGATIONS = ("mean", "median", "max")


def _validated_evaluation_arrays(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)
    group_ids = np.asarray(group_ids, dtype=object)
    for name, values in (
        ("y_true", y_true),
        ("probability", probability),
        ("group_ids", group_ids),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    if not (len(y_true) == len(probability) == len(group_ids)):
        raise ValueError("y_true, probability, and group_ids must have equal length")
    if len(y_true) == 0:
        raise ValueError("evaluation arrays must not be empty")
    try:
        numeric_targets = y_true.astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError("y_true must contain binary 0/1 labels") from error
    if not np.isfinite(numeric_targets).all() or not np.isin(
        numeric_targets, [0.0, 1.0]
    ).all():
        raise ValueError("y_true must contain only binary 0/1 labels")
    y_true = numeric_targets.astype(int)
    if not np.isfinite(probability).all():
        raise ValueError("probability must contain only finite values")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("probability values must be in [0, 1]")

    normalized_groups: list[str] = []
    for value in group_ids:
        if value is None:
            raise ValueError("group_ids contains a missing value")
        try:
            missing = bool(
                np.isscalar(value)
                and np.asarray(value).dtype.kind == "f"
                and np.isnan(value)
            )
        except (TypeError, ValueError):
            missing = False
        text = str(value).strip()
        if missing or not text or text.lower() in {"nan", "none", "<na>"}:
            raise ValueError("group_ids contains a blank or missing value")
        normalized_groups.append(text)
    return y_true, probability, np.asarray(normalized_groups, dtype=object)


def _aggregate(values: np.ndarray, aggregation: str) -> float:
    if aggregation == "mean":
        return float(np.mean(values))
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "max":
        return float(np.max(values))
    choices = ", ".join(SUPPORTED_PROBABILITY_AGGREGATIONS)
    raise ValueError(
        f"Unsupported probability aggregation {aggregation!r}; choose from {choices}"
    )


def aggregate_group_predictions(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
    aggregation: str = "mean",
) -> GroupedPredictions:
    """Collapse frame predictions to one observation per group.

    Groups retain first-seen order. A group is an evaluation unit only when all
    of its frames share one ground-truth label; conflicting labels are rejected
    rather than silently resolved by a majority vote.
    """

    y_true, probability, group_ids = _validated_evaluation_arrays(
        y_true, probability, group_ids
    )
    # Validate the requested aggregation even when the first group is unusual.
    if aggregation not in SUPPORTED_PROBABILITY_AGGREGATIONS:
        _aggregate(np.asarray([0.0]), aggregation)

    members: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        members.setdefault(str(group_id), []).append(index)

    grouped_targets: list[int] = []
    grouped_probability: list[float] = []
    frame_counts: list[int] = []
    conflicts: dict[str, list[int]] = {}
    for group_id, indices in members.items():
        labels = np.unique(y_true[indices])
        if len(labels) != 1:
            conflicts[group_id] = [int(value) for value in labels]
            continue
        grouped_targets.append(int(labels[0]))
        grouped_probability.append(_aggregate(probability[indices], aggregation))
        frame_counts.append(len(indices))
    if conflicts:
        preview = dict(list(conflicts.items())[:5])
        suffix = "" if len(conflicts) <= 5 else f" (and {len(conflicts) - 5} more)"
        raise ValueError(
            "Conflicting ground-truth labels within evaluation groups: "
            f"{preview}{suffix}"
        )

    return GroupedPredictions(
        group_ids=np.asarray(list(members), dtype=object),
        y_true=np.asarray(grouped_targets, dtype=int),
        probability=np.asarray(grouped_probability, dtype=float),
        frame_counts=np.asarray(frame_counts, dtype=int),
    )


def build_logistic(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def _require_two_classes(y: np.ndarray, name: str) -> None:
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name} must contain both normal and arched labels")


def select_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.5], probability)))
    scored = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        scored.append((f1_score(y_true, pred, zero_division=0), float(threshold)))
    # Prefer the higher threshold on ties to avoid gratuitous false alarms.
    return max(scored, key=lambda item: (item[0], item[1]))[1]


def train_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
) -> TrainedBinaryModel:
    _require_two_classes(y_train, "training split")
    _require_two_classes(y_val, "validation split")
    best: tuple[float, float, Pipeline, np.ndarray] | None = None
    for c_value in c_grid:
        pipeline = build_logistic(float(c_value))
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_val)[:, 1]
        score = float(average_precision_score(y_val, probability))
        candidate = (score, -float(c_value), pipeline, probability)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    score, negative_c, pipeline, probability = best
    return TrainedBinaryModel(
        pipeline=pipeline,
        c_value=-negative_c,
        threshold=select_threshold(y_val, probability),
        validation_pr_auc=score,
    )


def binary_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    roc_auc = float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) == 2 else float("nan")
    pr_auc = (
        float(average_precision_score(y_true, probability))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )
    return {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity_recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": specificity,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "confusion_matrix_tn_fp_fn_tp": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def group_binary_metrics(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
    threshold: float,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Calculate binary metrics after reducing frames to independent groups."""

    grouped = aggregate_group_predictions(y_true, probability, group_ids, aggregation)
    metrics: dict[str, Any] = binary_metrics(
        grouped.y_true, grouped.probability, threshold
    )
    metrics.update(
        {
            "n_groups": len(grouped),
            "n_frames": int(grouped.frame_counts.sum()),
            "probability_aggregation": aggregation,
        }
    )
    return metrics


BootstrapMetric = str | Callable[[np.ndarray, np.ndarray], float]


def _metric_value(
    y_true: np.ndarray,
    probability: np.ndarray,
    metric: BootstrapMetric,
    threshold: float,
) -> float:
    if callable(metric):
        return float(metric(y_true, probability))
    if metric == "pr_auc":
        return (
            float(average_precision_score(y_true, probability))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )
    if metric == "roc_auc":
        return (
            float(roc_auc_score(y_true, probability))
            if len(np.unique(y_true)) == 2
            else float("nan")
        )
    value = binary_metrics(y_true, probability, threshold).get(metric)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"Bootstrap metric must be scalar; got {metric!r}")
    return float(value)


def cluster_bootstrap_scores(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
    *,
    metric: BootstrapMetric = "pr_auc",
    threshold: float = 0.5,
    level: str = "group",
    aggregation: str = "mean",
    n_bootstrap: int = 2000,
    random_state: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Return cluster-bootstrap metric draws, sampling groups rather than frames.

    At ``level="frame"``, a sampled group contributes all of its frames and a
    group drawn twice contributes two copies of the complete cluster. At
    ``level="group"``, probabilities are aggregated first, then group rows are
    sampled. Invalid single-class PR/ROC draws are represented by ``nan``.
    """

    if level not in {"frame", "group"}:
        raise ValueError("level must be 'frame' or 'group'")
    if (
        isinstance(n_bootstrap, bool)
        or int(n_bootstrap) != n_bootstrap
        or n_bootstrap < 1
    ):
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")

    y_true, probability, group_ids = _validated_evaluation_arrays(
        y_true, probability, group_ids
    )
    # This also enforces the one-ground-truth-label-per-group contract.
    grouped = aggregate_group_predictions(y_true, probability, group_ids, aggregation)
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    scores = np.full(int(n_bootstrap), np.nan, dtype=float)

    if level == "group":
        for draw in range(int(n_bootstrap)):
            indices = rng.integers(0, len(grouped), size=len(grouped))
            scores[draw] = _metric_value(
                grouped.y_true[indices], grouped.probability[indices], metric, threshold
            )
        return scores

    members = {
        str(group_id): np.flatnonzero(group_ids == group_id)
        for group_id in grouped.group_ids
    }
    for draw in range(int(n_bootstrap)):
        sampled_groups = rng.integers(0, len(grouped), size=len(grouped))
        indices = np.concatenate(
            [members[str(grouped.group_ids[index])] for index in sampled_groups]
        )
        scores[draw] = _metric_value(
            y_true[indices], probability[indices], metric, threshold
        )
    return scores


def cluster_bootstrap_ci(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
    *,
    metric: BootstrapMetric = "pr_auc",
    threshold: float = 0.5,
    level: str = "group",
    aggregation: str = "mean",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = 0,
) -> tuple[float, float]:
    """Percentile confidence interval from a group-resampling bootstrap."""

    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    scores = cluster_bootstrap_scores(
        y_true,
        probability,
        group_ids,
        metric=metric,
        threshold=threshold,
        level=level,
        aggregation=aggregation,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    valid = scores[np.isfinite(scores)]
    if not len(valid):
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def evaluate_frame_and_group_metrics(
    y_true: np.ndarray | Iterable[int],
    probability: np.ndarray | Iterable[float],
    group_ids: np.ndarray | Iterable[object],
    threshold: float,
    *,
    aggregation: str = "mean",
    bootstrap_metric: BootstrapMetric = "pr_auc",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = 0,
) -> dict[str, Any]:
    """Summarize frame and group performance with group-resampled intervals."""

    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    y_true, probability, group_ids = _validated_evaluation_arrays(
        y_true, probability, group_ids
    )
    grouped = aggregate_group_predictions(y_true, probability, group_ids, aggregation)
    frame_metrics: dict[str, Any] = binary_metrics(y_true, probability, threshold)
    frame_metrics["n_groups"] = len(grouped)
    group_metrics = group_binary_metrics(
        y_true, probability, group_ids, threshold, aggregation
    )

    metric_name = bootstrap_metric if isinstance(bootstrap_metric, str) else "custom"
    intervals: dict[str, dict[str, Any]] = {}
    for level in ("frame", "group"):
        scores = cluster_bootstrap_scores(
            y_true,
            probability,
            group_ids,
            metric=bootstrap_metric,
            threshold=threshold,
            level=level,
            aggregation=aggregation,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        valid = scores[np.isfinite(scores)]
        if len(valid):
            alpha = (1.0 - confidence) / 2.0
            lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
        else:
            lower, upper = float("nan"), float("nan")
        values_y = y_true if level == "frame" else grouped.y_true
        values_p = probability if level == "frame" else grouped.probability
        intervals[level] = {
            "metric": metric_name,
            "point_estimate": _metric_value(
                values_y, values_p, bootstrap_metric, threshold
            ),
            "lower": float(lower),
            "upper": float(upper),
            "confidence": float(confidence),
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": int(len(valid)),
            "resampling_unit": "group",
        }
    return {
        "probability_aggregation": aggregation,
        "frame": frame_metrics,
        "group": group_metrics,
        "cluster_bootstrap": intervals,
    }
