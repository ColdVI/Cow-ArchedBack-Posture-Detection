from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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

