"""Few-shot prototyping and uncertainty-based sampling over frozen embeddings.

The loop is: a handful of human labels -> frozen embedding classifier -> the
frames the classifier is least sure about -> more human labels.

Two invariants are enforced here rather than left to notebook discipline:

1. Only the training split may seed or enter the active-learning pool.
2. The model only reorders what a human still has to decide. It never writes a
   label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize


class TestSetLeakError(RuntimeError):
    """Raised when test rows reach the active-learning pool."""


@dataclass
class SeedModel:
    kind: str
    predict_proba: object
    n_labeled: int
    classes_seen: tuple[str, ...]


def nearest_centroid_probabilities(
    embeddings: np.ndarray, labeled_mask: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Cosine nearest-centroid scores, mapped to a pseudo-probability in [0, 1].

    Usable from about five labels per class, where logistic regression is still
    unstable. It is a ranking aid, not a calibrated probability.
    """
    features = normalize(np.asarray(embeddings, dtype=np.float64))
    labeled = features[labeled_mask]
    y = np.asarray(targets, dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("nearest centroid needs at least one example of each class")
    positive = labeled[y == 1].mean(axis=0)
    negative = labeled[y == 0].mean(axis=0)
    positive /= max(np.linalg.norm(positive), 1e-12)
    negative /= max(np.linalg.norm(negative), 1e-12)
    margin = features @ positive - features @ negative
    return 1.0 / (1.0 + np.exp(-8.0 * margin))


def fit_seed_model(
    embeddings: np.ndarray,
    labeled_mask: np.ndarray,
    targets: np.ndarray,
    min_per_class_for_logreg: int = 8,
) -> SeedModel:
    """Pick the simplest model the label count can support."""
    y = np.asarray(targets, dtype=int)
    counts = {int(value): int((y == value).sum()) for value in (0, 1)}
    if min(counts.values()) < 1:
        raise ValueError(f"both classes are required; got {counts}")

    if min(counts.values()) >= min_per_class_for_logreg:
        scaler = StandardScaler()
        x = scaler.fit_transform(np.asarray(embeddings, dtype=np.float64)[labeled_mask])
        model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, solver="liblinear")
        model.fit(x, y)

        def predict(matrix: np.ndarray) -> np.ndarray:
            return model.predict_proba(scaler.transform(np.asarray(matrix, dtype=np.float64)))[:, 1]

        kind = "logistic_regression"
    else:
        def predict(matrix: np.ndarray, _mask=labeled_mask, _y=y) -> np.ndarray:
            everything = np.asarray(matrix, dtype=np.float64)
            return nearest_centroid_probabilities(everything, _mask, _y)

        kind = "nearest_centroid"

    return SeedModel(
        kind=kind,
        predict_proba=predict,
        n_labeled=int(labeled_mask.sum()),
        classes_seen=("normal", "arched"),
    )


def assert_pool_excludes_test(frame: pd.DataFrame, pool_mask: np.ndarray) -> None:
    leaked = frame.loc[pool_mask, "split"].astype(str).eq("test")
    if bool(leaked.any()):
        raise TestSetLeakError(
            f"{int(leaked.sum())} test rows entered the active-learning pool. "
            "Test frames must be labeled separately, in random order."
        )


def _validate_labelable_splits(labelable_splits: tuple[str, ...]) -> tuple[str, ...]:
    """Return normalized pool splits, rejecting every non-training split."""
    if isinstance(labelable_splits, str):
        splits = (labelable_splits,)
    else:
        splits = tuple(str(split) for split in labelable_splits)
    if not splits:
        raise ValueError("labelable_splits must contain 'train'")

    invalid = sorted(set(splits) - {"train"})
    if "test" in invalid:
        raise TestSetLeakError(
            "The test split cannot enter the active-learning pool. "
            "Test frames must be labeled separately, in random order."
        )
    if invalid:
        raise ValueError(
            "Active learning is training-only; labelable_splits cannot include "
            f"{invalid}. Label validation separately, in random order."
        )
    return splits


def build_pool_mask(frame: pd.DataFrame, labelable_splits=("train",)) -> np.ndarray:
    """Return accepted, unlabeled training rows eligible for active ordering."""
    from .io import as_bool

    _validate_labelable_splits(labelable_splits)
    accepted = as_bool(frame["accepted"]).to_numpy()
    unlabeled = frame["label"].astype(str).str.strip().eq("").to_numpy()
    train = frame["split"].astype(str).str.strip().eq("train").to_numpy()
    mask = accepted & unlabeled & train
    assert_pool_excludes_test(frame, mask)
    return mask


def rank_by_uncertainty(
    probabilities: np.ndarray, pool_mask: np.ndarray, batch_size: int = 20
) -> np.ndarray:
    """Positional indices of the pool rows closest to the decision boundary."""
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.flatnonzero(pool_mask)
    if candidates.size == 0:
        return np.empty(0, dtype=int)
    uncertainty = np.abs(probabilities[candidates] - 0.5)
    order = np.argsort(uncertainty, kind="stable")
    return candidates[order][:batch_size]


def select_next_batch(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    batch_size: int = 20,
    labelable_splits=("train",),
) -> tuple[np.ndarray, SeedModel, np.ndarray]:
    """Fit on train labels and return the next train-only uncertainty batch.

    Probabilities cover every input row so callers can inspect model behavior,
    but validation and test rows are never used for fitting or queue selection.
    """
    from .io import as_bool

    _validate_labelable_splits(labelable_splits)
    accepted = as_bool(frame["accepted"]).to_numpy()
    labels = frame["label"].astype(str).str.strip()
    labeled = labels.isin(["arched", "normal"]).to_numpy()
    train = frame["split"].astype(str).str.strip().eq("train").to_numpy()
    seed_mask = accepted & labeled & train
    if seed_mask.sum() < 2:
        raise ValueError(
            "at least one arched and one normal accepted train seed label are required"
        )

    targets = (
        labels.loc[seed_mask].map({"normal": 0, "arched": 1}).to_numpy(dtype=int)
    )
    model = fit_seed_model(embeddings, seed_mask, targets)
    probabilities = model.predict_proba(embeddings)

    pool_mask = build_pool_mask(frame, labelable_splits=labelable_splits)
    return rank_by_uncertainty(probabilities, pool_mask, batch_size), model, probabilities
