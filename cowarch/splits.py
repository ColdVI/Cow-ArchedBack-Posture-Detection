from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_group_splits(
    df: pd.DataFrame,
    group_column: str,
    train_fraction: float = 0.60,
    val_fraction: float = 0.20,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> pd.Series:
    fractions = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/val/test fractions must be positive and sum to 1")
    if group_column not in df.columns:
        raise ValueError(f"Missing group column: {group_column}")

    groups = df[group_column].astype(str).str.strip()
    if (groups == "").any():
        raise ValueError(f"Blank values found in group column: {group_column}")
    if groups.nunique() < 3:
        raise ValueError("At least three independent groups are required")

    indices = np.arange(len(df))
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=float(val_fraction + test_fraction),
        random_state=seed,
    )
    train_idx, temp_idx = next(outer.split(indices, groups=groups))

    temp_groups = groups.iloc[temp_idx]
    inner_test_fraction = float(test_fraction / (val_fraction + test_fraction))
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=inner_test_fraction,
        random_state=seed + 1,
    )
    val_rel, test_rel = next(inner.split(temp_idx, groups=temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    split = pd.Series("", index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    assert_no_group_leakage(pd.DataFrame({group_column: groups, "split": split}), group_column)
    return split


def assert_no_group_leakage(df: pd.DataFrame, group_column: str) -> None:
    if "split" not in df.columns:
        raise ValueError("split column is required")
    membership: dict[str, set[str]] = defaultdict(set)
    for group, split in zip(df[group_column].astype(str), df["split"].astype(str)):
        if split:
            membership[group].add(split)
    leaked = {group: sorted(splits) for group, splits in membership.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Group leakage detected: {leaked}")

