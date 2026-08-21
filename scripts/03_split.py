#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.io import as_bool, atomic_write_csv, read_manifest
from cowarch.splits import assign_group_splits, assert_no_group_leakage


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe group splits in the manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group-column", default="cow_id")
    parser.add_argument("--train", type=float, default=0.60)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = read_manifest(args.manifest)
    accepted = as_bool(frame["accepted"])
    eligible = frame.loc[accepted].copy()
    if eligible.empty:
        raise ValueError("Manifest has no accepted samples")

    assigned = assign_group_splits(
        eligible,
        group_column=args.group_column,
        train_fraction=args.train,
        val_fraction=args.val,
        test_fraction=args.test,
        seed=args.seed,
    )
    frame.loc[eligible.index, "split"] = assigned
    assert_no_group_leakage(frame.loc[accepted], args.group_column)
    atomic_write_csv(frame, args.manifest)

    summary = (
        frame.loc[accepted]
        .groupby("split")
        .agg(samples=("sample_id", "size"), groups=(args.group_column, "nunique"))
    )
    print(summary.to_string())
    print(f"updated {args.manifest}")


if __name__ == "__main__":
    main()
