#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.baseline import apply_threshold, capacity_threshold, score_changes
from cowarch.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEFERRED T5: build per-cow baseline/CUSUM change alerts."
    )
    parser.add_argument(
        "--enable-deferred-cusum",
        action="store_true",
        help="Explicitly opt into the v1-out-of-scope cow-specific path.",
    )
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--variance-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calvings", type=Path)
    parser.add_argument("--sigma-floor", required=True, type=float)
    parser.add_argument("--daily-capacity", required=True, type=int)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=28)
    parser.add_argument("--method", choices=["ewma", "cusum"], default="cusum")
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    parser.add_argument("--cusum-k", type=float, default=0.5)
    parser.add_argument("--peripartum-days-before", type=int, default=7)
    parser.add_argument("--peripartum-days-after", type=int, default=14)
    parser.add_argument("--calibration-end", default="")
    args = parser.parse_args()
    if not args.enable_deferred_cusum:
        raise ValueError(
            "T5 baseline/CUSUM is deferred outside v1; pass --enable-deferred-cusum "
            "only for an explicitly approved follow-up experiment."
        )

    with args.variance_summary.open(encoding="utf-8") as handle:
        variance = json.load(handle)
    if variance.get("decision") != "GO":
        raise ValueError(
            f"Deferred T5 is gated by a GO variance study; found {variance.get('decision')!r}"
        )
    passages = pd.read_csv(args.passages, keep_default_na=False)
    calvings = pd.read_csv(args.calvings, keep_default_na=False) if args.calvings else None
    scored, drift = score_changes(
        passages,
        sigma_floor=args.sigma_floor,
        min_history=args.min_history,
        window_days=args.window_days,
        method=args.method,
        ewma_alpha=args.ewma_alpha,
        cusum_k=args.cusum_k,
        calvings=calvings,
        peripartum_days_before=args.peripartum_days_before,
        peripartum_days_after=args.peripartum_days_after,
    )
    calibration = scored
    if args.calibration_end:
        end = pd.to_datetime(args.calibration_end, utc=True, errors="raise")
        calibration = scored.loc[pd.to_datetime(scored["timestamp"], utc=True) <= end]
    capacity = capacity_threshold(calibration, args.daily_capacity)
    passages_out, daily = apply_threshold(scored, capacity["threshold"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(passages_out, args.output_dir / "passage_scores.csv")
    atomic_write_csv(daily, args.output_dir / "daily_signals.csv")
    atomic_write_csv(drift, args.output_dir / "daily_camera_drift.csv")
    config = {
        **capacity,
        "method": args.method,
        "sigma_floor": args.sigma_floor,
        "min_history": args.min_history,
        "window_days": args.window_days,
        "calibration_end": args.calibration_end or "all_available_data",
    }
    with (args.output_dir / "change_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(
        f"threshold={capacity['threshold']:.6g}; "
        f"wrote {len(daily)} cow-days to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
