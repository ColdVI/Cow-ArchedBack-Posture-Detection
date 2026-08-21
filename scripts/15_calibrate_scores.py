#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.calibration import apply_score_bands, calibrate_scores
from cowarch.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive absolute posture-score bands from two blinded observers."
    )
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=Path("docs/SCORE_CALIBRATION.md"))
    parser.add_argument("--sagitta-column", default="sagitta_median")
    args = parser.parse_args()

    passages = pd.read_csv(args.passages, keep_default_na=False)
    summary, pairs = calibrate_scores(
        passages,
        pd.read_csv(args.observations, keep_default_na=False),
        sagitta_column=args.sagitta_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pairs, args.output_dir / "calibration_pairs.csv")
    passages["posture_score"] = apply_score_bands(
        passages[args.sagitta_column], summary
    )
    atomic_write_csv(passages, args.output_dir / "scored_passages.csv")
    (args.output_dir / "score_calibration.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Score Calibration",
        "",
        f"- Jointly scored passages: {summary['n_passages']}",
        f"- Observers: `{summary['observers'][0]}`, `{summary['observers'][1]}`",
        f"- Quadratic weighted kappa: {summary['weighted_kappa_quadratic']:.4f}",
        f"- Sagitta–consensus score Spearman correlation: {summary['sagitta_score_spearman']:.4f}",
        f"- Sagitta direction: `{summary['sagitta_direction']}`",
        "",
        "## Data-derived band boundaries",
        "",
        "| Lower score | Upper score | Sagitta boundary | Oriented boundary |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["thresholds"]:
        lines.append(
            f"| {row['lower_score']:g} | {row['upper_score']:g} | "
            f"{row['sagitta_threshold']:.8g} | {row['oriented_threshold']:.8g} |"
        )
    lines += [
        "",
        "Boundaries come from a monotone isotonic fit to the jointly scored passages; "
        "no fixed sagitta cutoff is hardcoded. Apply them using the rule stored in "
        "`score_calibration.json`.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote calibration artifacts to {args.output_dir} and {args.report}")


if __name__ == "__main__":
    main()
