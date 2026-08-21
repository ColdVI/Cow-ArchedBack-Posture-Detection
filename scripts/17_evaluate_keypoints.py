#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.anchors import evaluate_pckh
from cowarch.io import atomic_write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate withers/sacrum/head PCKh.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--required-anchor-pckh", type=float, default=0.85)
    args = parser.parse_args()
    summary, details = evaluate_pckh(
        pd.read_csv(args.predictions, keep_default_na=False),
        threshold=args.threshold,
        required_anchor_pckh=args.required_anchor_pckh,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(details, args.output_dir / "pckh_details.csv")
    (args.output_dir / "pckh_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"anchor gate {'PASS' if summary['anchor_gate_pass'] else 'FAIL'}: {summary}")


if __name__ == "__main__":
    main()
