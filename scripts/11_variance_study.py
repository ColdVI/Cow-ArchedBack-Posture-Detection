#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cow_arch_matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.io import atomic_write_csv
from cowarch.variance import variance_study


def read_cow_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
        if "cow_id" not in frame.columns:
            raise ValueError("Arched cow CSV must contain cow_id")
        values = frame["cow_id"]
    else:
        values = path.read_text(encoding="utf-8").splitlines()
    return {str(value).strip() for value in values if str(value).strip()}


def json_value(value):
    if value == float("inf"):
        return "Infinity"
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the longitudinal variance GO/NO-GO study.")
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--arched-cow-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--signal-ratio-min", type=float, default=2.0)
    parser.add_argument("--camera-drift-ratio", type=float, default=2.0)
    args = parser.parse_args()

    passages = pd.read_csv(args.passages, keep_default_na=False)
    if args.camera_id:
        passages = passages.loc[passages["camera_id"].astype(str).eq(args.camera_id)].copy()
    summary, per_cow, measurements = variance_study(
        passages,
        read_cow_ids(args.arched_cow_ids),
        signal_ratio_min=args.signal_ratio_min,
        camera_drift_ratio=args.camera_drift_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(per_cow, args.output_dir / "per_cow.csv")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({key: json_value(value) for key, value in summary.items()}, handle, indent=2)

    order = per_cow.sort_values("sagitta_median")["cow_id"].tolist()
    fig_width = max(9.0, len(order) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    sns.boxplot(
        data=measurements,
        x="cow_id",
        y="sagitta",
        hue="known_arched",
        order=order,
        dodge=False,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Passage sagitta distributions by cow")
    ax.set_xlabel("cow_id")
    fig.tight_layout()
    fig.savefig(args.output_dir / "cow_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    decision_text = {
        "GO": "Signal ayrımı yeterli ve günler-arası drift baskın değil; P3'e geçilebilir.",
        "NO_GO_SIGNAL": "Δ / σ_within_cow yetersiz; bireysel değişim tespitine geçilmemeli.",
        "NO_GO_CAMERA_ENVIRONMENT": "Sinyal ayrımı yeterli, fakat günler-arası kamera/ortam varyansı baskın; önce stabilizasyon gerekli.",
    }[summary["decision"]]
    lines = [
        "# Longitudinal Variance Study",
        "",
        f"**Karar: `{summary['decision']}`** — {decision_text}",
        "",
        "| Ölçüm | Değer |",
        "|---|---:|",
        f"| İnek | {summary['n_cows']} |",
        f"| Geçiş | {summary['n_passages']} |",
        f"| Gün | {summary['n_days']} |",
        f"| σ_within_day | {summary['sigma_within_day']:.6g} |",
        f"| σ_between_day | {summary['sigma_between_day']:.6g} |",
        f"| σ_within_cow | {summary['sigma_within_cow']:.6g} |",
        f"| Δ (signed) | {summary['delta_signed']:.6g} |",
        f"| abs(Δ) / σ_within_cow | {summary['delta_over_sigma_within_cow']:.6g} |",
        f"| σ_between_day / σ_within_day | {summary['between_over_within_day']:.6g} |",
        "",
        "Kapı eşikleri CLI parametresidir: sinyal oranı en az "
        f"`{summary['signal_ratio_min']}`, kamera/ortam uyarısı oranı "
        f"`>{summary['camera_drift_ratio']}`. Bunlar posture sınıflandırma eşikleri değildir.",
        "",
        "![İnek başına dağılımlar](cow_distributions.png)",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{summary['decision']}: wrote variance study to {args.output_dir}")


if __name__ == "__main__":
    main()
