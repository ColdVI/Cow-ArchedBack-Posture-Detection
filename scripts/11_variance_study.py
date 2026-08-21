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
    parser.add_argument(
        "--passages-before", "--passages", dest="passages_before", required=True, type=Path
    )
    parser.add_argument(
        "--passages-after",
        required=True,
        type=Path,
        help="Same passages after plumb-line undistortion and centre-band filtering.",
    )
    parser.add_argument("--arched-cow-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera-id", default="")
    parser.add_argument("--signal-ratio-min", type=float, default=2.0)
    parser.add_argument("--camera-drift-ratio", type=float, default=2.0)
    args = parser.parse_args()

    arched_cow_ids = read_cow_ids(args.arched_cow_ids)
    variants = []
    results = {}
    passage_id_sets = {}
    for mitigation, path in (("before", args.passages_before), ("after", args.passages_after)):
        passages = pd.read_csv(path, keep_default_na=False)
        if args.camera_id:
            passages = passages.loc[passages["camera_id"].astype(str).eq(args.camera_id)].copy()
        if "passage_id" not in passages.columns:
            raise ValueError(f"{path} is missing passage_id")
        if mitigation == "after":
            required_mitigation = {
                "camera_mitigation", "undistortion_applied", "center_band_fraction"
            }
            if missing := sorted(required_mitigation - set(passages.columns)):
                raise ValueError(
                    f"{path} cannot prove camera mitigation; missing columns: {missing}"
                )
            applied = passages["undistortion_applied"].astype(str).str.lower().isin(
                {"1", "true", "yes", "y"}
            )
            center_fraction = pd.to_numeric(
                passages["center_band_fraction"], errors="coerce"
            )
            marker = passages["camera_mitigation"].astype(str)
            if (
                not applied.all()
                or center_fraction.isna().any()
                or not center_fraction.lt(1.0).all()
                or not marker.str.contains("plumb_line_undistortion", regex=False).all()
                or not marker.str.contains("center_band", regex=False).all()
            ):
                raise ValueError(
                    "After passages must all prove plumb-line undistortion and a "
                    "restricted center band (center_band_fraction < 1)."
                )
        passage_id_sets[mitigation] = set(passages["passage_id"].astype(str))
        required_measurements = {
            "anchored": "sagitta_median",
            "fixed_trim_20pct": "auto_sagitta_median",
        }
        if missing := [
            column for column in required_measurements.values() if column not in passages.columns
        ]:
            raise ValueError(
                f"{path} is missing two-way variance columns: {missing}. "
                "Aggregate both anchored and fixed-trim measurements."
            )
        for geometry, column in required_measurements.items():
            summary, per_cow, measurements = variance_study(
                passages,
                arched_cow_ids,
                signal_ratio_min=args.signal_ratio_min,
                camera_drift_ratio=args.camera_drift_ratio,
                measurement_column=column,
            )
            summary["mitigation"] = mitigation
            summary["geometry"] = geometry
            variants.append(summary)
            results[(mitigation, geometry)] = (summary, per_cow, measurements)

    if passage_id_sets["before"] != passage_id_sets["after"]:
        raise ValueError(
            "Before/after files must contain the same passage_id set; invalid passages "
            "must be retained and marked, not silently dropped."
        )

    summary, per_cow, measurements = results[("after", "anchored")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(per_cow, args.output_dir / "per_cow.csv")
    atomic_write_csv(pd.DataFrame(variants), args.output_dir / "comparison.csv")
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
        "GO": "Mutlak skor ayrımı sağlıklı inekler arası varyansa göre yeterli ve günler-arası drift baskın değil.",
        "NO_GO_SIGNAL": "Δ / σ_between_healthy yetersiz; mutlak skor üretime alınmamalı.",
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
        f"| σ_within_day (after/anchored) | {summary['sigma_within_day']:.6g} |",
        f"| σ_between_day (after/anchored) | {summary['sigma_between_day']:.6g} |",
        f"| σ_within_cow (after/anchored) | {summary['sigma_within_cow']:.6g} |",
        f"| σ_between_healthy (after/anchored) | {summary['sigma_between_healthy']:.6g} |",
        f"| Δ (signed, after/anchored) | {summary['delta_signed']:.6g} |",
        f"| abs(Δ) / σ_within_cow (eski okuma) | {summary['delta_over_sigma_within_cow']:.6g} |",
        f"| abs(Δ) / σ_between_healthy (v3 kapısı) | {summary['delta_over_sigma_between_healthy']:.6g} |",
        f"| σ_between_day / σ_within_day | {summary['between_over_within_day']:.6g} |",
        "",
        "## Kamera hafifletmesi ve geometri karşılaştırması",
        "",
        "| Hafifletme | Geometri | Geçiş | σ_between_healthy | Δ/σ_between_healthy | Δ/σ_within_cow |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        lines.append(
            f"| {variant['mitigation']} | {variant['geometry']} | {variant['n_passages']} | "
            f"{variant['sigma_between_healthy']:.6g} | "
            f"{variant['delta_over_sigma_between_healthy']:.6g} | "
            f"{variant['delta_over_sigma_within_cow']:.6g} |"
        )
    lines += [
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
