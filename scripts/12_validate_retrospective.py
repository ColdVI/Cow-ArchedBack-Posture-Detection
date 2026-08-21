#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.io import atomic_write_csv
from cowarch.retrospective import validate_retrospective


def json_safe(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def fmt(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(number) else f"{number:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate daily posture-change signals against treatment records."
    )
    parser.add_argument("--daily-signals", required=True, type=Path)
    parser.add_argument("--treatments", required=True, type=Path)
    parser.add_argument("--calvings", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--routine-count-threshold", type=int, default=5)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--min-persistent-days", type=int, default=2)
    parser.add_argument("--alert-match-days", type=int, default=30)
    args = parser.parse_args()

    signals = pd.read_csv(args.daily_signals, keep_default_na=False)
    treatments = pd.read_csv(args.treatments, keep_default_na=False)
    calving_count = 0
    if args.calvings:
        calvings = pd.read_csv(args.calvings, keep_default_na=False)
        if missing := sorted({"cow_id", "date"} - set(calvings.columns)):
            raise ValueError(f"Calvings table is missing columns: {missing}")
        calving_count = len(calvings)
    metrics, latency, normalized = validate_retrospective(
        signals,
        treatments,
        routine_count_threshold=args.routine_count_threshold,
        lookback_days=args.lookback_days,
        min_persistent_days=args.min_persistent_days,
        alert_match_days=args.alert_match_days,
    )
    metrics["n_calving_records_supplied"] = calving_count
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(latency, args.output_dir / "detection_latency.csv")
    atomic_write_csv(normalized, args.output_dir / "treatments_normalized.csv")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({key: json_safe(value) for key, value in metrics.items()}, handle, indent=2)

    lower = metrics["measured_precision_lower_bound"]
    lower_text = "—" if not math.isfinite(lower) else f"en az %{100 * lower:.1f}"
    lines = [
        "# Retrospektif Duruş Anomalisi Triyaj Validasyonu",
        "",
        "> Bu rapor arched-back posture değişim sinyalini değerlendirir; topallık veya hastalık teşhisi değildir.",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| Gözlem üzerine tedavi olayı | {metrics['n_observed_treatment_events']} |",
        f"| Önceden kalıcı sinyal bulunan olay | {metrics['n_detected_observed_events']} |",
        f"| Medyan detection latency (gün) | {fmt(metrics['median_detection_latency_days'])} |",
        f"| Tedavi görmemiş ineklerde aylık yanlış alarm | {fmt(metrics['false_alerts_per_cow_month'])} |",
        f"| Ölçülen precision alt sınırı | {lower_text} |",
        "",
        "## Yorum sınırları",
        "",
        "- Kayıt tarihi gerçek başlangıçtan geçse, 'kaç gün erken' değeri olduğundan büyük görünebilir.",
        "- Fark edilmemiş olaylar negatif havuzda kaldığı için ölçülen precision aşağı yönlü sapar; bu nedenle sonuç alt sınır diliyle verilir.",
        "- `trigger` eksik kayıtlar aynı gün işlem gören inek sayısından yaklaşık üretilmiştir; bugünden itibaren `routine/observed` alanı doğrudan tutulmalıdır.",
        "- Binary accuracy kasıtlı olarak raporlanmamıştır.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote retrospective validation to {args.output_dir}")


if __name__ == "__main__":
    main()
