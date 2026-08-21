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
from cowarch.retrospective import validate_absolute_scores


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
        description="Validate absolute passage sagitta against treatment records."
    )
    parser.add_argument("--passages", required=True, type=Path)
    parser.add_argument("--treatments", required=True, type=Path)
    parser.add_argument("--calvings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--routine-count-threshold", type=int, default=5)
    parser.add_argument("--event-horizon-days", type=int, default=30)
    parser.add_argument("--peripartum-days-before", type=int, default=7)
    parser.add_argument("--peripartum-days-after", type=int, default=21)
    parser.add_argument("--score-column", default="sagitta_median")
    args = parser.parse_args()

    passages = pd.read_csv(args.passages, keep_default_na=False)
    treatments = pd.read_csv(args.treatments, keep_default_na=False)
    calvings = pd.read_csv(args.calvings, keep_default_na=False)
    metrics, scored_passages, normalized = validate_absolute_scores(
        passages,
        treatments,
        calvings,
        score_column=args.score_column,
        routine_count_threshold=args.routine_count_threshold,
        event_horizon_days=args.event_horizon_days,
        peripartum_days_before=args.peripartum_days_before,
        peripartum_days_after=args.peripartum_days_after,
    )
    metrics["n_calving_records_supplied"] = len(calvings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(scored_passages, args.output_dir / "passages_with_treatment_outcome.csv")
    atomic_write_csv(normalized, args.output_dir / "treatments_normalized.csv")
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({key: json_safe(value) for key, value in metrics.items()}, handle, indent=2)

    lines = [
        "# Retrospektif Mutlak Duruş Skoru Validasyonu",
        "",
        "> Bu rapor mutlak arched-back posture skorunun tedavi kayıtlarıyla ilişkisini değerlendirir; topallık veya hastalık teşhisi değildir.",
        "",
        "| Metrik | Değer |",
        "|---|---:|",
        f"| Analiz edilen geçiş | {metrics['n_passages_analyzed']} |",
        f"| Peripartum bastırılan geçiş | {metrics['n_passages_peripartum_suppressed']} |",
        f"| Gözlem üzerine tedavi olayı | {metrics['n_observed_treatment_events']} |",
        f"| Öncesinde skor bulunan olay | {metrics['n_observed_events_with_preceding_score']} |",
        f"| `{metrics['score_column']}`–tedavi Pearson korelasyonu | {fmt(metrics['absolute_score_treatment_pearson'])} |",
        f"| `{metrics['score_column']}`–tedavi Spearman korelasyonu | {fmt(metrics['absolute_score_treatment_spearman'])} |",
        f"| Tedavi öncesi medyan skor | {fmt(metrics['median_score_before_observed_treatment'])} |",
        f"| Tedavisiz ufukta medyan skor | {fmt(metrics['median_score_without_observed_treatment'])} |",
        "",
        "## Yorum sınırları",
        "",
        "- Korelasyon nedensellik veya tanı performansı değildir; yalnız kayıtlarla eşzamanlı ilişkiyi ölçer.",
        "- Fark edilmemiş vakalar karşılaştırma havuzunda kalabilir ve korelasyonu aşağı çekebilir.",
        "- `trigger` eksik kayıtlar aynı gün işlem gören inek sayısından yaklaşık üretilmiştir; bugünden itibaren `routine/observed` alanı doğrudan tutulmalıdır.",
        "- Routine işlemler pozitif olay sayılmamış, buzağılama çevresindeki geçişler bastırılmıştır.",
        "- T5 ertelendiği için detection latency ve aylık yanlış alarm kasıtlı olarak raporlanmamıştır.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote retrospective validation to {args.output_dir}")


if __name__ == "__main__":
    main()
