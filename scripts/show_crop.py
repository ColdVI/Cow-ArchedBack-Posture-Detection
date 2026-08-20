#!/usr/bin/env python3
"""Render one manifest crop with its topline/reference-line overlay.

Usage: python scripts/show_crop.py <sample_id_or_substring> [out.png]

Not part of the pipeline - a quick lookup helper for eyeballing a specific
crop next to its geometric reference line while labeling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cowarch.geometry import chord_values, extract_topline

def main():
    if len(sys.argv) < 2:
        print("usage: show_crop.py <sample_id_or_substring> [out.png]")
        sys.exit(1)
    query = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/crop_lookup.png"

    df = pd.read_csv("data/manifest.csv", keep_default_na=False)
    matches = df[df["sample_id"].str.contains(query, regex=False)]
    if matches.empty:
        print(f"no sample_id matches '{query}'")
        sys.exit(1)
    if len(matches) > 1:
        print(f"{len(matches)} matches, using the first:")
        print(matches["sample_id"].head(10).to_string(index=False))
    row = matches.iloc[0]

    crop = cv2.imread(row["crop_path"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(crop[:, :, ::-1])

    mask_path = str(row.get("mask_path", "")).strip()
    if mask_path and Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
        extracted = extract_topline(mask, trim=0.20)
        if extracted is not None:
            px, py = extracted
            chord = chord_values(px, py)
            ax.plot(px, py, color="#f4a259", linewidth=3, label="gercek sirt hatti")
            ax.plot(px, chord, color="white", linewidth=1.5, linestyle="--", label="duz referans")
            peak = int(np.argmax(chord - py))
            ax.plot([px[peak], px[peak]], [py[peak], chord[peak]], color="#e63946", linewidth=3)
            ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
            sagitta = float(row.get("auto_sagitta", float("nan")))
            ax.set_title(f"{row['sample_id']}  |  auto_sagitta={sagitta:.3f}", fontsize=10)
        else:
            ax.set_title(f"{row['sample_id']}  |  topline cikmadi (maske bozuk/kucuk)", fontsize=10)
    else:
        ax.set_title(f"{row['sample_id']}  |  maske yok", fontsize=10)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"saved: {out_path}")
    print(f"source_id: {row['source_id']}  view_hint: {row.get('view_hint','')}  "
          f"aspect_ratio: {row.get('aspect_ratio','')}  split: {row.get('split','')}")

if __name__ == "__main__":
    main()
