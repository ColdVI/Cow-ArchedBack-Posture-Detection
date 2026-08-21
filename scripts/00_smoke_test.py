#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the arched-back PoC pipeline.

Generates fake side-view cow silhouettes with a known dorsal arch, then runs
prepare -> split -> train -> report. This validates plumbing, split hygiene and
the feature contract WITHOUT any real data, YOLO weights or Torch.

It is a plumbing check only. Metrics produced here are meaningless as evidence
about real cattle and must never appear in the report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cowarch.geometry import auto_topline_features  # noqa: E402
from cowarch.io import atomic_write_csv, read_manifest  # noqa: E402

WIDTH, HEIGHT = 480, 320
N_KEYPOINTS = 5


def dorsal_y(t: np.ndarray, base_y: float, arch: float) -> np.ndarray:
    """Dorsal line height. Image y grows downward, so an arch subtracts."""
    return base_y - arch * np.sin(np.pi * np.clip(t, 0.0, 1.0))


def render_cow(arch: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (BGR frame, 5x2 dorsal keypoints, exact silhouette mask).

    The mask is drawn from the same polygons as the frame rather than recovered
    by thresholding. Thresholding here would be a second thing that can be wrong,
    and a broken mask would silently invalidate the geometry panels.
    """
    hide = int(rng.integers(60, 200))
    bg = int(rng.integers(90, 170))
    frame = np.full((HEIGHT, WIDTH, 3), bg, dtype=np.uint8)
    frame += rng.integers(-12, 12, frame.shape, dtype=np.int16).astype(np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    x0 = float(rng.integers(60, 90))
    x1 = float(WIDTH - rng.integers(60, 90))
    base_y = float(rng.integers(120, 150))
    belly_y = base_y + float(rng.integers(85, 105))

    xs = np.linspace(x0, x1, 120)
    t = (xs - x0) / (x1 - x0)
    top = dorsal_y(t, base_y, arch)
    bottom = np.full_like(top, belly_y) - 8.0 * np.sin(np.pi * t)

    body = np.concatenate(
        [np.stack([xs, top], axis=1), np.stack([xs[::-1], bottom[::-1]], axis=1)]
    ).astype(np.int32)
    cv2.fillPoly(frame, [body], (hide, hide, hide))
    cv2.fillPoly(mask, [body], 1)

    # legs and head, so the silhouette is not a bare blob
    for lx in (x0 + 25, x0 + 55, x1 - 55, x1 - 25):
        leg_bottom = int(belly_y) + int(rng.integers(45, 60))
        corners = ((int(lx) - 7, int(belly_y) - 10), (int(lx) + 7, leg_bottom))
        cv2.rectangle(frame, *corners, (hide, hide, hide), -1)
        cv2.rectangle(mask, *corners, 1, -1)
    head_y = int(base_y + rng.integers(-10, 25))
    cv2.ellipse(frame, (int(x0) - 22, head_y), (26, 16), 0, 0, 360, (hide, hide, hide), -1)
    cv2.ellipse(mask, (int(x0) - 22, head_y), (26, 16), 0, 0, 360, 1, -1)

    kp_t = np.linspace(0.0, 1.0, N_KEYPOINTS)
    kp_x = x0 + kp_t * (x1 - x0)
    kp_y = dorsal_y(kp_t, base_y, arch)
    keypoints = np.stack([kp_x, kp_y], axis=1)
    # annotator jitter: a human does not click the exact silhouette pixel
    keypoints += rng.normal(0.0, 2.0, keypoints.shape)
    return frame, keypoints, mask.astype(bool)


def build_dataset(root: Path, n_groups: int, per_group: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    raw = root / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    rows, truth = [], []

    for group in range(n_groups):
        source_id = f"smoke_{group:02d}"
        # A group is one "video" holding several animals, so both classes occur
        # inside every group. Single-class groups would make a group-wise split
        # degenerate, which is exactly the failure this test must not mask.
        out_dir = raw / source_id
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = raw / "_masks" / source_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        prevalence = float(rng.uniform(0.35, 0.65))
        posture_counts = {"arched": 0, "normal": 0}

        for index in range(per_group):
            arched = bool(rng.random() < prevalence)
            label = "arched" if arched else "normal"
            passage_id = (
                f"{source_id}_{label}_passage_{posture_counts[label] // 2:03d}"
            )
            posture_counts[label] += 1
            base = float(rng.uniform(26.0, 40.0) if arched else rng.uniform(0.0, 8.0))
            arch = max(0.0, base + float(rng.normal(0.0, 3.5)))
            frame, keypoints, mask = render_cow(arch, rng)
            name = f"{source_id}_{index:04d}.png"
            cv2.imwrite(str(out_dir / name), frame)
            mask_path = mask_dir / f"{source_id}_{index:04d}.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            truth.append(
                {
                    "sample_id": f"{source_id}_{index:08d}",
                    "label": label,
                    # Each synthetic passage contains one posture and usually
                    # two frames, matching grouped-evaluation's GT contract.
                    "passage_id": passage_id,
                    "mask_path": str(mask_path.resolve()),
                    "keypoints_json": json.dumps(
                        [[round(float(x), 2), round(float(y), 2)] for x, y in keypoints]
                    ),
                }
            )
        rows.append(
            {
                "source_id": source_id,
                "kind": "image_dir",
                "url": "",
                "local_path": str(out_dir.resolve()),
                "license": "synthetic",
                "source_score": "",
                "notes": "SYNTHETIC smoke-test data, not real cattle",
            }
        )

    sources = root / "sources_smoke.csv"
    pd.DataFrame(rows).to_csv(sources, index=False)
    pd.DataFrame(truth).to_csv(root / "truth_smoke.csv", index=False)
    return sources


def run(step: str, argv: list[str]) -> None:
    print(f"\n=== {step} ===", flush=True)
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "smoke")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "smoke")
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--per-group", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    sources = build_dataset(args.root, args.groups, args.per_group, args.seed)
    manifest = args.root / "manifest_smoke.csv"

    run(
        "02_prepare",
        [
            "scripts/02_prepare.py",
            "--sources", str(sources),
            "--output-dir", str(args.root / "prepared"),
            "--manifest", str(manifest),
            "--detector", "none",
            "--dedup-hamming", "0",
        ],
    )

    # Stand in for what a segmentation checkpoint would have produced, so the
    # mask-dependent panels and auto_* features have something to work on.
    frame = read_manifest(manifest)
    truth = pd.read_csv(args.root / "truth_smoke.csv").set_index("sample_id")
    frame["mask_path"] = frame["sample_id"].map(truth["mask_path"]).fillna("")
    auto_rows = []
    for _, row in frame.iterrows():
        path = str(row["mask_path"]).strip()
        raw_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None
        if raw_mask is None:
            auto_rows.append({k: np.nan for k in
                              ["auto_sagitta", "auto_chord_rmse", "auto_circle_curvature_norm"]})
            continue
        auto_rows.append(auto_topline_features(raw_mask > 127))
    for column in ["auto_sagitta", "auto_chord_rmse", "auto_circle_curvature_norm"]:
        frame[column] = [row[column] for row in auto_rows]

    # Inject the synthetic labels and keypoints that a human would supply in F2/F3.
    frame["label"] = frame["sample_id"].map(truth["label"]).fillna("")
    frame["passage_id"] = frame["sample_id"].map(truth["passage_id"]).fillna("")
    frame["keypoints_json"] = frame["sample_id"].map(truth["keypoints_json"]).fillna("")
    frame["reviewed_by"] = "synthetic"
    atomic_write_csv(frame, manifest)
    usable = int(frame["mask_path"].astype(str).str.strip().ne("").sum())
    print(f"injected synthetic labels for {int((frame['label'] != '').sum())} rows, "
          f"{usable} masks, auto_sagitta median={frame['auto_sagitta'].median():.4f}")

    run("03_split", ["scripts/03_split.py", "--manifest", str(manifest), "--group-column", "video_id"])
    run(
        "05_train",
        [
            "scripts/05_train.py",
            "--manifest", str(manifest),
            "--output-dir", str(args.run_dir),
            "--models", "geometry",
            "--allow-legacy-keypoints",
            # Synthetic fixtures predate RFID/cow identity. Production defaults
            # remain cow_id; this legacy plumbing check opts into video groups.
            "--group-column", "video_id",
        ],
    )
    run(
        "06_report",
        [
            "scripts/06_report.py",
            "--run-dir", str(args.run_dir),
            "--manifest", str(manifest),
            "--group-column", "passage_id",
            "--allow-legacy-keypoints",
        ],
    )

    print("\nSMOKE TEST PASSED - plumbing only. These numbers are not evidence.")


if __name__ == "__main__":
    main()
