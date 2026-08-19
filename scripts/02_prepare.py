#!/usr/bin/env python3
"""Batch backend for frame extraction, cow cropping and the auditable manifest.

All per-frame logic lives in ``cowarch.prepare`` so the inspection notebooks run
exactly the same code path without writing anything to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.detect import load_detector
from cowarch.frames import iter_frames
from cowarch.io import atomic_write_csv
from cowarch.prepare import PrepareConfig, blank_record, process_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cow crops and an auditable manifest.")
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--detector", choices=["yolo", "none"], default="yolo")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--min-area-ratio", type=float, default=0.05)
    parser.add_argument("--max-area-ratio", type=float, default=0.90)
    parser.add_argument("--side-aspect", type=float, default=1.40)
    parser.add_argument("--hard-side-filter", action="store_true")
    parser.add_argument("--reject-border", action="store_true")
    parser.add_argument("--dedup-hamming", type=int, default=4)
    parser.add_argument("--max-per-source", type=int, default=0)
    args = parser.parse_args()

    if args.target_fps <= 0:
        raise ValueError("target-fps must be positive")

    import cv2

    config = PrepareConfig(
        confidence=args.confidence,
        padding=args.padding,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        side_aspect=args.side_aspect,
        hard_side_filter=args.hard_side_filter,
        reject_border=args.reject_border,
        dedup_hamming=args.dedup_hamming,
    )
    model_name = "none" if args.detector == "none" else args.model
    detector, cow_class = load_detector(model_name)

    sources = pd.read_csv(args.sources, keep_default_na=False, comment="#")
    required = {"source_id", "kind"}
    if missing := sorted(required - set(sources.columns)):
        raise ValueError(f"sources CSV is missing columns: {missing}")
    if "resolved_path" not in sources.columns and "local_path" not in sources.columns:
        raise ValueError("sources CSV needs resolved_path or local_path")

    crop_dir = args.output_dir / "crops"
    mask_dir = args.output_dir / "masks"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for _, source in sources.iterrows():
        source_id = str(source["source_id"]).strip()
        kind = str(source["kind"]).strip().lower()
        raw_path = str(source.get("resolved_path", "") or source.get("local_path", ""))
        path = Path(raw_path)
        if not path.is_absolute():
            path = args.sources.resolve().parent / path
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {path}")
        video_id = source_id
        last_kept_hash: int | None = None
        kept_count = 0

        for frame_idx, frame, original_name in iter_frames(path, kind, args.target_fps):
            if args.max_per_source and kept_count >= args.max_per_source:
                break
            record = blank_record(source_id, video_id, frame_idx, original_name)
            record["source_score"] = source.get("source_score", "")
            record["license"] = source.get("license", "")

            outcome = process_frame(
                frame,
                record,
                detector=detector,
                cow_class=cow_class,
                config=config,
                last_hash=last_kept_hash,
            )
            if not outcome.accepted:
                records.append(record)
                continue

            crop_path = crop_dir / f"{record['sample_id']}.jpg"
            if not cv2.imwrite(str(crop_path), outcome.crop):
                raise RuntimeError(f"Could not write crop: {crop_path}")
            record["crop_path"] = str(crop_path.resolve())

            if outcome.mask is not None:
                mask_path = mask_dir / f"{record['sample_id']}.png"
                cv2.imwrite(str(mask_path), outcome.mask.astype(np.uint8) * 255)
                record["mask_path"] = str(mask_path.resolve())

            records.append(record)
            last_kept_hash = outcome.hash_value
            kept_count += 1

        print(f"{source_id}: kept {kept_count} samples")

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise RuntimeError("No frames were processed")
    atomic_write_csv(manifest, args.manifest)
    accepted = int(manifest["accepted"].astype(bool).sum())
    print(f"wrote {len(manifest)} rows ({accepted} accepted) to {args.manifest}")


if __name__ == "__main__":
    main()
