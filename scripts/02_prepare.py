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

from cowarch.anchors import UltralyticsAnchorPredictor
from cowarch.camera import load_calibration, undistort_image
from cowarch.detect import load_detector
from cowarch.frames import iter_frames, probe_source
from cowarch.io import atomic_write_csv
from cowarch.prepare import PrepareConfig, blank_record, frame_timestamp_utc, process_frame


def scalar_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cow crops and an auditable manifest.")
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-fps", type=float, default=1.0)
    parser.add_argument("--model", default="yolo11m-seg.pt")
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
    parser.add_argument(
        "--measurement-stream",
        action="store_true",
        help="Process consecutive high-fps measurement frames without deduplication.",
    )
    parser.add_argument(
        "--anchor-model",
        default="",
        help="Custom 3-keypoint YOLO-pose checkpoint ordered withers,sacrum,head.",
    )
    parser.add_argument("--anchor-device", default=None)
    parser.add_argument("--anchor-confidence", type=float, default=0.5)
    parser.add_argument("--head-drop-max-norm", type=float, default=0.30)
    parser.add_argument("--center-band-fraction", type=float, default=0.50)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        help="Plumb-line calibration JSON produced by 14_calibrate_camera.py.",
    )
    args = parser.parse_args()

    if args.target_fps <= 0:
        raise ValueError("target-fps must be positive")
    if not 0 < args.center_band_fraction <= 1:
        raise ValueError("center-band-fraction must be in (0, 1]")
    if args.measurement_stream and args.target_fps < 20:
        raise ValueError("measurement-stream requires target-fps >= 20 for passage aggregation")
    if args.measurement_stream and not args.anchor_model:
        raise ValueError("measurement-stream requires --anchor-model")

    import cv2

    config = PrepareConfig(
        confidence=args.confidence,
        padding=args.padding,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        side_aspect=args.side_aspect,
        hard_side_filter=args.hard_side_filter,
        reject_border=args.reject_border,
        dedup_hamming=-1 if args.measurement_stream else args.dedup_hamming,
        anchor_confidence=args.anchor_confidence,
        head_drop_max_norm=args.head_drop_max_norm,
        center_band_fraction=args.center_band_fraction,
    )
    sources = pd.read_csv(args.sources, keep_default_na=False, comment="#")
    required = {"source_id", "kind"}
    if missing := sorted(required - set(sources.columns)):
        raise ValueError(f"sources CSV is missing columns: {missing}")
    if "resolved_path" not in sources.columns and "local_path" not in sources.columns:
        raise ValueError("sources CSV needs resolved_path or local_path")
    if "camera_role" in sources.columns:
        sources["camera_role"] = (
            sources["camera_role"].astype(str).str.strip().str.lower()
        )
        invalid_roles = sorted(
            set(sources.loc[sources["camera_role"].ne(""), "camera_role"])
            - {"measurement", "passage_detection"}
        )
        if invalid_roles:
            raise ValueError(f"Unsupported camera_role values: {invalid_roles}")
    if args.anchor_model or args.camera_calibration is not None:
        if "camera_role" not in sources.columns:
            raise ValueError(
                "Anchor measurement and camera calibration require "
                "camera_role='measurement' in sources CSV"
            )
        measurement_rows = sources["camera_role"].eq("measurement")
        if not measurement_rows.any():
            raise ValueError(
                "Anchor model or camera calibration was supplied but no source has "
                "camera_role='measurement'"
            )
        if args.camera_calibration is not None and "camera_id" in sources.columns:
            measurement_cameras = {
                value
                for value in sources.loc[measurement_rows, "camera_id"].astype(str).str.strip()
                if value
            }
            if len(measurement_cameras) > 1:
                raise ValueError(
                    "One camera calibration cannot be applied to multiple measurement camera_id values"
                )
    if args.measurement_stream:
        sources = sources.loc[sources["camera_role"].eq("measurement")].copy()

    model_name = "none" if args.detector == "none" else args.model
    detector, cow_class = load_detector(model_name)
    anchor_predictor = (
        UltralyticsAnchorPredictor(args.anchor_model, device=args.anchor_device)
        if args.anchor_model
        else None
    )
    calibration = load_calibration(args.camera_calibration) if args.camera_calibration else None

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
        video_id = str(source.get("video_id", "")).strip() or source_id
        metadata = probe_source(path, kind)
        source_fps = metadata.get("fps")
        start_timestamp = source.get("start_timestamp_utc", "") or source.get(
            "timestamp_utc", ""
        )
        last_kept_hash: int | None = None
        kept_count = 0
        camera_role = str(source.get("camera_role", "")).strip() or "unspecified"
        source_anchor_predictor = anchor_predictor if camera_role == "measurement" else None
        source_calibration = calibration if camera_role == "measurement" else None

        for frame_idx, frame, original_name in iter_frames(path, kind, args.target_fps):
            if args.max_per_source and kept_count >= args.max_per_source:
                break
            if source_calibration is not None:
                frame = undistort_image(frame, source_calibration)
            record = blank_record(source_id, video_id, frame_idx, original_name)
            record["raw_source_path"] = str(path.resolve())
            record["timestamp_utc"] = frame_timestamp_utc(
                start_timestamp, frame_idx, source_fps
            )
            record["source_score"] = source.get("source_score", "")
            record["license"] = source.get("license", "")
            for column in (
                "license_status", "license_name", "license_url", "farm_id", "cow_id",
                "passage_id", "camera_id",
            ):
                record[column] = source.get(column, "")
            record["is_ir"] = scalar_bool(source.get("is_ir", False))
            record["camera_role"] = camera_role
            record["undistortion_applied"] = source_calibration is not None
            record["center_band_fraction"] = args.center_band_fraction
            mitigations = []
            if source_calibration is not None:
                mitigations.append("plumb_line_undistortion")
            if camera_role == "measurement" and args.center_band_fraction < 1.0:
                mitigations.append("center_band")
            record["camera_mitigation"] = "+".join(mitigations) or "none"

            outcome = process_frame(
                frame,
                record,
                detector=detector,
                cow_class=cow_class,
                config=config,
                last_hash=last_kept_hash,
                anchor_predictor=source_anchor_predictor,
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
