#!/usr/bin/env python3
"""Discover local cow media and prepare a full-pose annotation round.

Files are referenced in place rather than copied.  Generated crops, masks,
inventory and manifests live under ``--output-dir`` (normally ignored data).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.annotations import ensure_annotation_columns
from cowarch.io import atomic_write_csv
from cowarch.pose import POSE_SCHEMA, pose_prefilter_reason
from cowarch.sources import SOURCE_COLUMNS


VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "media"


def infer_ir_image(path: Path) -> tuple[bool, str]:
    """Flag near-grayscale CCTV images while keeping the inference auditable."""
    try:
        from PIL import Image
        import numpy as np

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((640, 360))
            pixels = np.asarray(rgb, dtype=np.int16)
    except Exception as exc:
        return False, f"image_inference_failed:{type(exc).__name__}"
    chroma = np.ptp(pixels, axis=2)
    colored_fraction = float(np.mean(chroma > 12))
    if colored_fraction < 0.02:
        return True, "low_chroma_image"
    return False, "color_image"


def discover_media(input_dir: Path, source_prefix: str) -> pd.DataFrame:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_dir}")
    files = sorted(
        path.resolve()
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES | IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"no supported video/image files found under {input_dir}")

    used: dict[str, int] = {}
    rows = []
    for path in files:
        base = safe_id(f"{source_prefix}_{path.stem}")
        collision = used.get(base, 0)
        used[base] = collision + 1
        source_id = base if collision == 0 else f"{base}_{collision + 1}"
        kind = "video" if path.suffix.lower() in VIDEO_SUFFIXES else "image"
        row = {column: "" for column in SOURCE_COLUMNS}
        row.update(
            {
                "source_id": source_id,
                "kind": kind,
                "local_path": str(path),
                "license_status": "approved",
                "license_name": "Wellztech internal field media",
                "farm_id": "wellztech_field",
                "video_id": source_id,
                "passage_id": source_id if kind == "video" else "",
                "camera_role": "passage_detection",
                "notes": (
                    "Full-pose annotation candidate; not measurement-camera "
                    "ground truth unless separately reviewed."
                ),
            }
        )
        if kind == "image":
            row["is_ir"], row["ir_inference"] = infer_ir_image(path)
        else:
            row["is_ir"] = False
            row["ir_inference"] = "not_inferred_video"
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local cow videos/images for 19-keypoint annotation."
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-prefix", default="wellztech")
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=0,
        help="Optional accepted-crop cap; zero covers the complete source.",
    )
    parser.add_argument("--model", default="yolo11m-seg.pt")
    parser.add_argument("--detector", choices=("yolo", "none"), default="yolo")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    sources_path = output_dir / "sources.csv"
    manifest_path = output_dir / "manifest.csv"
    prepared_dir = output_dir / "prepared"
    protected = [sources_path, manifest_path]
    if not args.overwrite and any(path.exists() for path in protected):
        raise FileExistsError(
            f"annotation round already exists under {output_dir}; "
            "use a new output directory or pass --overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_media(args.input_dir.resolve(), args.source_prefix)
    atomic_write_csv(sources, sources_path)
    print(f"inventory: {len(sources)} media files -> {sources_path}")
    if args.inventory_only:
        return

    prepare_script = Path(__file__).resolve().with_name("02_prepare.py")
    command = [
        sys.executable,
        str(prepare_script),
        "--sources", str(sources_path),
        "--output-dir", str(prepared_dir),
        "--manifest", str(manifest_path),
        "--target-fps", str(args.target_fps),
        "--model", args.model,
        "--detector", args.detector,
        "--confidence", str(args.confidence),
        "--padding", str(args.padding),
        "--min-area-ratio", "0.02",
        "--max-area-ratio", "0.98",
        "--side-aspect", "1.20",
        "--dedup-hamming", "-1",
        "--allow-fragmented-mask",
        "--all-cows",
        "--overlap-iou-max", "0.30",
    ]
    if args.max_per_source > 0:
        command.extend(["--max-per-source", str(args.max_per_source)])
    subprocess.run(command, check=True)

    manifest = ensure_annotation_columns(
        pd.read_csv(manifest_path, keep_default_na=False)
    )
    manifest["pose_schema"] = manifest["pose_schema"].replace("", POSE_SCHEMA)
    manifest["pose_prefilter_reason"] = [
        pose_prefilter_reason(row) for row in manifest.to_dict(orient="records")
    ]
    manifest["pose_candidate"] = manifest["pose_prefilter_reason"].eq("")
    atomic_write_csv(manifest, manifest_path)
    accepted = manifest["accepted"].astype(str).str.lower().isin({"true", "1", "yes"})
    queued = accepted & manifest["pose_candidate"].astype(bool)
    print(
        f"pose round ready: {int(accepted.sum())} accepted crops / "
        f"{len(manifest)} candidates -> {manifest_path}"
    )
    print(f"first-pass labeling queue: {int(queued.sum())} crops")
    print(
        "label with: python scripts/16_label_pose.py "
        f"--manifest {manifest_path} --reviewer <name>"
    )


if __name__ == "__main__":
    main()
