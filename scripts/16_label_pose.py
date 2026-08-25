#!/usr/bin/env python3
"""Mouse-first 19-keypoint cattle pose labeling on prepared cow crops.

Left click marks a visible point, right click marks an inferable-but-occluded
point, and ``0`` records the current point as not labeled.  Every action advances
to the next anatomical name automatically.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.annotations import ensure_annotation_columns
from cowarch.io import as_bool, atomic_write_csv, read_manifest, resolve_data_path
from cowarch.pose import POSE_EDGES, POSE_KEYPOINTS, POSE_SCHEMA, parse_pose_points, serialize_pose_points


HUD_HEIGHT = 86


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the full-pose labeling UI") from exc
    return cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mouse-first full-pose annotation for accepted cow crops."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reviewer", default=getpass.getuser())
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Also include crops already labeled or explicitly skipped.",
    )
    parser.add_argument(
        "--include-noncandidates",
        action="store_true",
        help="Include accepted crops rejected by the automatic pose prefilter.",
    )
    parser.add_argument("--order", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=1500)
    parser.add_argument("--max-height", type=int, default=900)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the queue size without opening an annotation window.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reviewer = str(args.reviewer).strip()
    if not reviewer:
        raise ValueError("reviewer must not be empty")
    if args.max_width < 320 or args.max_height <= HUD_HEIGHT + 100:
        raise ValueError("max-width/max-height are too small for the labeling UI")

    frame = ensure_annotation_columns(read_manifest(args.manifest))
    selected = as_bool(frame["accepted"])
    has_pose_prefilter = frame["pose_candidate"].astype(str).str.strip().ne("").any()
    if has_pose_prefilter and not args.include_noncandidates:
        selected &= as_bool(frame["pose_candidate"])
    if args.split != "all":
        selected &= frame["split"].astype(str).eq(args.split)
    if not args.include_reviewed:
        selected &= frame["pose_keypoints_json"].astype(str).str.strip().eq("")
        selected &= frame["pose_status"].astype(str).str.strip().eq("")
    indices = list(frame.index[selected])
    if args.order == "random":
        np.random.default_rng(args.seed).shuffle(indices)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    if args.dry_run:
        print(f"full-pose queue: {len(indices)} crops from {args.manifest}")
        return
    if not indices:
        print("No crops match the requested filters.")
        return

    existing_paths = [
        resolve_data_path(str(frame.at[index, "crop_path"]), args.manifest)
        for index in indices
    ]
    if not any(path.is_file() for path in existing_paths):
        example = existing_paths[0]
        raise FileNotFoundError(
            f"None of the {len(existing_paths)} selected crop files exists. "
            f"Example missing path: {example}. The manifest may have been "
            "generated on another machine; run scripts/19_ingest_pose_media.py "
            "locally to create portable crop paths and images."
        )

    cv2 = require_cv2()
    window = "Cow full pose - 19 keypoints"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cursor = 0
    current_points: list[list[float | int]] = []
    current_scale = 1.0

    def load_existing(row_index: int) -> list[list[float | int]]:
        value = str(frame.at[row_index, "pose_keypoints_json"]).strip()
        return parse_pose_points(value) if value else []

    def on_mouse(event, x, y, _flags, _param) -> None:
        nonlocal current_points
        if len(current_points) >= len(POSE_KEYPOINTS):
            return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if y < HUD_HEIGHT:
            return
        visibility = 2 if event == cv2.EVENT_LBUTTONDOWN else 1
        current_points.append(
            [
                round(float(x) / current_scale, 3),
                round(float(y - HUD_HEIGHT) / current_scale, 3),
                visibility,
            ]
        )

    cv2.setMouseCallback(window, on_mouse)

    while 0 <= cursor < len(indices):
        row_index = int(indices[cursor])
        row = frame.loc[row_index]
        image_path = resolve_data_path(str(row["crop_path"]), args.manifest)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable crop: {image_path}")
            cursor += 1
            current_points = []
            continue
        height, width = image.shape[:2]
        current_scale = min(
            args.max_width / width,
            (args.max_height - HUD_HEIGHT) / height,
        )
        display_width = max(1, int(round(width * current_scale)))
        display_image_height = max(1, int(round(height * current_scale)))
        cv2.resizeWindow(
            window,
            display_width,
            display_image_height + HUD_HEIGHT,
        )
        if not current_points:
            current_points = load_existing(row_index)

        while True:
            resized = cv2.resize(
                image,
                (display_width, display_image_height),
                interpolation=(
                    cv2.INTER_CUBIC if current_scale > 1.0 else cv2.INTER_AREA
                ),
            )
            display = cv2.copyMakeBorder(
                resized,
                HUD_HEIGHT,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(20, 27, 34),
            )
            scaled_points = [
                (
                    int(round(float(x) * current_scale)),
                    int(round(float(y) * current_scale)) + HUD_HEIGHT,
                    int(v),
                )
                for x, y, v in current_points
            ]
            for left, right in POSE_EDGES:
                if left < len(scaled_points) and right < len(scaled_points):
                    x1, y1, v1 = scaled_points[left]
                    x2, y2, v2 = scaled_points[right]
                    if v1 > 0 and v2 > 0:
                        cv2.line(display, (x1, y1), (x2, y2), (255, 210, 80), 2, cv2.LINE_AA)
            for point_index, (x, y, visibility) in enumerate(scaled_points):
                if visibility == 0:
                    continue
                color = (80, 220, 80) if visibility == 2 else (0, 165, 255)
                thickness = -1 if visibility == 2 else 2
                cv2.circle(display, (x, y), 6, color, thickness, cv2.LINE_AA)
                cv2.putText(
                    display,
                    str(point_index + 1),
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            next_name = (
                POSE_KEYPOINTS[len(current_points)]
                if len(current_points) < len(POSE_KEYPOINTS)
                else "READY_TO_SAVE"
            )
            lines = [
                f"{cursor + 1}/{len(indices)}  {len(current_points)}/{len(POSE_KEYPOINTS)}  NEXT: {next_name}",
                "LEFT visible | RIGHT occluded | 0 missing | Z/U undo | R reset",
                "ENTER save+next | N skip image | B previous | Q quit",
            ]
            for line_number, text in enumerate(lines):
                y = 26 + line_number * 25
                cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), ord("Q"), 27):
                atomic_write_csv(frame, args.manifest)
                cv2.destroyAllWindows()
                print(f"saved full-pose progress to {args.manifest}")
                return
            if key in (ord("0"), ord("m"), ord("M")) and len(current_points) < len(POSE_KEYPOINTS):
                current_points.append([0.0, 0.0, 0])
            elif key in (ord("z"), ord("Z"), ord("u"), ord("U"), 8, 127):
                if current_points:
                    current_points.pop()
            elif key in (ord("r"), ord("R")):
                current_points = []
            elif key in (10, 13):
                if len(current_points) != len(POSE_KEYPOINTS):
                    continue
                frame.at[row_index, "pose_keypoints_json"] = serialize_pose_points(current_points)
                frame.at[row_index, "pose_schema"] = POSE_SCHEMA
                frame.at[row_index, "pose_status"] = "labeled"
                frame.at[row_index, "pose_reviewed_by"] = reviewer
                atomic_write_csv(frame, args.manifest)
                cursor += 1
                current_points = []
                break
            elif key in (ord("n"), ord("N")):
                frame.at[row_index, "pose_status"] = "skipped"
                frame.at[row_index, "pose_reviewed_by"] = reviewer
                atomic_write_csv(frame, args.manifest)
                cursor += 1
                current_points = []
                break
            elif key in (ord("b"), ord("B")):
                cursor = max(0, cursor - 1)
                current_points = []
                break

    atomic_write_csv(frame, args.manifest)
    cv2.destroyAllWindows()
    print(f"full-pose annotation complete; saved {args.manifest}")


if __name__ == "__main__":
    main()
