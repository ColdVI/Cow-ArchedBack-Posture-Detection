#!/usr/bin/env python3
"""Keyboard-first Pass A posture labeling.

Geometry capture is deliberately absent. Use ``DorsalGeometryLabeler`` in
``notebooks/03_labeling.ipynb`` as Pass B after posture decisions are saved.
"""
from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.io import as_bool, atomic_write_csv, read_manifest, resolve_data_path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the labeling UI") from exc
    return cv2


def random_indices(indices: list[int], seed: int) -> list[int]:
    result = indices.copy()
    np.random.default_rng(seed).shuffle(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keyboard-first Pass A posture labeling (image-only, random order).",
        epilog=(
            "Pass B geometry: open notebooks/03_labeling.ipynb and run "
            "DorsalGeometryLabeler after posture labeling."
        ),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument(
        "--reviewer",
        default=getpass.getuser(),
        help="posture reviewer written to posture_reviewed_by (default: OS user)",
    )
    parser.add_argument(
        "--order",
        choices=["random", "auto-sagitta"],
        default="random",
        help="random is the only supported Pass A order; auto-sagitta is retained to fail clearly",
    )
    parser.add_argument(
        "--keypoints",
        action="store_true",
        help="deprecated combined workflow; exits with Pass B migration guidance",
    )
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.keypoints:
        parser.error(
            "--keypoints can no longer be combined with posture labeling. "
            "Complete Pass A here, then use DorsalGeometryLabeler in "
            "notebooks/03_labeling.ipynb for Pass B."
        )
    if args.order == "auto-sagitta":
        parser.error(
            "--order auto-sagitta is disabled because geometry must not influence "
            "Pass A posture decisions. Use random order here; train-only active "
            "ordering is available in notebooks/03_labeling.ipynb."
        )
    if not str(args.reviewer).strip():
        parser.error("--reviewer must not be empty")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    from cowarch.labeling import POSTURE_LABELS, ensure_annotation_columns

    cv2 = require_cv2()
    frame = ensure_annotation_columns(read_manifest(args.manifest))
    if "split" not in frame.columns:
        frame["split"] = ""
    eligible_mask = as_bool(frame["accepted"]) & frame["split"].eq(args.split)
    if not args.include_labeled:
        eligible_mask &= frame["label"].astype(str).str.strip().eq("")
    indices = random_indices(list(frame.index[eligible_mask]), args.seed)
    if not indices:
        print("No samples match the requested split/filter.")
        return

    reviewer = str(args.reviewer).strip()
    window = "Cow posture labeling - Pass A"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cursor = 0
    history: list[int] = []

    while 0 <= cursor < len(indices):
        row_index = indices[cursor]
        row = frame.loc[row_index]
        image_path = resolve_data_path(str(row["crop_path"]), args.manifest)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable crop without assigning a label: {image_path}")
            cursor += 1
            continue

        height, width = image.shape[:2]
        scale = min(1.0, args.max_width / width, args.max_height / height)

        while True:
            display = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            # Pass A deliberately overlays controls and progress only: no row
            # identifier, split, prior label, keypoint, geometry, or score.
            lines = [
                f"Pass A  {cursor + 1}/{len(indices)}",
                "A arched | N normal | U uncertain | X invalid | S skip | B back | Q quit",
            ]
            for line_number, text in enumerate(lines):
                y = 28 + line_number * 28
                cv2.putText(
                    display,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (20, 20, 20),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    text,
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                atomic_write_csv(frame, args.manifest)
                cv2.destroyAllWindows()
                print(f"saved Pass A progress to {args.manifest}")
                return
            if key == ord("b"):
                if history:
                    cursor = history.pop()
                    break
            elif key == ord("s"):
                history.append(cursor)
                cursor += 1
                break
            elif key in (ord("a"), ord("n"), ord("u"), ord("x")):
                label_map = {
                    ord("a"): "arched",
                    ord("n"): "normal",
                    ord("u"): "uncertain",
                    ord("x"): "invalid",
                }
                next_label = label_map[key]
                if next_label not in POSTURE_LABELS:  # pragma: no cover - invariant
                    raise RuntimeError(f"unsupported posture label: {next_label}")
                frame.at[row_index, "label"] = next_label
                frame.at[row_index, "posture_reviewed_by"] = reviewer
                # Keep old manifest consumers working while new code uses the
                # pass-specific reviewer fields.
                frame.at[row_index, "reviewed_by"] = reviewer
                frame.at[row_index, "annotation_pass"] = "posture"
                atomic_write_csv(frame, args.manifest)
                history.append(cursor)
                cursor += 1
                break

    atomic_write_csv(frame, args.manifest)
    cv2.destroyAllWindows()
    print(f"Pass A labeling complete; saved {args.manifest}")


if __name__ == "__main__":
    main()
