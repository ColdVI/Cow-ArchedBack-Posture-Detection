#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.geometry import DORSAL_KEYPOINTS, decode_keypoints
from cowarch.io import as_bool, atomic_write_csv, read_manifest, resolve_data_path


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the labeling UI") from exc
    return cv2


@dataclass
class UIState:
    points: list[tuple[float, float]] = field(default_factory=list)
    scale: float = 1.0
    keypoint_mode: bool = False


def encoded_points(points: list[tuple[float, float]]) -> str:
    payload = [
        {"name": name, "x": round(float(x), 2), "y": round(float(y), 2)}
        for name, (x, y) in zip(DORSAL_KEYPOINTS, points)
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def ordered_indices(frame, indices: list[int], order: str, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if order == "random":
        result = indices.copy()
        rng.shuffle(result)
        return result
    values = frame.loc[indices, "auto_sagitta"].replace("", np.nan).astype(float)
    ranks = values.rank(pct=True, na_option="bottom")
    priority = (ranks - 0.5).abs()
    return list(priority.sort_values(ascending=False).index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard-first posture and dorsal-keypoint labeling UI.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--order", choices=["random", "auto-sagitta"], default="random")
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--include-labeled", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-width", type=int, default=1400)
    parser.add_argument("--max-height", type=int, default=900)
    args = parser.parse_args()

    if args.order == "auto-sagitta" and args.split != "train":
        raise ValueError("auto-sagitta ordering is allowed only for the training split")

    cv2 = require_cv2()
    frame = read_manifest(args.manifest)
    for column in ["label", "keypoints_json", "split"]:
        if column not in frame.columns:
            frame[column] = ""
    eligible_mask = as_bool(frame["accepted"]) & frame["split"].eq(args.split)
    if not args.include_labeled:
        eligible_mask &= frame["label"].astype(str).str.strip().eq("")
    indices = list(frame.index[eligible_mask])
    if not indices:
        print("No samples match the requested split/filter.")
        return
    if args.order == "auto-sagitta" and "auto_sagitta" not in frame.columns:
        raise ValueError("Manifest has no auto_sagitta column")
    indices = ordered_indices(frame, indices, args.order, args.seed)

    state = UIState(keypoint_mode=args.keypoints)
    window = "Cow posture labeling"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and state.keypoint_mode and len(state.points) < 5:
            state.points.append((x / state.scale, y / state.scale))

    cv2.setMouseCallback(window, mouse_callback)
    cursor = 0
    history: list[int] = []

    while 0 <= cursor < len(indices):
        row_index = indices[cursor]
        row = frame.loc[row_index]
        image_path = resolve_data_path(str(row["crop_path"]), args.manifest)
        image = cv2.imread(str(image_path))
        if image is None:
            frame.at[row_index, "label"] = "invalid"
            atomic_write_csv(frame, args.manifest)
            cursor += 1
            continue

        existing = decode_keypoints(row.get("keypoints_json", ""))
        if not state.points and existing is not None:
            state.points = [tuple(point) for point in existing]
        height, width = image.shape[:2]
        state.scale = min(1.0, args.max_width / width, args.max_height / height)

        while True:
            display = cv2.resize(
                image,
                (int(round(width * state.scale)), int(round(height * state.scale))),
                interpolation=cv2.INTER_AREA,
            )
            for number, point in enumerate(state.points):
                px = int(round(point[0] * state.scale))
                py = int(round(point[1] * state.scale))
                cv2.circle(display, (px, py), 6, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    str(number + 1),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            label = str(row.get("label", "")) or "unlabeled"
            mode = "KP ON" if state.keypoint_mode else "KP OFF"
            lines = [
                f"{cursor + 1}/{len(indices)}  {row['sample_id']}  split={args.split}",
                f"label={label}  {mode}  points={len(state.points)}/5",
                "A arched | N normal | U uncertain | X invalid | S skip | B back | K keypoints | R reset | Q quit",
            ]
            for line_number, text in enumerate(lines):
                y = 28 + line_number * 28
                cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                atomic_write_csv(frame, args.manifest)
                cv2.destroyAllWindows()
                print(f"saved progress to {args.manifest}")
                return
            if key == ord("k"):
                state.keypoint_mode = not state.keypoint_mode
            elif key == ord("r"):
                state.points = []
            elif key in (8, 127):
                if state.points:
                    state.points.pop()
            elif key == ord("b"):
                if history:
                    cursor = history.pop()
                    state.points = []
                    break
            elif key == ord("s"):
                history.append(cursor)
                cursor += 1
                state.points = []
                break
            elif key in (ord("a"), ord("n"), ord("u"), ord("x")):
                label_map = {
                    ord("a"): "arched",
                    ord("n"): "normal",
                    ord("u"): "uncertain",
                    ord("x"): "invalid",
                }
                next_label = label_map[key]
                if state.keypoint_mode and next_label in {"arched", "normal"} and len(state.points) != 5:
                    continue
                frame.at[row_index, "label"] = next_label
                if len(state.points) == 5:
                    frame.at[row_index, "keypoints_json"] = encoded_points(state.points)
                elif next_label in {"uncertain", "invalid"}:
                    frame.at[row_index, "keypoints_json"] = ""
                atomic_write_csv(frame, args.manifest)
                history.append(cursor)
                cursor += 1
                state.points = []
                break

    atomic_write_csv(frame, args.manifest)
    cv2.destroyAllWindows()
    print(f"labeling complete; saved {args.manifest}")


if __name__ == "__main__":
    main()
