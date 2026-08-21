#!/usr/bin/env python3
"""Export auditable annotated videos and frames from real local/public data.

Three evidence levels are kept visually distinct:

* local videos: YOLO segmentation and silhouette-derived topline estimates;
* Mendeley side views: the same model estimate plus existing human posture review;
* LivestockKeypoints: dataset-provided 18-point ground-truth anatomy.

The script never presents a silhouette endpoint as a withers/sacrum ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cowarch.geometry import chord_values, extract_topline
from cowarch.io import atomic_write_csv


DORSAL_KEYPOINT_NAMES = ("eye", "neck", "body1", "body2", "body3", "body4")


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
        thickness + 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def banner(image: np.ndarray, title: str, subtitle: str) -> None:
    height = 58
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    put_text(image, title, (12, 24), color=(80, 235, 255), scale=0.62, thickness=2)
    put_text(image, subtitle, (12, 48), color=(230, 230, 230), scale=0.46)


def fit_canvas(image: np.ndarray, width: int = 1280, height: int = 720):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas, scale, x0, y0


def iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    inter = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_left = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    area_right = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = area_left + area_right - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    box: np.ndarray
    age: int = 0


class IoUTracker:
    def __init__(self, threshold: float = 0.25, max_age: int = 12) -> None:
        self.threshold = threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, boxes: np.ndarray) -> list[int]:
        for track in self.tracks.values():
            track.age += 1
        assignments = [-1] * len(boxes)
        candidates = sorted(
            (
                (iou(box, track.box), detection, track_id)
                for detection, box in enumerate(boxes)
                for track_id, track in self.tracks.items()
            ),
            reverse=True,
        )
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        for overlap, detection, track_id in candidates:
            if overlap < self.threshold:
                break
            if detection in used_detections or track_id in used_tracks:
                continue
            assignments[detection] = track_id
            used_detections.add(detection)
            used_tracks.add(track_id)
        for detection, box in enumerate(boxes):
            if assignments[detection] < 0:
                assignments[detection] = self.next_id
                self.next_id += 1
            self.tracks[assignments[detection]] = Track(box=np.asarray(box, dtype=float), age=0)
        self.tracks = {
            track_id: track for track_id, track in self.tracks.items()
            if track.age <= self.max_age
        }
        return assignments


def track_color(track_id: int) -> tuple[int, int, int]:
    return (
        int(70 + (track_id * 83) % 170),
        int(70 + (track_id * 47) % 170),
        int(70 + (track_id * 131) % 170),
    )


def robust_topline(mask: np.ndarray, box: np.ndarray):
    """Return a smoothed fixed-trim silhouette and a conservative quality flag."""
    extracted = extract_topline(mask, trim=0.20, min_columns=20)
    if extracted is None:
        return None
    px, py = extracted
    window = min(21, len(py) if len(py) % 2 else len(py) - 1)
    window = max(3, window)
    radius = window // 2
    padded = np.pad(py, (radius, radius), mode="edge")
    median = np.asarray(
        [np.median(padded[index:index + window]) for index in range(len(py))]
    )
    smooth_window = min(11, len(median) if len(median) % 2 else len(median) - 1)
    smooth_window = max(3, smooth_window)
    smooth_radius = smooth_window // 2
    smoothed = np.convolve(
        np.pad(median, (smooth_radius, smooth_radius), mode="edge"),
        np.full(smooth_window, 1.0 / smooth_window),
        mode="valid",
    )
    chord = chord_values(px, smoothed)
    body_length = float(px[-1] - px[0])
    sagitta = float(np.max(chord - smoothed) / body_length)
    rmse = float(np.sqrt(np.mean(np.square(chord - smoothed))) / body_length)
    box_height = max(1.0, float(box[3] - box[1]))
    vertical_span = float(np.quantile(smoothed, 0.95) - np.quantile(smoothed, 0.05))
    upper_body = float(np.median(smoothed)) <= float(box[1]) + 0.58 * box_height
    # A true lateral dorsal curve occupies only a modest fraction of body
    # height. Larger excursions are typically a leg/head fragment or a rail
    # cutting the segmentation mask, and must not be drawn as anatomy.
    reliable = bool(vertical_span <= 0.18 * box_height and upper_body)
    return px, smoothed, chord, {
        "auto_sagitta": sagitta,
        "auto_chord_rmse": rmse,
        "body_length_px": body_length,
        "topline_reliable": reliable,
        "topline_vertical_span_fraction": vertical_span / box_height,
    }


def cow_class_id(model) -> int:
    matches = [int(index) for index, name in model.names.items() if str(name).lower() == "cow"]
    if len(matches) != 1:
        raise RuntimeError(f"Detector must expose exactly one cow class; found {matches}")
    return matches[0]


def suppress_contained_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: list[np.ndarray | None],
    *,
    containment_threshold: float = 0.80,
):
    """Drop smaller fragment boxes mostly contained by a larger cow box."""
    if len(boxes) < 2:
        return boxes, scores, masks
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    kept: list[int] = []
    for index in np.argsort(-areas):
        candidate = boxes[index]
        contained = False
        for existing_index in kept:
            existing = boxes[existing_index]
            x1, y1 = np.maximum(candidate[:2], existing[:2])
            x2, y2 = np.minimum(candidate[2:], existing[2:])
            intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if areas[index] > 0 and intersection / areas[index] >= containment_threshold:
                contained = True
                break
        if not contained:
            kept.append(int(index))
    kept.sort()
    return boxes[kept], scores[kept], [masks[index] for index in kept]


def model_detections(model, image: np.ndarray, *, confidence: float, device: str | None):
    result = model.predict(
        image, classes=[cow_class_id(model)], conf=confidence, imgsz=640,
        device=device, verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 4)), np.empty(0), []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    raw_masks = [] if result.masks is None else result.masks.data.detach().cpu().numpy()
    masks = []
    for index in range(len(boxes)):
        if index >= len(raw_masks):
            masks.append(None)
        else:
            masks.append(
                cv2.resize(
                    raw_masks[index].astype(np.float32),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
            )
    return suppress_contained_detections(boxes, scores, masks)


def draw_model_overlay(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: list[np.ndarray | None],
    track_ids: list[int],
) -> list[dict]:
    overlay = image.copy()
    metrics = []
    for box, score, mask, track_id in zip(boxes, scores, masks, track_ids):
        color = track_color(track_id)
        if mask is not None:
            overlay[mask] = (
                0.65 * overlay[mask] + 0.35 * np.asarray(color, dtype=float)
            ).astype(np.uint8)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    for box, score, mask, track_id in zip(boxes, scores, masks, track_ids):
        color = track_color(track_id)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        geometry = {
            "auto_sagitta": float("nan"),
            "auto_chord_rmse": float("nan"),
            "body_length_px": float("nan"),
            "topline_reliable": False,
            "topline_vertical_span_fraction": float("nan"),
        }
        if mask is not None:
            extracted = robust_topline(mask, box)
            if extracted is not None:
                px, py, chord_y, geometry = extracted
            if extracted is not None and geometry["topline_reliable"]:
                contour = np.rint(np.column_stack((px, py))).astype(np.int32)
                cv2.polylines(image, [contour], False, (0, 255, 255), 3, cv2.LINE_AA)
                chord = np.rint(np.column_stack((px, chord_y))).astype(np.int32)
                cv2.polylines(image, [chord], False, (255, 255, 255), 1, cv2.LINE_AA)
                for point, name in ((contour[0], "trim start"), (contour[-1], "trim end")):
                    cv2.circle(image, tuple(point), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    put_text(image, name, (int(point[0]) + 6, int(point[1]) - 6), scale=0.38)
        sagitta = geometry.get("auto_sagitta", float("nan"))
        sagitta_text = "n/a" if not math.isfinite(float(sagitta)) else f"{float(sagitta):.4f}"
        quality_text = "topline ok" if geometry["topline_reliable"] else "topline unreliable"
        label_y = max(78, y1 - 8)
        put_text(
            image,
            f"T{track_id} | cow {score:.2f} | sagitta {sagitta_text} | {quality_text}",
            (max(5, x1), label_y), color=color, scale=0.48, thickness=2,
        )
        metrics.append(
            {
                "track_id": track_id,
                "detection_confidence": float(score),
                "bbox_x1": float(box[0]), "bbox_y1": float(box[1]),
                "bbox_x2": float(box[2]), "bbox_y2": float(box[3]),
                **geometry,
            }
        )
    return metrics


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4: {path}")
    return writer


def export_local_videos(
    model,
    videos: list[Path],
    output_dir: Path,
    *,
    confidence: float,
    device: str | None,
    frame_stride: int,
) -> list[dict]:
    rows = []
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for video_index, source in enumerate(videos, start=1):
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unreadable video: {source}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output = output_dir / f"local_{video_index:02d}_tracked.mp4"
        writer = open_writer(output, fps, (width, height))
        tracker = IoUTracker()
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                boxes, scores, masks = model_detections(
                    model, frame, confidence=confidence, device=device
                )
                track_ids = tracker.update(boxes)
                metrics = draw_model_overlay(frame, boxes, scores, masks, track_ids)
                banner(
                    frame,
                    f"LOCAL REAL VIDEO {video_index} | frame {frame_index}/{max(0, total - 1)}",
                    "YOLO11n-seg tracking + fixed-trim silhouette estimate; NOT anatomical ground truth",
                )
                writer.write(frame)
                frame_file = ""
                if frame_index % frame_stride == 0:
                    frame_path = frames_dir / f"local_{video_index:02d}_{frame_index:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 91])
                    frame_file = str(frame_path.resolve())
                for item in metrics or [{}]:
                    rows.append(
                        {
                            "collection": "local_real_video",
                            "source": str(source.resolve()),
                            "source_frame": frame_index,
                            "output_video": str(output.resolve()),
                            "output_frame": frame_file,
                            "annotation_type": "model_prediction",
                            "license": "user-provided/local",
                            "posture_label": "",
                            **item,
                        }
                    )
                frame_index += 1
                if frame_index % 60 == 0:
                    print(f"local video {video_index}: {frame_index}/{total} frames", flush=True)
        finally:
            capture.release()
            writer.release()
        print(f"local video {video_index}: wrote {output} ({frame_index} frames)", flush=True)
    return rows


def posture_reviews(manifest_path: Path) -> dict[str, str]:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    reviewed = manifest.loc[
        manifest.get("license_status", "").astype(str).eq("approved")
        & manifest.get("label", "").astype(str).str.strip().ne("")
    ]
    result = {}
    for source_id, group in reviewed.groupby("source_id"):
        labels = sorted(set(group["label"].astype(str).str.strip()) - {""})
        result[str(source_id)] = "/".join(labels)
    return result


def export_mendeley(
    model,
    sources_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    confidence: float,
    device: str | None,
    fps: float,
) -> list[dict]:
    sources = pd.read_csv(sources_path, keep_default_na=False)
    sources = sources.loc[
        sources["license_status"].astype(str).eq("approved")
        & sources["kind"].astype(str).str.strip().str.lower().eq("image")
    ].copy()
    sources = sources.loc[sources["local_path"].map(lambda value: Path(value).is_file())].copy()
    if sources.empty:
        raise ValueError("No approved Mendeley image files were found")
    reviews = posture_reviews(manifest_path)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "mendeley_side_views_model_overlay.mp4"
    writer = open_writer(video_path, fps, (1280, 720))
    rows = []
    try:
        for index, source in enumerate(sources.itertuples(index=False), start=1):
            image = cv2.imread(str(source.local_path))
            if image is None:
                raise FileNotFoundError(f"Unreadable Mendeley image: {source.local_path}")
            frame, _, _, _ = fit_canvas(image)
            boxes, scores, masks = model_detections(
                model, frame, confidence=confidence, device=device
            )
            track_ids = list(range(1, len(boxes) + 1))
            metrics = draw_model_overlay(frame, boxes, scores, masks, track_ids)
            review = reviews.get(str(source.source_id), "not reviewed")
            banner(
                frame,
                f"MENDELEY SIDE VIEW {index}/{len(sources)} | posture review: {review}",
                "CC BY 4.0 | model mask/topline; posture label is human review when present",
            )
            writer.write(frame)
            frame_path = frames_dir / f"{index:03d}_{source.source_id}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 91])
            rows.append(
                {
                    "collection": "mendeley_side_views",
                    "source": str(Path(source.local_path).resolve()),
                    "source_frame": 0,
                    "output_video": str(video_path.resolve()),
                    "output_frame": str(frame_path.resolve()),
                    "annotation_type": "model_prediction+human_posture_review" if review != "not reviewed" else "model_prediction",
                    "license": "CC BY 4.0",
                    "posture_label": "" if review == "not reviewed" else review,
                    "source_id": source.source_id,
                    "detections": len(metrics),
                }
            )
            if index % 12 == 0:
                print(f"Mendeley: {index}/{len(sources)} images", flush=True)
    finally:
        writer.release()
    print(f"Mendeley: wrote {video_path} and {len(rows)} frames", flush=True)
    return rows


def transform_point(point: tuple[float, float], scale: float, x0: int, y0: int):
    return int(round(point[0] * scale + x0)), int(round(point[1] * scale + y0))


def add_keypoint_panel(
    image: np.ndarray,
    keypoint_names: tuple[str, ...],
    dorsal_chain: tuple[int, ...],
) -> np.ndarray:
    panel_width = 360
    output = np.full((image.shape[0], image.shape[1] + panel_width, 3), 24, dtype=np.uint8)
    output[:, :image.shape[1]] = image
    put_text(output, "DATASET GROUND TRUTH", (image.shape[1] + 14, 28), color=(80, 235, 255), scale=0.56, thickness=2)
    put_text(output, "18 anatomical landmarks", (image.shape[1] + 14, 51), scale=0.44)
    for index, name in enumerate(keypoint_names):
        y = 84 + index * 33
        color = (0, 220, 255) if index in dorsal_chain else (130, 220, 130)
        cv2.circle(output, (image.shape[1] + 22, y - 5), 7, color, -1, cv2.LINE_AA)
        put_text(output, f"{index + 1:02d}  {name}", (image.shape[1] + 38, y), color=color, scale=0.44)
    put_text(output, "yellow line: sparse dorsal chain", (image.shape[1] + 14, 690), color=(0, 220, 255), scale=0.41)
    return output


def draw_ground_truth(
    image: np.ndarray,
    annotation: dict,
    scale: float,
    x0: int,
    y0: int,
    *,
    skeleton: tuple[tuple[int, int], ...],
    dorsal_chain: tuple[int, ...],
):
    raw = np.asarray(annotation["keypoints"], dtype=float).reshape(-1, 3)
    points = [transform_point((x, y), scale, x0, y0) for x, y, _ in raw]
    visible = raw[:, 2] > 0
    for first, second in skeleton:
        if visible[first] and visible[second]:
            cv2.line(image, points[first], points[second], (100, 180, 100), 2, cv2.LINE_AA)
    dorsal_points = [points[index] for index in dorsal_chain if visible[index]]
    if len(dorsal_points) >= 2:
        cv2.polylines(
            image, [np.asarray(dorsal_points, dtype=np.int32)], False,
            (0, 220, 255), 4, cv2.LINE_AA,
        )
    for index, ((_, _, state), point) in enumerate(zip(raw, points)):
        if state <= 0:
            continue
        color = (0, 220, 255) if index in dorsal_chain else (130, 220, 130)
        if state == 1:
            cv2.circle(image, point, 7, color, 2, cv2.LINE_AA)
        else:
            cv2.circle(image, point, 6, color, -1, cv2.LINE_AA)
        put_text(image, str(index + 1), (point[0] + 7, point[1] - 5), color=color, scale=0.38, thickness=2)
    x, y, width, height = annotation["bbox"]
    p1 = transform_point((x, y), scale, x0, y0)
    p2 = transform_point((x + width, y + height), scale, x0, y0)
    cv2.rectangle(image, p1, p2, (180, 180, 180), 1)
    return raw


def export_livestock_ground_truth(root: Path, output_dir: Path, *, fps: float) -> list[dict]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "livestock_keypoints_ground_truth.mp4"
    writer = open_writer(video_path, fps, (1640, 720))
    rows = []
    frame_index = 0
    try:
        for batch in ("A", "B", "C"):
            annotation_path = root / batch / "annotations" / "_annotations.coco.json"
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotated_categories = [
                category for category in payload["categories"] if category.get("keypoints")
            ]
            if len(annotated_categories) != 1:
                raise ValueError(
                    f"Expected one keypoint category in {annotation_path}; "
                    f"found {len(annotated_categories)}"
                )
            category = annotated_categories[0]
            keypoint_names = tuple(str(name) for name in category["keypoints"])
            if not all(name in keypoint_names for name in DORSAL_KEYPOINT_NAMES):
                raise ValueError(
                    f"{annotation_path} lacks the expected dorsal points: "
                    f"{DORSAL_KEYPOINT_NAMES}"
                )
            dorsal_chain = tuple(keypoint_names.index(name) for name in DORSAL_KEYPOINT_NAMES)
            # COCO skeleton pairs are one-based keypoint indices.
            skeleton = tuple(
                (int(first) - 1, int(second) - 1)
                for first, second in category.get("skeleton", [])
            )
            if any(
                index < 0 or index >= len(keypoint_names)
                for edge in skeleton for index in edge
            ):
                raise ValueError(f"Invalid COCO skeleton indices in {annotation_path}")
            images = {int(item["id"]): item for item in payload["images"]}
            annotations = {int(item["image_id"]): item for item in payload["annotations"]}
            for image_id in sorted(images):
                info = images[image_id]
                annotation = annotations[image_id]
                source = root / batch / "images" / info["file_name"]
                image = cv2.imread(str(source))
                if image is None:
                    raise FileNotFoundError(f"Unreadable LivestockKeypoints image: {source}")
                frame, scale, x0, y0 = fit_canvas(image)
                raw = draw_ground_truth(
                    frame, annotation, scale, x0, y0,
                    skeleton=skeleton, dorsal_chain=dorsal_chain,
                )
                banner(
                    frame,
                    f"LIVESTOCKKEYPOINTS GT | batch {batch} | image {image_id}",
                    "18 COCO labels; yellow = eye-neck-body1-body2-body3-body4 dorsal chain",
                )
                rendered = add_keypoint_panel(frame, keypoint_names, dorsal_chain)
                writer.write(rendered)
                frame_path = frames_dir / f"{frame_index:04d}_{batch}_{image_id:04d}.jpg"
                cv2.imwrite(str(frame_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 91])
                rows.append(
                    {
                        "collection": "livestock_keypoints",
                        "source": str(source.resolve()),
                        "source_frame": image_id,
                        "output_video": str(video_path.resolve()),
                        "output_frame": str(frame_path.resolve()),
                        "annotation_type": "dataset_ground_truth_18_keypoints",
                        "license": "CC BY 4.0",
                        "posture_label": "",
                        "batch": batch,
                        "visible_keypoints": int((raw[:, 2] == 2).sum()),
                        "occluded_keypoints": int((raw[:, 2] == 1).sum()),
                        "keypoint_names": "|".join(keypoint_names),
                        "dorsal_chain": "|".join(DORSAL_KEYPOINT_NAMES),
                    }
                )
                frame_index += 1
                if frame_index % 100 == 0:
                    print(f"LivestockKeypoints: {frame_index} frames", flush=True)
    finally:
        writer.release()
    print(f"LivestockKeypoints: wrote {video_path} and {len(rows)} frames", flush=True)
    return rows


def write_contact_sheet(paths: list[Path], output: Path) -> None:
    selected = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is not None:
            selected.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    if not selected:
        return
    while len(selected) < 8:
        selected.append(np.full_like(selected[0], 25))
    sheet = np.vstack((np.hstack(selected[:4]), np.hstack(selected[4:8])))
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tracked/annotated real cattle media.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/annotated_real_data"))
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--video-frame-stride", type=int, default=15)
    parser.add_argument("--slideshow-fps", type=float, default=8.0)
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-mendeley", action="store_true")
    parser.add_argument("--skip-livestock", action="store_true")
    parser.add_argument("--sources-selected", type=Path, default=Path("data/sources_selected.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--livestock-root", type=Path,
        default=Path("data/external/Livestock-keypoint-detection/data_process/cattle"),
    )
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()
    if not 0 < args.confidence <= 1:
        raise ValueError("confidence must be in (0, 1]")
    if args.video_frame_stride < 1 or args.slideshow_fps <= 0:
        raise ValueError("frame stride and slideshow FPS must be positive")
    missing = [str(path) for path in args.videos if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Local videos not found: {missing}")

    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    all_rows = []

    def run_or_load(stage: str, skip: bool, callback):
        stage_manifest = args.output_dir / stage / "artifact_manifest.csv"
        if skip:
            if not stage_manifest.is_file():
                raise FileNotFoundError(
                    f"Cannot skip {stage}; its stage manifest does not exist: {stage_manifest}"
                )
            return pd.read_csv(stage_manifest, keep_default_na=False).to_dict("records")
        rows = callback()
        atomic_write_csv(pd.DataFrame(rows), stage_manifest)
        return rows

    all_rows += run_or_load(
        "local_videos", args.skip_local,
        lambda: export_local_videos(
            model, args.videos, args.output_dir / "local_videos",
            confidence=args.confidence, device=args.device,
            frame_stride=args.video_frame_stride,
        ),
    )
    all_rows += run_or_load(
        "mendeley_side_views", args.skip_mendeley,
        lambda: export_mendeley(
            model, args.sources_selected, args.manifest,
            args.output_dir / "mendeley_side_views",
            confidence=args.confidence, device=args.device, fps=args.slideshow_fps,
        ),
    )
    all_rows += run_or_load(
        "livestock_keypoints_gt", args.skip_livestock,
        lambda: export_livestock_ground_truth(
            args.livestock_root, args.output_dir / "livestock_keypoints_gt",
            fps=args.slideshow_fps,
        ),
    )
    atomic_write_csv(pd.DataFrame(all_rows), args.output_dir / "artifact_manifest.csv")

    candidates = []
    for video_index in range(1, len(args.videos) + 1):
        frames = sorted(
            (args.output_dir / "local_videos" / "frames").glob(
                f"local_{video_index:02d}_*.jpg"
            )
        )
        if frames:
            candidates.append(frames[len(frames) // 2])
    mendeley_frames = sorted(
        (args.output_dir / "mendeley_side_views" / "frames").glob("*.jpg")
    )
    livestock_frames = sorted(
        (args.output_dir / "livestock_keypoints_gt" / "frames").glob("*.jpg")
    )
    if mendeley_frames:
        candidates.append(mendeley_frames[len(mendeley_frames) // 2])
    if livestock_frames:
        candidates.extend(
            [livestock_frames[len(livestock_frames) // 3], livestock_frames[2 * len(livestock_frames) // 3]]
        )
    write_contact_sheet(candidates, args.output_dir / "preview_contact_sheet.jpg")
    readme = """# Annotated real-data export

## Evidence levels

- `local_videos/`: YOLO11n-seg tracking, mask and fixed-trim upper-silhouette estimate. These are model predictions, not anatomical ground truth.
- `mendeley_side_views/`: CC BY 4.0 side-view images with the same model overlay. Existing human posture reviews are shown where available; they are visual posture labels, not veterinary diagnoses.
- `livestock_keypoints_gt/`: dataset-provided 18-point ground truth read directly from each COCO category. The yellow dorsal chain follows `eye,neck,body1,body2,body3,body4`. The dataset does not label exact v3 withers/sacrum points, so no such claim is made.

## Attribution

- Yiwei Wang et al., LivestockKeypoints v1.0, CC BY 4.0: https://github.com/yww0411/Livestock-keypoint-detection
- Lili Bai, Cattle side view and back view dataset, CC BY 4.0, DOI 10.17632/h2s22wr5py.1: https://data.mendeley.com/datasets/h2s22wr5py/1

Derived overlays were generated by this repository. See `artifact_manifest.csv` for per-frame provenance and annotation type.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
