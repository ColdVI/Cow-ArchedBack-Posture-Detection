"""Shared per-frame preparation logic.

``scripts/02_prepare.py`` and the inspection notebooks both call
:func:`process_frame`. It never touches disk, so a notebook can render exactly
what the batch backend would have written without producing any files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import bbox_crop, predict_cows, resize_mask
from .frames import dhash, hamming_distance, require_cv2, safe_token
from .geometry import auto_topline_features

REJECT_REASONS = (
    "no_cow",
    "multiple_cows",
    "bbox_area",
    "touches_border",
    "aspect_ratio",
    "empty_crop",
    "near_duplicate",
    "overlaps_accepted",
)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class PrepareConfig:
    """Every filter threshold in one auditable object."""

    confidence: float = 0.35
    padding: float = 0.03
    min_area_ratio: float = 0.05
    max_area_ratio: float = 0.90
    side_aspect: float = 1.40
    hard_side_filter: bool = False
    reject_border: bool = False
    dedup_hamming: int = 4

    def as_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "padding": self.padding,
            "min_area_ratio": self.min_area_ratio,
            "max_area_ratio": self.max_area_ratio,
            "side_aspect": self.side_aspect,
            "hard_side_filter": self.hard_side_filter,
            "reject_border": self.reject_border,
            "dedup_hamming": self.dedup_hamming,
        }


@dataclass
class FrameOutcome:
    """Result of processing one frame, including the intermediates to inspect."""

    record: dict
    frame: np.ndarray
    detections: list = field(default_factory=list)
    crop: np.ndarray | None = None
    mask: np.ndarray | None = None
    crop_box: tuple[int, int, int, int] | None = None
    box: np.ndarray | None = None
    hash_value: int | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.record.get("accepted"))

    @property
    def reject_reason(self) -> str:
        return str(self.record.get("reject_reason", ""))


def blank_record(source_id: str, video_id: str, frame_idx: int, original_name: str) -> dict:
    return {
        "sample_id": f"{safe_token(source_id)}_{frame_idx:08d}",
        "source_id": source_id,
        "video_id": video_id,
        "frame_idx": int(frame_idx),
        "original_name": original_name,
        "crop_path": "",
        "mask_path": "",
        "accepted": False,
        "reject_reason": "",
        "split": "",
        "label": "",
        "keypoints_json": "",
        "anchors_json": "",
        "reviewed_by": "",
        "posture_reviewed_by": "",
        "geometry_reviewed_by": "",
        "annotation_pass": "",
        "farm_id": "",
        "cow_id": "",
        "passage_id": "",
        "camera_id": "",
    }


def detect_frame_cows(
    frame: np.ndarray,
    *,
    detector,
    cow_class: int | None,
    confidence: float,
) -> list:
    """Return inspectable cow candidates without accepting or rejecting a frame."""
    frame_height, frame_width = frame.shape[:2]
    if detector is None:
        return [
            (
                np.array([0, 0, frame_width, frame_height], dtype=float),
                1.0,
                None,
            )
        ]
    if cow_class is None:
        raise ValueError("cow_class is required when a detector is enabled")
    return predict_cows(detector, cow_class, frame, confidence)


def process_frame(
    frame: np.ndarray,
    record: dict,
    *,
    detector,
    cow_class: int | None,
    config: PrepareConfig,
    last_hash: int | None = None,
    detection_index: int | None = None,
    detections: list | None = None,
) -> FrameOutcome:
    """Run detection, quality filters and silhouette geometry for one frame.

    ``detector=None`` treats the whole frame as an already-cropped cow, which is
    how the pipeline consumes hand-prepared crops.

    ``detections`` lets a caller pass in an already-computed candidate list
    (see :func:`process_frame_all_cows`) so a crowded frame is only run through
    the detector once, not once per animal.
    """
    cv2 = require_cv2()
    frame_height, frame_width = frame.shape[:2]
    outcome = FrameOutcome(record=record, frame=frame)

    if detections is None:
        detections = detect_frame_cows(
            frame,
            detector=detector,
            cow_class=cow_class,
            confidence=config.confidence,
        )
    outcome.detections = detections
    record["cow_count"] = len(detections)

    if detection_index is None and len(detections) != 1:
        record["reject_reason"] = "no_cow" if not detections else "multiple_cows"
        return outcome

    if detection_index is not None:
        if isinstance(detection_index, bool) or not isinstance(detection_index, (int, np.integer)):
            raise ValueError("detection_index must be an integer or None")
        if not 0 <= int(detection_index) < len(detections):
            raise IndexError(
                f"detection_index {detection_index} is out of range for {len(detections)} cows"
            )
        selected_index = int(detection_index)
    else:
        selected_index = 0

    record["selected_detection_index"] = selected_index
    box, score, raw_mask = detections[selected_index]
    outcome.box = np.asarray(box, dtype=float)
    x1, y1, x2, y2 = [float(value) for value in box]
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    bbox_area_ratio = (bbox_width * bbox_height) / float(frame_width * frame_height)
    aspect_ratio = bbox_width / max(bbox_height, 1e-6)
    touches_border = x1 <= 1 or y1 <= 1 or x2 >= frame_width - 1 or y2 >= frame_height - 1
    record.update(
        {
            "detection_conf": score,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_area_ratio": bbox_area_ratio,
            "aspect_ratio": aspect_ratio,
            "touches_border": touches_border,
            "view_hint": "side_candidate" if aspect_ratio >= config.side_aspect else "review_oblique",
        }
    )

    if detector is not None and not config.min_area_ratio <= bbox_area_ratio <= config.max_area_ratio:
        record["reject_reason"] = "bbox_area"
        return outcome
    if detector is not None and config.reject_border and touches_border:
        record["reject_reason"] = "touches_border"
        return outcome
    if config.hard_side_filter and aspect_ratio < config.side_aspect:
        record["reject_reason"] = "aspect_ratio"
        return outcome

    crop, crop_box = bbox_crop(frame, box, config.padding)
    if crop.size == 0:
        record["reject_reason"] = "empty_crop"
        return outcome
    outcome.crop = crop
    outcome.crop_box = crop_box

    hash_value = dhash(crop)
    outcome.hash_value = hash_value
    record["dhash"] = f"{hash_value:016x}"
    if last_hash is not None and hamming_distance(hash_value, last_hash) <= config.dedup_hamming:
        record["reject_reason"] = "near_duplicate"
        return outcome

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    record["blur_laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if raw_mask is not None:
        full_mask = resize_mask(raw_mask, (frame_height, frame_width))
        left, top, right, bottom = crop_box
        crop_mask = full_mask[top:bottom, left:right]
        outcome.mask = crop_mask
        record["mask_area_ratio"] = float(full_mask.mean())
        record.update(auto_topline_features(crop_mask))
    else:
        record.update(
            {
                "mask_area_ratio": np.nan,
                "auto_sagitta": np.nan,
                "auto_chord_rmse": np.nan,
                "auto_circle_curvature_norm": np.nan,
            }
        )

    record["accepted"] = True
    record["reject_reason"] = ""
    return outcome


def process_frame_all_cows(
    frame: np.ndarray,
    record_factory,
    *,
    detector,
    cow_class: int | None,
    config: PrepareConfig,
    last_hash: int | None = None,
    overlap_iou_max: float = 0.3,
) -> list[FrameOutcome]:
    """Crop every detected cow in a frame instead of rejecting the whole frame.

    ``02_prepare.py``'s default behavior treats "more than one cow" as a reject
    reason, which throws away individually clean crops whenever two animals
    happen to share a frame (a passage camera catching them side by side).
    Detection still runs once; each candidate is then filtered and cropped with
    the exact same per-detection logic ``process_frame`` uses for a single cow,
    so quality standards don't loosen just because more animals are visible.

    ``record_factory(cow_index)`` builds the blank record for one candidate;
    the caller controls sample_id/cow_id naming (e.g. append ``_cow{index}``).

    Two safeguards keep occluded/crowded scenes from flooding the output with
    fragments of the same animal rather than genuinely separate ones:

    - **Overlap suppression**: a candidate whose box overlaps an
      already-accepted box in *this same frame* by more than
      ``overlap_iou_max`` is rejected as ``overlaps_accepted`` before it is
      even cropped. Occluding rails routinely split one cow into several
      YOLO boxes; without this, each fragment would be written out as if it
      were an independent animal.
    - **Near-duplicate dedup** (inherited from ``process_frame``) still runs
      sequentially across whatever survives overlap suppression, catching the
      case where the *same* uncropped animal reappears nearly unchanged a few
      frames later.
    """
    detections = detect_frame_cows(
        frame, detector=detector, cow_class=cow_class, confidence=config.confidence
    )
    if not detections:
        record = record_factory(0)
        record["cow_count"] = 0
        record["reject_reason"] = "no_cow"
        return [FrameOutcome(record=record, frame=frame, detections=detections)]

    outcomes = []
    accepted_boxes: list[np.ndarray] = []
    running_hash = last_hash
    for index in range(len(detections)):
        record = record_factory(index)
        candidate_box = np.asarray(detections[index][0], dtype=float)
        overlap = max((box_iou(candidate_box, box) for box in accepted_boxes), default=0.0)
        if overlap > overlap_iou_max:
            record["cow_count"] = len(detections)
            record["reject_reason"] = "overlaps_accepted"
            outcomes.append(FrameOutcome(record=record, frame=frame, detections=detections, box=candidate_box))
            continue
        outcome = process_frame(
            frame,
            record,
            detector=detector,
            cow_class=cow_class,
            config=config,
            last_hash=running_hash,
            detection_index=index,
            detections=detections,
        )
        if outcome.accepted:
            running_hash = outcome.hash_value
            accepted_boxes.append(candidate_box)
        outcomes.append(outcome)
    return outcomes
