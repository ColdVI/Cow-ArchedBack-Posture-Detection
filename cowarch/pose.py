"""Full-body cattle pose annotation schema.

The production posture measurement remains the three-anchor
``withers,sacrum,head`` contract in :mod:`cowarch.anchors`.  This module is an
independent annotation schema for experiments that also need a sampled dorsal
curve and distal limb motion.  Keeping the schemas separate prevents a new
gait experiment from silently changing the validated v1 measurement input.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence


POSE_SCHEMA = "cow_pose_19_v1"

# Click order is part of the stored-data contract.  Never reorder an existing
# schema; introduce a new POSE_SCHEMA instead.
POSE_KEYPOINTS = (
    "poll",
    "neck_mid",
    "withers",
    "dorsal_25",
    "dorsal_50",
    "dorsal_75",
    "sacrum",
    "near_front_carpus",
    "near_front_fetlock",
    "near_front_hoof",
    "far_front_carpus",
    "far_front_fetlock",
    "far_front_hoof",
    "near_hind_hock",
    "near_hind_fetlock",
    "near_hind_hoof",
    "far_hind_hock",
    "far_hind_fetlock",
    "far_hind_hoof",
)

# Zero-based edges used only for visualization.  Geometry is computed from the
# dense segmented topline and its production anchors, not from these lines.
POSE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3), (3, 4), (4, 5), (5, 6),
    (7, 8), (8, 9),
    (10, 11), (11, 12),
    (13, 14), (14, 15),
    (16, 17), (17, 18),
)

VISIBILITY_NAMES = {
    0: "not_labeled",
    1: "occluded",
    2: "visible",
}


def pose_prefilter_reason(record: dict) -> str:
    """Return an auditable reason a detector crop is not first-pass pose work.

    This is deliberately a queue-ordering heuristic, not ground truth.  It
    removes border cuts, tiny/low-confidence fragments and clearly non-lateral
    boxes before a human starts clicking.  ``--include-noncandidates`` in the
    labeler can still expose every accepted crop.
    """
    truthy = str(record.get("accepted", "")).strip().lower() in {
        "1", "true", "yes", "y"
    }
    if not truthy:
        return "not_accepted"
    touches_border = str(record.get("touches_border", "")).strip().lower() in {
        "1", "true", "yes", "y"
    }
    if touches_border:
        return "touches_border"
    try:
        confidence = float(record.get("detection_conf", "nan"))
        area_ratio = float(record.get("bbox_area_ratio", "nan"))
        aspect_ratio = float(record.get("aspect_ratio", "nan"))
    except (TypeError, ValueError):
        return "missing_detection_geometry"
    if not all(math.isfinite(value) for value in (confidence, area_ratio, aspect_ratio)):
        return "missing_detection_geometry"
    if confidence < 0.40:
        return "low_detection_confidence"
    if area_ratio < 0.035:
        return "small_box"
    if not 1.10 <= aspect_ratio <= 4.0:
        return "non_lateral_box"
    return ""


def normalize_pose_points(
    points: Iterable[Sequence[float | int]],
) -> list[list[float | int]]:
    """Validate and normalize 19 ``[x, y, visibility]`` triplets.

    Visibility follows the COCO/YOLO convention: 0 is not labeled, 1 is
    labeled but occluded, and 2 is labeled and visible.  Coordinates for
    visibility 0 are canonicalized to zero so an accidental stale click cannot
    leak into training.
    """
    rows = list(points)
    if len(rows) != len(POSE_KEYPOINTS):
        raise ValueError(
            f"{POSE_SCHEMA} requires {len(POSE_KEYPOINTS)} keypoints, got {len(rows)}"
        )

    normalized: list[list[float | int]] = []
    for name, row in zip(POSE_KEYPOINTS, rows, strict=True):
        if len(row) != 3:
            raise ValueError(f"keypoint {name!r} must be [x, y, visibility]")
        x, y, visibility = row
        if isinstance(visibility, bool) or int(visibility) != visibility:
            raise ValueError(f"keypoint {name!r} has non-integer visibility")
        visibility = int(visibility)
        if visibility not in VISIBILITY_NAMES:
            raise ValueError(
                f"keypoint {name!r} visibility must be 0, 1 or 2"
            )
        if visibility == 0:
            normalized.append([0.0, 0.0, 0])
            continue
        x = float(x)
        y = float(y)
        if not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0:
            raise ValueError(
                f"keypoint {name!r} needs finite non-negative coordinates"
            )
        normalized.append([round(x, 3), round(y, 3), visibility])
    return normalized


def serialize_pose_points(
    points: Iterable[Sequence[float | int]],
) -> str:
    """Serialize pose points with an explicit schema/version wrapper."""
    payload = {
        "schema": POSE_SCHEMA,
        "keypoints": normalize_pose_points(points),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_pose_points(value: str) -> list[list[float | int]]:
    """Parse a stored full-pose annotation and reject schema drift."""
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError("pose annotation is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("pose annotation must be a JSON object")
    if payload.get("schema") != POSE_SCHEMA:
        raise ValueError(
            f"pose annotation schema must be {POSE_SCHEMA!r}, "
            f"got {payload.get('schema')!r}"
        )
    return normalize_pose_points(payload.get("keypoints", []))


def geometry_keypoints_visible(points: Iterable[Sequence[float | int]]) -> bool:
    """Return whether withers and sacrum have usable coordinates."""
    normalized = normalize_pose_points(points)
    return bool(normalized[2][2] > 0 and normalized[6][2] > 0)
