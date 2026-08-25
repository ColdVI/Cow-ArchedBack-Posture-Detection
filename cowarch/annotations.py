"""Shared annotation schema helpers without notebook or plotting dependencies."""
from __future__ import annotations

import pandas as pd

POSTURE_LABELS = ("arched", "normal", "uncertain", "invalid")
LABEL_COLORS = {
    "arched": "#d1495b",
    "normal": "#2e86ab",
    "uncertain": "#f4a259",
    "invalid": "#6c757d",
}
ANNOTATION_COLUMNS = (
    "label",
    "keypoints_json",
    "anchors_json",
    "pose_keypoints_json",
    "pose_schema",
    "pose_status",
    "pose_candidate",
    "pose_prefilter_reason",
    "reviewed_by",
    "posture_reviewed_by",
    "geometry_reviewed_by",
    "pose_reviewed_by",
    "annotation_pass",
)


def ensure_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add two-pass columns and interpret legacy ``reviewed_by`` values."""
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    annotation_values = frame[list(ANNOTATION_COLUMNS)]
    frame[list(ANNOTATION_COLUMNS)] = annotation_values.where(
        annotation_values.notna(), ""
    ).astype(object)

    label = frame["label"].astype(str).str.strip()
    legacy_reviewer = frame["reviewed_by"].astype(str).str.strip()
    posture_reviewer = frame["posture_reviewed_by"].astype(str).str.strip()
    geometry_reviewer = frame["geometry_reviewed_by"].astype(str).str.strip()
    has_posture = label.isin(POSTURE_LABELS)
    has_geometry = (
        frame["keypoints_json"].astype(str).str.strip().ne("")
        | frame["anchors_json"].astype(str).str.strip().ne("")
    )

    migrate_posture = has_posture & posture_reviewer.eq("") & legacy_reviewer.ne("")
    migrate_geometry = has_geometry & geometry_reviewer.eq("") & legacy_reviewer.ne("")
    frame.loc[migrate_posture, "posture_reviewed_by"] = legacy_reviewer[migrate_posture]
    frame.loc[migrate_geometry, "geometry_reviewed_by"] = legacy_reviewer[migrate_geometry]

    missing_pass = frame["annotation_pass"].astype(str).str.strip().eq("")
    frame.loc[missing_pass & has_posture, "annotation_pass"] = "posture"
    frame.loc[missing_pass & has_geometry, "annotation_pass"] = "geometry"
    return frame
