import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from cowarch.labeling import (
    DorsalGeometryLabeler,
    PostureLabeler,
    ensure_annotation_columns,
)


def make_manifest(path: Path) -> None:
    pd.DataFrame(
        {
            "sample_id": ["normal", "arched", "uncertain", "unlabeled", "done"],
            "source_id": ["src"] * 5,
            "video_id": [f"v{i}" for i in range(5)],
            "crop_path": [f"crop-{i}.jpg" for i in range(5)],
            "accepted": [True] * 5,
            "split": ["train"] * 5,
            "label": ["normal", "arched", "uncertain", "", "normal"],
            "keypoints_json": ["", "", "", "", "[[0, 0]]"],
            "reviewed_by": ["legacy"] * 5,
        }
    ).to_csv(path, index=False)


class LabelingTests(unittest.TestCase):
    def test_legacy_reviewer_is_migrated_without_removing_old_column(self):
        frame = pd.DataFrame(
            {
                "label": ["arched", "normal", ""],
                "keypoints_json": ["[[0, 0]]", "", ""],
                "reviewed_by": ["old-a", "old-b", "old-c"],
            }
        )

        migrated = ensure_annotation_columns(frame)

        self.assertEqual(migrated.loc[0, "posture_reviewed_by"], "old-a")
        self.assertEqual(migrated.loc[0, "geometry_reviewed_by"], "old-a")
        self.assertEqual(migrated.loc[0, "annotation_pass"], "geometry")
        self.assertEqual(migrated.loc[1, "posture_reviewed_by"], "old-b")
        self.assertEqual(migrated.loc[1, "annotation_pass"], "posture")
        self.assertEqual(migrated.loc[2, "posture_reviewed_by"], "")
        self.assertIn("reviewed_by", migrated.columns)

    def test_validation_and_test_cannot_use_active_or_sequential_order(self):
        for split, order in [("val", "priority"), ("test", "sequential")]:
            with self.subTest(split=split, order=order):
                with self.assertRaisesRegex(ValueError, "random order"):
                    PostureLabeler("unused.csv", split, "reviewer", order=order)

    def test_deprecated_keypoint_argument_does_not_enable_posture_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            make_manifest(path)
            with patch.object(PostureLabeler, "_build_ui"):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    labeler = PostureLabeler(
                        path, "train", "posture-r", collect_keypoints=True
                    )

        self.assertFalse(hasattr(labeler, "points"))
        self.assertTrue(any(item.category is DeprecationWarning for item in caught))

    def test_posture_save_writes_posture_audit_fields_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            make_manifest(path)
            with patch.object(PostureLabeler, "_build_ui"):
                labeler = PostureLabeler(path, "train", "posture-r", order="sequential")
            labeler._advance = Mock()
            labeler._on_label("arched")

            saved = pd.read_csv(path, keep_default_na=False)

        row = saved.loc[saved["sample_id"].eq("unlabeled")].iloc[0]
        self.assertEqual(row["label"], "arched")
        self.assertEqual(row["posture_reviewed_by"], "posture-r")
        self.assertEqual(row["reviewed_by"], "posture-r")
        self.assertEqual(row["geometry_reviewed_by"], "")
        self.assertEqual(row["annotation_pass"], "posture")

    def test_geometry_queue_requires_completed_binary_posture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            make_manifest(path)
            with patch.object(DorsalGeometryLabeler, "_build_ui"):
                labeler = DorsalGeometryLabeler(
                    path, "train", "geometry-r", order="sequential"
                )

        queued = labeler.frame.loc[labeler.queue, "sample_id"].tolist()
        self.assertEqual(queued, ["normal", "arched"])

    def test_geometry_save_preserves_posture_and_uses_separate_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            make_manifest(path)
            with patch.object(DorsalGeometryLabeler, "_build_ui"):
                labeler = DorsalGeometryLabeler(
                    path, "train", "geometry-r", order="sequential"
                )
            labeler.status = SimpleNamespace(value="")
            labeler._advance = Mock()
            labeler.points = [(0, 10), (2, 8), (4, 7), (6, 8), (8, 10)]
            labeler._on_save()

            saved = pd.read_csv(path, keep_default_na=False)

        row = saved.loc[saved["sample_id"].eq("normal")].iloc[0]
        self.assertEqual(row["label"], "normal")
        self.assertEqual(row["reviewed_by"], "legacy")
        self.assertEqual(row["posture_reviewed_by"], "legacy")
        self.assertEqual(row["geometry_reviewed_by"], "geometry-r")
        self.assertEqual(row["annotation_pass"], "geometry")
        self.assertEqual(len(json.loads(row["keypoints_json"])), 5)
        self.assertIsNotNone(labeler.last_saved_features)

    def test_anchor_mode_saves_two_points_without_changing_posture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            make_manifest(path)
            with patch.object(DorsalGeometryLabeler, "_build_ui"):
                labeler = DorsalGeometryLabeler(
                    path, "train", "anchor-r", mode="anchors", order="sequential"
                )
            labeler.status = SimpleNamespace(value="")
            labeler._advance = Mock()
            labeler.points = [(10, 12), (80, 14)]
            labeler._on_save()

            saved = pd.read_csv(path, keep_default_na=False)

        row = saved.loc[saved["sample_id"].eq("normal")].iloc[0]
        self.assertEqual(row["label"], "normal")
        self.assertEqual(json.loads(row["anchors_json"]), [[10, 12], [80, 14]])
        self.assertEqual(row["geometry_reviewed_by"], "anchor-r")


if __name__ == "__main__":
    unittest.main()
