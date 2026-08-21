import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cowarch.label_studio import (
    build_label_studio_tasks,
    import_label_studio_results,
)


def make_manifest(path: Path, data_root: Path) -> None:
    crops = data_root / "prepared" / "crops"
    crops.mkdir(parents=True)
    for name in ("posture.jpg", "geometry.jpg", "rejected.jpg"):
        (crops / name).write_bytes(b"image-placeholder")
    pd.DataFrame(
        {
            "sample_id": ["posture", "geometry", "rejected"],
            "source_id": ["src"] * 3,
            "video_id": ["v1", "v2", "v3"],
            "crop_path": [str(crops / name) for name in ("posture.jpg", "geometry.jpg", "rejected.jpg")],
            "accepted": [True, True, False],
            "split": ["train"] * 3,
            "label": ["", "arched", ""],
            "keypoints_json": ["", "", ""],
        }
    ).to_csv(path, index=False)


class LabelStudioBridgeTests(unittest.TestCase):
    def test_task_exports_keep_passes_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest = data_root / "manifest.csv"
            make_manifest(manifest, data_root)
            posture_path = root / "posture.json"
            geometry_path = root / "geometry.json"

            posture_summary = build_label_studio_tasks(
                manifest, posture_path, phase="posture", data_root=data_root
            )
            geometry_summary = build_label_studio_tasks(
                manifest, geometry_path, phase="geometry", data_root=data_root
            )
            posture = json.loads(posture_path.read_text())
            geometry = json.loads(geometry_path.read_text())

        self.assertEqual(posture_summary["tasks_written"], 1)
        self.assertEqual(geometry_summary["tasks_written"], 1)
        self.assertEqual(posture[0]["data"]["sample_id"], "posture")
        self.assertEqual(geometry[0]["data"]["sample_id"], "geometry")
        self.assertEqual(
            posture[0]["data"]["image"],
            "/data/local-files/?d=prepared/crops/posture.jpg",
        )
        self.assertNotIn("label", posture[0]["data"])

    def test_approved_source_gate_rejects_unapproved_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest = data_root / "manifest.csv"
            make_manifest(manifest, data_root)
            sources = root / "sources.csv"
            pd.DataFrame(
                {
                    "source_id": ["src"],
                    "license_status": ["unresolved"],
                }
            ).to_csv(sources, index=False)

            with self.assertRaisesRegex(ValueError, "non-approved"):
                build_label_studio_tasks(
                    manifest,
                    root / "tasks.json",
                    phase="posture",
                    data_root=data_root,
                    approved_sources_path=sources,
                )

    def test_posture_import_writes_a_new_manifest_and_audit_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest = data_root / "manifest.csv"
            make_manifest(manifest, data_root)
            export = root / "export.json"
            output = root / "merged.csv"
            export.write_text(
                json.dumps(
                    [
                        {
                            "data": {"sample_id": "posture"},
                            "annotations": [
                                {
                                    "completed_by": {"email": "vet@example.test"},
                                    "result": [
                                        {
                                            "from_name": "posture",
                                            "type": "choices",
                                            "value": {"choices": ["normal"]},
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                )
            )

            summary = import_label_studio_results(
                manifest, export, output, phase="posture"
            )
            original = pd.read_csv(manifest, keep_default_na=False)
            merged = pd.read_csv(output, keep_default_na=False)

        self.assertEqual(summary["annotations_imported"], 1)
        self.assertEqual(original.loc[0, "label"], "")
        self.assertEqual(merged.loc[0, "label"], "normal")
        self.assertEqual(merged.loc[0, "posture_reviewed_by"], "vet@example.test")
        self.assertEqual(merged.loc[0, "geometry_reviewed_by"], "")

    def test_geometry_import_converts_percentages_in_anatomical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest = data_root / "manifest.csv"
            make_manifest(manifest, data_root)
            export = root / "export.json"
            output = root / "merged.csv"
            names = ["head", "sacrum", "withers"]
            results = []
            for index, name in enumerate(names):
                results.append(
                    {
                        "from_name": "dorsal_keypoints",
                        "type": "keypointlabels",
                        "original_width": 200,
                        "original_height": 100,
                        "value": {
                            "x": 50 - index * 10,
                            "y": 20 + index * 5,
                            "keypointlabels": [name],
                        },
                    }
                )
            export.write_text(
                json.dumps(
                    [
                        {
                            "data": {"sample_id": "geometry"},
                            "annotations": [
                                {"completed_by": 7, "result": results}
                            ],
                        }
                    ]
                )
            )

            import_label_studio_results(
                manifest, export, output, phase="geometry", reviewer="geometry-r"
            )
            merged = pd.read_csv(output, keep_default_na=False)
            row = merged.loc[merged["sample_id"].eq("geometry")].iloc[0]
            points = json.loads(row["keypoints_json"])

        self.assertEqual(points, [[60.0, 30.0], [80.0, 25.0], [100.0, 20.0]])
        self.assertEqual(row["label"], "arched")
        self.assertEqual(row["geometry_reviewed_by"], "geometry-r")

    def test_import_refuses_source_manifest_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            manifest = data_root / "manifest.csv"
            make_manifest(manifest, data_root)
            export = root / "export.json"
            export.write_text("[]")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                import_label_studio_results(
                    manifest, export, manifest, phase="posture"
                )


if __name__ == "__main__":
    unittest.main()
