import json
import unittest

import numpy as np
import pandas as pd

from cowarch.passages import aggregate_passages


def frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "passage_id": ["p1"] * 5 + ["p2"],
            "cow_id": ["cow1"] * 6,
            "camera_id": ["cam1"] * 6,
            "timestamp_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:02Z",
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:04Z",
                "2026-01-02T00:00:00Z",
            ],
            "accepted": [True, True, False, True, False, True],
            "measurement_accepted": [True, True, False, True, False, True],
            "measurement_reject_reason": ["", "", "blur", "", "fragmented_mask", ""],
            "reject_reason": ["", "", "blur", "", "fragmented_mask", ""],
            "auto_sagitta": [0.1, 0.2, np.nan, 0.9, np.nan, 0.3],
            "auto_chord_rmse": [0.01, 0.02, np.nan, 0.09, np.nan, 0.03],
            "auto_circle_curvature_norm": [1.0, 2.0, np.nan, 9.0, np.nan, 3.0],
            "anchored_sagitta_signed_norm": [0.1, 0.2, np.nan, 0.9, np.nan, 0.3],
            "anchored_chord_rmse_norm": [0.01, 0.02, np.nan, 0.09, np.nan, 0.03],
            "anchored_peak_position": [0.4, 0.5, np.nan, 0.6, np.nan, 0.5],
            "detection_conf": [0.9, 0.8, np.nan, 0.7, np.nan, 1.0],
            "is_ir": [False] * 6,
            "pipeline_version": ["v1"] * 6,
            "camera_mitigation": ["plumb_line_undistortion+center_band"] * 6,
            "undistortion_applied": [True] * 6,
            "center_band_fraction": [0.5] * 6,
        }
    )


class PassageTests(unittest.TestCase):
    def test_robust_statistics_transit_and_rejects(self):
        out = aggregate_passages(frame_rows(), min_quality=0.4, min_valid_frames=3)
        row = out.set_index("passage_id").loc["p1"]
        self.assertEqual(row["n_frames_total"], 5)
        self.assertEqual(row["n_frames_valid"], 3)
        self.assertAlmostEqual(row["sagitta_median"], 0.2)
        self.assertAlmostEqual(row["sagitta_iqr"], 0.4)
        self.assertAlmostEqual(row["sagitta_p90"], 0.76)
        self.assertAlmostEqual(row["transit_seconds"], 4.0)
        self.assertEqual(json.loads(row["reject_breakdown"]), {"blur": 1, "fragmented_mask": 1})
        self.assertAlmostEqual(row["passage_quality"], 0.6 * 0.8)
        self.assertTrue(row["baseline_eligible"])
        self.assertTrue(row["score_eligible"])
        self.assertEqual(row["geometry_method"], "anchored")
        self.assertTrue(row["undistortion_applied"])
        self.assertEqual(row["center_band_fraction"], 0.5)
        self.assertEqual(
            row["camera_mitigation"], "plumb_line_undistortion+center_band"
        )

    def test_single_frame_passage_is_retained_and_ineligible(self):
        out = aggregate_passages(frame_rows(), min_valid_frames=2)
        row = out.set_index("passage_id").loc["p2"]
        self.assertEqual(row["transit_seconds"], 0.0)
        self.assertFalse(row["baseline_eligible"])
        self.assertEqual(row["n_frames_valid"], 1)

    def test_missing_timestamps_do_not_drop_passage(self):
        frame = frame_rows().iloc[:2].copy()
        frame["timestamp_utc"] = ""
        out = aggregate_passages(frame, min_valid_frames=1)
        self.assertTrue(np.isnan(out.loc[0, "transit_seconds"]))
        self.assertEqual(out.loc[0, "timestamp_utc"], "")

    def test_blank_identity_fails_loudly(self):
        frame = frame_rows()
        frame.loc[0, "cow_id"] = ""
        with self.assertRaisesRegex(ValueError, "blank cow_id"):
            aggregate_passages(frame)

    def test_mixed_camera_mitigation_fails_loudly(self):
        frame = frame_rows().iloc[:2].copy()
        frame.loc[1, "undistortion_applied"] = False
        with self.assertRaisesRegex(ValueError, "mixes undistorted and raw"):
            aggregate_passages(frame)


if __name__ == "__main__":
    unittest.main()
