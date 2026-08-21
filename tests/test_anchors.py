import unittest

import numpy as np
import pandas as pd

from cowarch.anchors import anchor_array, evaluate_pckh, predict_anchors


class AnchorTests(unittest.TestCase):
    def test_framework_neutral_prediction_contract(self):
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        prediction = predict_anchors(
            crop,
            lambda _image: np.array(
                [[5.0, 8.0, 0.9], [30.0, 9.0, 0.8], [3.0, 12.0, 0.7]]
            ),
        )
        self.assertEqual(list(prediction), ["withers", "sacrum", "head"])
        points = anchor_array(prediction, min_confidence=0.7)
        np.testing.assert_allclose(points, [[5, 8], [30, 9], [3, 12]])

    def test_low_confidence_rejects_whole_measurement(self):
        prediction = {
            "withers": {"x": 1, "y": 2, "confidence": 0.9},
            "sacrum": {"x": 8, "y": 2, "confidence": 0.2},
            "head": {"x": 0, "y": 5, "confidence": 0.9},
        }
        self.assertIsNone(anchor_array(prediction, min_confidence=0.5))

    def test_missing_predictor_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "No anchor predictor"):
            predict_anchors(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_pckh_reports_each_production_point_and_anchor_gate(self):
        rows = []
        for sample in range(10):
            for keypoint in ("withers", "sacrum", "head"):
                error = 1.0 if keypoint != "head" else 3.0
                rows.append(
                    {
                        "sample_id": sample,
                        "keypoint": keypoint,
                        "pred_x": error,
                        "pred_y": 0,
                        "true_x": 0,
                        "true_y": 0,
                        "head_scale_px": 10,
                    }
                )
        summary, details = evaluate_pckh(pd.DataFrame(rows))
        self.assertTrue(summary["anchor_gate_pass"])
        self.assertEqual(summary["pckh_withers"], 1.0)
        self.assertEqual(summary["pckh_sacrum"], 1.0)
        self.assertEqual(summary["pckh_head"], 0.0)
        self.assertEqual(len(details), 30)

    def test_pckh_rejects_infinite_coordinates(self):
        rows = []
        for keypoint in ("withers", "sacrum", "head"):
            rows.append(
                {
                    "sample_id": "p1",
                    "keypoint": keypoint,
                    "pred_x": np.inf if keypoint == "head" else 0,
                    "pred_y": 0,
                    "true_x": 0,
                    "true_y": 0,
                    "head_scale_px": 10,
                }
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_pckh(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
