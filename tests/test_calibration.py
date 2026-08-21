import unittest

import numpy as np
import pandas as pd

from cowarch.calibration import apply_score_bands, calibrate_scores


class ScoreCalibrationTests(unittest.TestCase):
    def test_two_observers_produce_agreement_correlation_and_bands(self):
        passages = pd.DataFrame(
            {
                "passage_id": [f"p{i}" for i in range(9)],
                "sagitta_median": [0.01, 0.015, 0.02, 0.04, 0.045, 0.05, 0.08, 0.09, 0.10],
            }
        )
        scores_a = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        scores_b = [1, 1, 2, 2, 2, 2, 3, 3, 3]
        observations = pd.DataFrame(
            [
                {"passage_id": f"p{i}", "observer_id": observer, "score": scores[i]}
                for observer, scores in (("a", scores_a), ("b", scores_b))
                for i in range(9)
            ]
        )
        summary, pairs = calibrate_scores(passages, observations)
        self.assertGreater(summary["weighted_kappa_quadratic"], 0.7)
        self.assertGreater(summary["sagitta_score_spearman"], 0.8)
        self.assertEqual(len(summary["thresholds"]), 2)
        bands = apply_score_bands(passages["sagitta_median"], summary)
        self.assertEqual(bands.iloc[0], 1)
        self.assertEqual(bands.iloc[-1], 3)
        self.assertEqual(len(pairs), 9)

    def test_missing_second_observer_is_rejected(self):
        passages = pd.DataFrame({"passage_id": ["p1", "p2", "p3"], "sagitta_median": [1, 2, 3]})
        observations = pd.DataFrame(
            {
                "passage_id": ["p1", "p2", "p3", "p1", "p2"],
                "observer_id": ["a", "a", "a", "b", "b"],
                "score": [1, 2, 3, 1, 2],
            }
        )
        with self.assertRaisesRegex(ValueError, "Both observers"):
            calibrate_scores(passages, observations)

    def test_nonfinite_sagitta_is_rejected(self):
        passages = pd.DataFrame(
            {"passage_id": ["p1", "p2", "p3"], "sagitta_median": [1, np.inf, 3]}
        )
        observations = pd.DataFrame(
            {
                "passage_id": ["p1", "p2", "p3", "p1", "p2", "p3"],
                "observer_id": ["a", "a", "a", "b", "b", "b"],
                "score": [1, 2, 3, 1, 2, 3],
            }
        )
        with self.assertRaisesRegex(ValueError, "finite sagitta"):
            calibrate_scores(passages, observations)


if __name__ == "__main__":
    unittest.main()
