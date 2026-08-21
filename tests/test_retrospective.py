import unittest

import pandas as pd

from cowarch.retrospective import (
    normalize_treatments,
    validate_absolute_scores,
    validate_retrospective,
)


class RetrospectiveTests(unittest.TestCase):
    def test_trigger_inference_uses_same_day_count(self):
        treatments = pd.DataFrame(
            {"date": ["2026-01-01"] * 3 + ["2026-01-02"], "cow_id": ["a", "b", "c", "d"]}
        )
        out = normalize_treatments(treatments, routine_count_threshold=3)
        self.assertEqual(out["trigger"].tolist(), ["routine", "routine", "routine", "observed"])
        self.assertTrue(out["trigger_inferred"].all())

    def test_latency_false_alert_rate_and_lower_bound(self):
        rows = []
        for cow in ["treated", "untreated"]:
            for day in range(1, 11):
                rows.append(
                    {
                        "date": f"2026-01-{day:02d}",
                        "cow_id": cow,
                        "alert": (cow == "treated" and day in {6, 7, 8})
                        or (cow == "untreated" and day == 4),
                    }
                )
        treatments = pd.DataFrame(
            {"date": ["2026-01-08"], "cow_id": ["treated"], "trigger": ["observed"]}
        )
        metrics, latency, _ = validate_retrospective(
            pd.DataFrame(rows), treatments, min_persistent_days=2
        )
        self.assertTrue(latency.loc[0, "detected"])
        self.assertEqual(latency.loc[0, "detection_latency_days"], 2)
        self.assertEqual(metrics["false_alerts_untreated_cows"], 1)
        self.assertGreater(metrics["false_alerts_per_cow_month"], 0)
        self.assertAlmostEqual(metrics["measured_precision_lower_bound"], 3 / 4)

    def test_nonpersistent_signal_is_not_detection(self):
        signals = pd.DataFrame(
            {"date": ["2026-01-05", "2026-01-06"], "cow_id": ["a", "a"], "alert": [True, False]}
        )
        treatments = pd.DataFrame(
            {"date": ["2026-01-06"], "cow_id": ["a"], "trigger": ["observed"]}
        )
        metrics, latency, _ = validate_retrospective(signals, treatments)
        self.assertFalse(latency.loc[0, "detected"])
        self.assertEqual(metrics["n_detected_observed_events"], 0)

    def test_absolute_score_correlation_excludes_routine_and_peripartum(self):
        passages = pd.DataFrame(
            {
                "passage_id": ["p1", "p2", "p3", "p4", "p5"],
                "cow_id": ["a", "a", "b", "b", "c"],
                "timestamp_utc": [
                    "2026-01-01T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                ],
                "sagitta_median": [0.8, 0.9, 0.1, 0.2, 0.95],
                "score_eligible": [True] * 5,
            }
        )
        treatments = pd.DataFrame(
            {
                "date": ["2026-01-10", "2026-01-10"],
                "cow_id": ["a", "b"],
                "trigger": ["observed", "routine"],
            }
        )
        calvings = pd.DataFrame({"date": ["2026-01-08"], "cow_id": ["c"]})
        metrics, labeled, normalized = validate_absolute_scores(
            passages, treatments, calvings, event_horizon_days=14
        )
        self.assertGreater(metrics["absolute_score_treatment_spearman"], 0.8)
        self.assertEqual(metrics["n_passages_peripartum_suppressed"], 1)
        self.assertEqual(metrics["n_observed_treatment_events"], 1)
        self.assertEqual(normalized["trigger"].tolist(), ["observed", "routine"])
        self.assertTrue(labeled.loc[labeled["cow_id"].eq("c"), "peripartum_suppressed"].all())


if __name__ == "__main__":
    unittest.main()
