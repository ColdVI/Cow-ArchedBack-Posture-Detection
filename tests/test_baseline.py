import unittest

import numpy as np
import pandas as pd

from cowarch.baseline import apply_threshold, capacity_threshold, score_changes


def passages() -> pd.DataFrame:
    rows = []
    for cow_id, bump in [("cow1", 0.0), ("cow2", 0.0), ("cow3", 0.0)]:
        for day in range(8):
            sagitta = 0.1 + day * 0.01  # shared camera drift
            if cow_id == "cow3" and day >= 6:
                sagitta += 0.08
            rows.append(
                {
                    "passage_id": f"{cow_id}-{day}",
                    "cow_id": cow_id,
                    "camera_id": "cam1",
                    "timestamp_utc": f"2026-01-{day + 1:02d}T08:00:00Z",
                    "sagitta_median": sagitta + bump,
                    "passage_quality": 0.9,
                    "baseline_eligible": True,
                    "pipeline_version": "v1",
                }
            )
    return pd.DataFrame(rows)


class BaselineTests(unittest.TestCase):
    def test_cold_start_drift_correction_and_change_score(self):
        scored, drift = score_changes(passages(), sigma_floor=0.005, min_history=3)
        cow1 = scored[scored["cow_id"] == "cow1"]
        self.assertTrue((cow1.iloc[:3]["status"] == "insufficient_history").all())
        self.assertAlmostEqual(float(drift["camera_drift_offset"].iloc[0]), -0.035)
        last_cow3 = scored[scored["cow_id"] == "cow3"].iloc[-1]
        last_cow1 = cow1.iloc[-1]
        self.assertGreater(last_cow3["change_score"], last_cow1["change_score"])
        self.assertGreaterEqual(last_cow3["baseline_sigma"], 0.005)

    def test_low_quality_is_retained_but_not_scored(self):
        frame = passages()
        frame.loc[0, "baseline_eligible"] = False
        scored, _ = score_changes(frame, sigma_floor=0.005, min_history=1)
        self.assertEqual(scored.iloc[0]["status"], "low_quality")

    def test_calving_resets_history_and_suppresses_alerts(self):
        calvings = pd.DataFrame({"cow_id": ["cow3"], "date": ["2026-01-06"]})
        scored, _ = score_changes(
            passages(), sigma_floor=0.005, min_history=2, calvings=calvings,
            peripartum_days_before=1, peripartum_days_after=1,
        )
        row = scored[(scored["cow_id"] == "cow3") & (scored["date"] == pd.Timestamp("2026-01-06", tz="UTC"))].iloc[0]
        self.assertEqual(row["status"], "insufficient_history")
        self.assertTrue(row["peripartum_suppressed"])

    def test_capacity_threshold_is_data_derived_and_applied(self):
        scored, _ = score_changes(passages(), sigma_floor=0.005, min_history=2)
        capacity = capacity_threshold(scored, daily_capacity=1)
        self.assertTrue(np.isfinite(capacity["threshold"]))
        self.assertLessEqual(capacity["max_flags_per_day"], 1)
        passage_out, daily = apply_threshold(scored, capacity["threshold"])
        self.assertIn("alert", passage_out)
        self.assertIn("alert", daily)

    def test_sigma_floor_is_mandatory(self):
        with self.assertRaisesRegex(ValueError, "sigma_floor"):
            score_changes(passages(), sigma_floor=0)


if __name__ == "__main__":
    unittest.main()
