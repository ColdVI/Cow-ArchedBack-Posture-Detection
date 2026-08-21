import unittest

import pandas as pd

from cowarch.variance import variance_study


def passages(delta=0.2, day_shift=0.0):
    rows = []
    for cow_id, base in [("normal1", 0.1), ("normal2", 0.11), ("arched1", 0.1 + delta)]:
        for day in range(3):
            for passage in range(2):
                rows.append(
                    {
                        "cow_id": cow_id,
                        "camera_id": "cam1",
                        "pipeline_version": "v1",
                        "timestamp_utc": f"2026-01-0{day + 1}T0{passage}:00:00Z",
                        "sagitta_median": base + day * day_shift + passage * 0.002,
                        "baseline_eligible": True,
                    }
                )
    return pd.DataFrame(rows)


class VarianceTests(unittest.TestCase):
    def test_clear_separation_passes_gate(self):
        summary, per_cow, measurements = variance_study(passages(), {"arched1"})
        self.assertEqual(summary["decision"], "GO")
        self.assertGreaterEqual(summary["delta_over_sigma_within_cow"], 2.0)
        self.assertGreaterEqual(summary["delta_over_sigma_between_healthy"], 2.0)
        self.assertGreater(summary["sigma_between_healthy"], 0)
        self.assertEqual(len(per_cow), 3)
        self.assertEqual(len(measurements), 18)

    def test_small_signal_is_no_go(self):
        summary, _, _ = variance_study(passages(delta=0.005), {"arched1"})
        self.assertEqual(summary["decision"], "NO_GO_SIGNAL")

    def test_large_day_variance_reports_environment_problem(self):
        frame = passages()
        # Shared day movement is camera-like and dominates within-day spread.
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame["sagitta_median"] += (timestamps.dt.day - 1) * 0.08
        summary, _, _ = variance_study(frame, {"arched1"})
        self.assertEqual(summary["decision"], "NO_GO_CAMERA_ENVIRONMENT")

    def test_multiple_cameras_must_be_stratified(self):
        frame = passages()
        frame.loc[0, "camera_id"] = "cam2"
        with self.assertRaisesRegex(ValueError, "per camera"):
            variance_study(frame, {"arched1"})


if __name__ == "__main__":
    unittest.main()
