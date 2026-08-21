import unittest

import pandas as pd

from cowarch.splits import assign_group_splits, assert_no_group_leakage


class SplitTests(unittest.TestCase):
    def test_groups_do_not_cross_splits(self):
        frame = pd.DataFrame(
            {
                "video_id": [f"video_{group:02d}" for group in range(12) for _ in range(3)],
                "value": range(36),
            }
        )
        frame["split"] = assign_group_splits(frame, "video_id", seed=7)
        self.assertEqual(set(frame["split"]), {"train", "val", "test"})
        assert_no_group_leakage(frame, "video_id")
        self.assertTrue((frame.groupby("video_id")["split"].nunique() == 1).all())

    def test_detects_manual_leakage(self):
        frame = pd.DataFrame({"video_id": ["same", "same"], "split": ["train", "test"]})
        with self.assertRaises(ValueError):
            assert_no_group_leakage(frame, "video_id")

    def test_requires_three_groups(self):
        frame = pd.DataFrame({"video_id": ["a", "a", "b", "b"]})
        with self.assertRaises(ValueError):
            assign_group_splits(frame, "video_id")

    def test_missing_cow_id_does_not_fall_back_to_video(self):
        frame = pd.DataFrame({"video_id": ["a", "b", "c"]})
        with self.assertRaisesRegex(ValueError, "Missing group column: cow_id"):
            assign_group_splits(frame, "cow_id")

    def test_blank_cow_id_is_rejected(self):
        frame = pd.DataFrame({"cow_id": ["cow-1", "", "cow-3"]})
        with self.assertRaisesRegex(ValueError, "Blank values.*cow_id"):
            assign_group_splits(frame, "cow_id")


if __name__ == "__main__":
    unittest.main()
