import unittest

import numpy as np

from cowarch.frames import hamming_distance, safe_token
from cowarch.detect import find_cow_class
from cowarch.prepare import PrepareConfig, blank_record, process_frame


def blank_frame(width=240, height=120):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[30:90, 40:200] = 210
    return frame


class PrepareTests(unittest.TestCase):
    def test_cow_class_is_resolved_by_name_not_index(self):
        self.assertEqual(find_cow_class({0: "person", 7: "cow"}), 7)
        with self.assertRaises(RuntimeError):
            find_cow_class({0: "person"})

    def test_detectorless_pass_accepts_whole_frame(self):
        record = blank_record("src", "vid", 0, "a.png")
        outcome = process_frame(
            blank_frame(), record, detector=None, cow_class=None, config=PrepareConfig()
        )
        self.assertTrue(outcome.accepted)
        self.assertEqual(outcome.crop.shape[:2], (120, 240))
        self.assertAlmostEqual(record["aspect_ratio"], 2.0, places=6)

    def test_near_duplicate_is_rejected_against_the_previous_hash(self):
        frame = blank_frame()
        first = blank_record("src", "vid", 0, "a.png")
        outcome = process_frame(
            frame, first, detector=None, cow_class=None, config=PrepareConfig()
        )
        second = blank_record("src", "vid", 1, "b.png")
        repeat = process_frame(
            frame,
            second,
            detector=None,
            cow_class=None,
            config=PrepareConfig(),
            last_hash=outcome.hash_value,
        )
        self.assertFalse(repeat.accepted)
        self.assertEqual(repeat.reject_reason, "near_duplicate")

    def test_hard_side_filter_rejects_a_tall_frame(self):
        record = blank_record("src", "vid", 0, "a.png")
        outcome = process_frame(
            blank_frame(width=100, height=240),
            record,
            detector=None,
            cow_class=None,
            config=PrepareConfig(hard_side_filter=True),
        )
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reject_reason, "aspect_ratio")
        self.assertEqual(record["view_hint"], "review_oblique")

    def test_mask_produces_auto_features_marked_as_auto(self):
        frame = blank_frame()
        mask = np.zeros((120, 240), dtype=bool)
        mask[30:90, 40:200] = True
        record = blank_record("src", "vid", 0, "a.png")

        class FakeDetector:
            pass

        outcome = process_frame(
            frame,
            record,
            detector=None,
            cow_class=None,
            config=PrepareConfig(),
        )
        self.assertTrue(np.isnan(record["auto_sagitta"]))
        self.assertTrue(all(key.startswith("auto_") for key in record if "sagitta" in key))

    def test_helpers(self):
        self.assertEqual(safe_token("a b/c"), "a_b_c")
        self.assertEqual(hamming_distance(0b1010, 0b1001), 2)


if __name__ == "__main__":
    unittest.main()
