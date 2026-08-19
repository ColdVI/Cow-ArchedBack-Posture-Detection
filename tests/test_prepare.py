import unittest
from unittest.mock import patch

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

    def test_mask_from_detector_result_produces_auto_features(self):
        frame = blank_frame()
        mask = np.zeros((120, 240), dtype=bool)
        mask[30:90, 40:200] = True
        record = blank_record("src", "vid", 0, "a.png")

        detection = (np.array([40, 30, 200, 90], dtype=float), 0.91, mask)
        with patch("cowarch.prepare.predict_cows", return_value=[detection]):
            outcome = process_frame(
                frame,
                record,
                detector=object(),
                cow_class=0,
                config=PrepareConfig(),
            )
        self.assertTrue(outcome.accepted)
        self.assertIsNotNone(outcome.mask)
        self.assertTrue(np.isfinite(record["auto_sagitta"]))
        self.assertTrue(all(key.startswith("auto_") for key in record if "sagitta" in key))

    def test_single_image_can_select_one_of_multiple_detections(self):
        frame = blank_frame()
        small_mask = np.zeros((120, 240), dtype=bool)
        small_mask[32:88, 42:118] = True
        detections = [
            (np.array([40, 30, 120, 90], dtype=float), 0.80, small_mask),
            (np.array([120, 30, 200, 90], dtype=float), 0.95, small_mask),
        ]
        with patch("cowarch.prepare.predict_cows", return_value=detections):
            batch_record = blank_record("src", "vid", 0, "a.png")
            batch = process_frame(
                frame, batch_record, detector=object(), cow_class=0, config=PrepareConfig()
            )
            chosen_record = blank_record("src", "vid", 0, "a.png")
            chosen = process_frame(
                frame,
                chosen_record,
                detector=object(),
                cow_class=0,
                config=PrepareConfig(),
                detection_index=1,
            )
        self.assertEqual(batch.reject_reason, "multiple_cows")
        self.assertTrue(chosen.accepted)
        self.assertEqual(chosen_record["selected_detection_index"], 1)
        np.testing.assert_allclose(chosen.box, detections[1][0])

    def test_invalid_detection_index_is_clear(self):
        detection = (np.array([40, 30, 200, 90], dtype=float), 0.91, None)
        with patch("cowarch.prepare.predict_cows", return_value=[detection]):
            with self.assertRaisesRegex(IndexError, "out of range"):
                process_frame(
                    blank_frame(),
                    blank_record("src", "vid", 0, "a.png"),
                    detector=object(),
                    cow_class=0,
                    config=PrepareConfig(),
                    detection_index=4,
                )

    def test_helpers(self):
        self.assertEqual(safe_token("a b/c"), "a_b_c")
        self.assertEqual(hamming_distance(0b1010, 0b1001), 2)


if __name__ == "__main__":
    unittest.main()
