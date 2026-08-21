import unittest
from unittest.mock import patch

import numpy as np

from cowarch.frames import hamming_distance, safe_token
from cowarch.detect import find_cow_class
from cowarch.prepare import PrepareConfig, blank_record, frame_timestamp_utc, process_frame


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
        self.assertGreater(record["body_length_px"], 0)
        self.assertEqual(record["mask_component_count"], 1)
        self.assertTrue(all(key.startswith("auto_") for key in record if "sagitta" in key))

    def test_fragmented_mask_has_specific_reject_reason(self):
        frame = blank_frame()
        mask = np.zeros((120, 240), dtype=bool)
        mask[30:60, 40:90] = True
        mask[30:60, 120:180] = True
        record = blank_record("src", "vid", 0, "a.png")
        detection = (np.array([40, 25, 185, 65], dtype=float), 0.91, mask)
        with patch("cowarch.prepare.predict_cows", return_value=[detection]):
            outcome = process_frame(
                frame, record, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.reject_reason, "fragmented_mask")
        self.assertEqual(record["mask_component_count"], 2)

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
        self.assertEqual(
            frame_timestamp_utc("2026-01-01T03:00:00+03:00", 50, 25.0),
            "2026-01-01T00:00:02Z",
        )
        self.assertEqual(frame_timestamp_utc("", 0, None), "")

    def test_blank_record_contains_longitudinal_schema(self):
        record = blank_record("src", "vid", 0, "a.png")
        expected = {
            "cow_id", "passage_id", "timestamp_utc", "camera_id", "is_ir",
            "mask_component_count", "body_length_px", "pipeline_version",
        }
        self.assertTrue(expected.issubset(record))


if __name__ == "__main__":
    unittest.main()


class MultiCowTests(unittest.TestCase):
    def test_no_detections_returns_single_no_cow_outcome(self):
        class NoDetector:
            names = {0: "cow"}

        def factory(index):
            return blank_record("src", "vid", index, "a.png")

        from cowarch.prepare import process_frame_all_cows

        frame = blank_frame()

        import unittest.mock as mock
        with mock.patch("cowarch.prepare.detect_frame_cows", return_value=[]):
            outcomes = process_frame_all_cows(
                frame, factory, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0].accepted)
        self.assertEqual(outcomes[0].reject_reason, "no_cow")

    def test_each_detection_becomes_its_own_candidate(self):
        import unittest.mock as mock

        from cowarch.prepare import process_frame_all_cows

        # Distinct pixel content per region: identical crops would dhash to the
        # same value and the second cow would be dropped as a near-duplicate,
        # which is a real (documented) limitation, not what this test checks.
        frame = np.zeros((200, 600, 3), dtype=np.uint8)
        frame[40:160, 0:250] = 60
        frame[40:160, 300:560] = 200
        two_cows = [
            (np.array([0.0, 40.0, 250.0, 160.0]), 0.9, None),
            (np.array([300.0, 40.0, 560.0, 160.0]), 0.85, None),
        ]

        def factory(index):
            return blank_record("src", "vid", index, "a.png")

        with mock.patch("cowarch.prepare.detect_frame_cows", return_value=two_cows):
            outcomes = process_frame_all_cows(
                frame, factory, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertEqual(len(outcomes), 2)
        self.assertTrue(all(o.accepted for o in outcomes))
        self.assertEqual(outcomes[0].record["cow_count"], 2)
        self.assertEqual(outcomes[0].record["selected_detection_index"], 0)
        self.assertEqual(outcomes[1].record["selected_detection_index"], 1)
        # distinct crops, not the same region duplicated
        self.assertFalse(np.array_equal(outcomes[0].crop.shape, outcomes[1].crop.shape) and
                          np.array_equal(outcomes[0].box, outcomes[1].box))

    def test_detector_runs_once_regardless_of_cow_count(self):
        import unittest.mock as mock

        from cowarch.prepare import process_frame_all_cows

        frame = np.zeros((200, 600, 3), dtype=np.uint8)
        three_cows = [
            (np.array([0.0, 40.0, 150.0, 160.0]), 0.9, None),
            (np.array([200.0, 40.0, 350.0, 160.0]), 0.9, None),
            (np.array([400.0, 40.0, 550.0, 160.0]), 0.9, None),
        ]

        def factory(index):
            return blank_record("src", "vid", index, "a.png")

        with mock.patch(
            "cowarch.prepare.detect_frame_cows", return_value=three_cows
        ) as mocked:
            process_frame_all_cows(
                frame, factory, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertEqual(mocked.call_count, 1)


class OverlapSuppressionTests(unittest.TestCase):
    def test_iou_of_identical_boxes_is_one(self):
        from cowarch.prepare import box_iou
        box = np.array([10.0, 10.0, 110.0, 110.0])
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        from cowarch.prepare import box_iou
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([100.0, 100.0, 110.0, 110.0])
        self.assertEqual(box_iou(a, b), 0.0)

    def test_heavily_overlapping_fragment_is_suppressed(self):
        import unittest.mock as mock
        from cowarch.prepare import process_frame_all_cows

        frame = np.zeros((200, 600, 3), dtype=np.uint8)
        frame[40:160, 0:300] = 120
        # second box is almost the same region as the first: a rail-fragmented
        # detection of the same physical animal, not a second cow.
        fragmented = [
            (np.array([0.0, 40.0, 300.0, 160.0]), 0.9, None),
            (np.array([10.0, 42.0, 295.0, 158.0]), 0.8, None),
        ]

        def factory(index):
            return blank_record("src", "vid", index, "a.png")

        with mock.patch("cowarch.prepare.detect_frame_cows", return_value=fragmented):
            outcomes = process_frame_all_cows(
                frame, factory, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertTrue(outcomes[0].accepted)
        self.assertFalse(outcomes[1].accepted)
        self.assertEqual(outcomes[1].reject_reason, "overlaps_accepted")

    def test_well_separated_cows_both_survive_overlap_check(self):
        import unittest.mock as mock
        from cowarch.prepare import process_frame_all_cows

        frame = np.zeros((200, 600, 3), dtype=np.uint8)
        frame[40:160, 0:250] = 60
        frame[40:160, 300:560] = 200
        two_cows = [
            (np.array([0.0, 40.0, 250.0, 160.0]), 0.9, None),
            (np.array([300.0, 40.0, 560.0, 160.0]), 0.85, None),
        ]

        def factory(index):
            return blank_record("src", "vid", index, "a.png")

        with mock.patch("cowarch.prepare.detect_frame_cows", return_value=two_cows):
            outcomes = process_frame_all_cows(
                frame, factory, detector=object(), cow_class=0, config=PrepareConfig()
            )
        self.assertTrue(all(o.accepted for o in outcomes))
