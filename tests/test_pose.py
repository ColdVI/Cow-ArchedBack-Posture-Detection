import json
import unittest

from cowarch.pose import (
    POSE_KEYPOINTS,
    POSE_SCHEMA,
    geometry_keypoints_visible,
    parse_pose_points,
    pose_prefilter_reason,
    serialize_pose_points,
)


class PoseSchemaTests(unittest.TestCase):
    def make_points(self):
        return [[float(i), float(i + 1), 2] for i in range(len(POSE_KEYPOINTS))]

    def test_round_trip_preserves_order_and_schema(self):
        value = serialize_pose_points(self.make_points())
        payload = json.loads(value)
        parsed = parse_pose_points(value)
        self.assertEqual(payload["schema"], POSE_SCHEMA)
        self.assertEqual(len(parsed), 19)
        self.assertEqual(parsed[2], [2.0, 3.0, 2])

    def test_missing_point_coordinates_are_canonicalized(self):
        points = self.make_points()
        points[10] = [999.0, 999.0, 0]
        parsed = parse_pose_points(serialize_pose_points(points))
        self.assertEqual(parsed[10], [0.0, 0.0, 0])

    def test_wrong_number_of_points_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires 19"):
            serialize_pose_points(self.make_points()[:-1])

    def test_geometry_eligibility_requires_withers_and_sacrum(self):
        points = self.make_points()
        self.assertTrue(geometry_keypoints_visible(points))
        points[6] = [0.0, 0.0, 0]
        self.assertFalse(geometry_keypoints_visible(points))

    def test_pose_prefilter_keeps_a_clean_lateral_crop(self):
        record = {
            "accepted": True,
            "touches_border": False,
            "detection_conf": 0.8,
            "bbox_area_ratio": 0.2,
            "aspect_ratio": 2.1,
        }
        self.assertEqual(pose_prefilter_reason(record), "")

    def test_pose_prefilter_rejects_border_cuts_before_geometry(self):
        record = {
            "accepted": True,
            "touches_border": True,
            "detection_conf": 0.8,
            "bbox_area_ratio": 0.2,
            "aspect_ratio": 2.1,
        }
        self.assertEqual(pose_prefilter_reason(record), "touches_border")


if __name__ == "__main__":
    unittest.main()
