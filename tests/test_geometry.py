import unittest

import numpy as np

from cowarch.geometry import (
    auto_topline_features,
    extract_topline,
    keypoint_features,
    normalized_sagitta,
)


def silhouette(arch_height: float) -> np.ndarray:
    mask = np.zeros((100, 160), dtype=np.uint8)
    for x in range(20, 140):
        phase = (x - 20) / 119.0
        top = int(round(45 - arch_height * np.sin(np.pi * phase)))
        mask[top:85, x] = 1
    return mask


class GeometryTests(unittest.TestCase):
    def test_empty_mask_returns_none(self):
        self.assertIsNone(extract_topline(np.zeros((20, 20), dtype=np.uint8)))

    def test_arch_has_larger_sagitta_than_flat(self):
        flat = auto_topline_features(silhouette(0.0))
        arched = auto_topline_features(silhouette(14.0))
        self.assertLess(abs(flat["auto_sagitta"]), 1e-6)
        self.assertGreater(arched["auto_sagitta"], flat["auto_sagitta"] + 0.05)

    def test_real_x_coordinates_are_used(self):
        px = np.array([0.0, 1.0, 8.0, 9.0, 10.0])
        py = np.array([5.0, 5.0, 1.0, 5.0, 5.0])
        self.assertAlmostEqual(normalized_sagitta(px, py, endpoint_window=1), 0.4)

    def test_keypoint_features_are_direction_invariant(self):
        points = np.array([[10, 40], [30, 34], [50, 27], [70, 34], [90, 40]], dtype=float)
        forward = keypoint_features(points)
        reverse = keypoint_features(points[::-1])
        for name in forward:
            self.assertAlmostEqual(forward[name], reverse[name], places=8)

    def test_non_monotonic_keypoints_are_rejected(self):
        points = np.array([[10, 40], [30, 34], [25, 27], [70, 34], [90, 40]], dtype=float)
        with self.assertRaises(ValueError):
            keypoint_features(points)


if __name__ == "__main__":
    unittest.main()

