import unittest

import numpy as np

from cowarch.geometry import (
    anchored_topline,
    anchored_topline_features,
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


def anchored_silhouette(arch_height: float) -> tuple[np.ndarray, np.ndarray]:
    """A body with a known topline and extra silhouette outside its anchors."""
    mask = np.zeros((100, 160), dtype=np.uint8)
    for x in range(10, 151):
        if 20 <= x <= 140:
            phase = (x - 20) / 120.0
            top = int(round(45 - arch_height * np.sin(np.pi * phase)))
        else:
            top = 45
        mask[top:90, x] = 1
    return mask, np.array([[20.0, 45.0], [140.0, 45.0]])


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

    def test_keypoint_signed_features_distinguish_arch_from_sag(self):
        arch = np.array([[10, 40], [30, 34], [50, 27], [70, 34], [90, 40]], dtype=float)
        sag = arch.copy()
        sag[1:-1, 1] = 80.0 - sag[1:-1, 1]

        arch_features = keypoint_features(arch)
        sag_features = keypoint_features(sag)

        self.assertGreater(arch_features["kp_arch_height_signed"], 0.0)
        self.assertLess(sag_features["kp_arch_height_signed"], 0.0)
        self.assertGreater(arch_features["kp_mean_signed_deviation_norm"], 0.0)
        self.assertLess(sag_features["kp_mean_signed_deviation_norm"], 0.0)
        self.assertAlmostEqual(
            arch_features["kp_arch_height_abs"],
            sag_features["kp_arch_height_abs"],
        )
        self.assertAlmostEqual(
            arch_features["kp_mean_abs_deviation_norm"],
            sag_features["kp_mean_abs_deviation_norm"],
        )

    def test_keypoint_signed_features_survive_horizontal_flip(self):
        points = np.array([[10, 42], [30, 34], [50, 27], [70, 36], [90, 40]], dtype=float)
        flipped = points.copy()
        flipped[:, 0] = 100.0 - flipped[:, 0]

        original_features = keypoint_features(points)
        flipped_features = keypoint_features(flipped)
        for name in original_features:
            self.assertAlmostEqual(original_features[name], flipped_features[name], places=8)

    def test_keypoint_normalized_features_survive_resize(self):
        points = np.array([[10, 42], [30, 34], [50, 27], [70, 36], [90, 40]], dtype=float)
        original_features = keypoint_features(points)
        resized_features = keypoint_features(points * 3.0)

        normalized_names = [name for name in original_features if name.endswith("_norm")]
        for name in normalized_names:
            self.assertAlmostEqual(original_features[name], resized_features[name], places=8)
        self.assertAlmostEqual(
            resized_features["kp_arch_height_signed"],
            3.0 * original_features["kp_arch_height_signed"],
        )
        self.assertAlmostEqual(
            resized_features["kp_arch_height_abs"],
            3.0 * original_features["kp_arch_height_abs"],
        )

    def test_anchored_signed_features_distinguish_arch_from_sag(self):
        arch_mask, anchors = anchored_silhouette(14.0)
        sag_mask, _ = anchored_silhouette(-14.0)

        arch = anchored_topline_features(arch_mask, anchors)
        sag = anchored_topline_features(sag_mask, anchors)

        self.assertGreater(arch["anchored_sagitta_signed_norm"], 0.0)
        self.assertLess(sag["anchored_sagitta_signed_norm"], 0.0)
        self.assertAlmostEqual(
            arch["anchored_sagitta_abs_norm"],
            sag["anchored_sagitta_abs_norm"],
        )
        self.assertAlmostEqual(
            arch["anchored_chord_rmse_norm"],
            sag["anchored_chord_rmse_norm"],
        )

    def test_anchored_features_survive_horizontal_flip(self):
        mask, anchors = anchored_silhouette(14.0)
        flipped_anchors = anchors.copy()
        flipped_anchors[:, 0] = mask.shape[1] - 1 - flipped_anchors[:, 0]

        original = anchored_topline_features(mask, anchors)
        flipped = anchored_topline_features(np.fliplr(mask), flipped_anchors)

        for name in original:
            self.assertAlmostEqual(original[name], flipped[name], places=8)

    def test_anchored_normalized_features_survive_resize(self):
        mask, anchors = anchored_silhouette(14.0)
        resized_mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)

        original = anchored_topline_features(mask, anchors)
        resized = anchored_topline_features(resized_mask, anchors * 2.0)

        for name in original:
            self.assertAlmostEqual(original[name], resized[name], delta=0.002)

    def test_anchored_features_ignore_mask_outside_anchors(self):
        mask, anchors = anchored_silhouette(14.0)
        with_artifacts = mask.copy()
        with_artifacts[2:90, :15] = 1
        with_artifacts[3:90, 146:] = 1

        baseline = anchored_topline_features(mask, anchors)
        artifact_features = anchored_topline_features(with_artifacts, anchors)

        for name in baseline:
            self.assertAlmostEqual(baseline[name], artifact_features[name], places=12)

    def test_anchored_topline_returns_anatomical_dense_points(self):
        mask, anchors = anchored_silhouette(14.0)
        topline = anchored_topline(mask, anchors, samples=51)

        self.assertEqual(topline.shape, (51, 2))
        np.testing.assert_allclose(topline[[0, -1]], anchors)

    def test_bad_or_overlapping_anchors_raise_clear_errors(self):
        mask, anchors = anchored_silhouette(14.0)
        cases = [
            (np.array([20.0, 45.0]), "shape"),
            (np.array([[20.0, 45.0], [20.0, 45.0]]), "cannot coincide"),
            (np.array([[-1.0, 45.0], [140.0, 45.0]]), "mask bounds"),
            (np.array([[20.0, 45.0], [20.0, 70.0]]), "horizontal positions"),
            (np.array([[20.0, np.nan], [140.0, 45.0]]), "finite"),
        ]
        for bad_anchors, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    anchored_topline_features(mask, bad_anchors)


if __name__ == "__main__":
    unittest.main()
