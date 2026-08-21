import unittest

import numpy as np

from cowarch.camera import fit_plumb_line_calibration, undistort_image


def distort(points, width, height, focal, k1):
    center = np.array([width / 2, height / 2], dtype=float)
    normalized = (points - center) / focal
    radius2 = np.sum(normalized * normalized, axis=1, keepdims=True)
    return center + focal * normalized * (1 + k1 * radius2)


class CameraTests(unittest.TestCase):
    def test_plumb_line_fit_reduces_line_error(self):
        width, height, focal = 640, 480, 640.0
        lines = [
            np.column_stack((np.full(12, 80.0), np.linspace(20, 460, 12))),
            np.column_stack((np.full(12, 560.0), np.linspace(20, 460, 12))),
            np.column_stack((np.linspace(20, 620, 14), np.full(14, 80.0))),
            np.column_stack((np.linspace(20, 620, 14), np.full(14, 400.0))),
        ]
        distorted = [distort(points, width, height, focal, 0.35) for points in lines]
        calibration = fit_plumb_line_calibration(
            distorted, image_width=width, image_height=height, focal_length_px=focal
        )
        self.assertLess(
            calibration.rms_line_error_after_px,
            calibration.rms_line_error_before_px * 0.25,
        )
        output = undistort_image(np.zeros((height, width, 3), dtype=np.uint8), calibration)
        self.assertEqual(output.shape, (height, width, 3))


if __name__ == "__main__":
    unittest.main()
