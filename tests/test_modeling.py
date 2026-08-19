import unittest

import numpy as np

from cowarch.modeling import binary_metrics, train_with_validation


class ModelingTests(unittest.TestCase):
    def test_train_select_threshold_and_evaluate(self):
        rng = np.random.default_rng(3)
        x_train = np.r_[rng.normal(-2, 0.4, (30, 3)), rng.normal(2, 0.4, (30, 3))]
        y_train = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
        x_val = np.r_[rng.normal(-2, 0.4, (12, 3)), rng.normal(2, 0.4, (12, 3))]
        y_val = np.r_[np.zeros(12, dtype=int), np.ones(12, dtype=int)]
        result = train_with_validation(x_train, y_train, x_val, y_val)
        probability = result.pipeline.predict_proba(x_val)[:, 1]
        metrics = binary_metrics(y_val, probability, result.threshold)
        self.assertGreater(metrics["f1"], 0.95)
        self.assertGreaterEqual(result.threshold, 0.0)
        self.assertLessEqual(result.threshold, 1.0)


if __name__ == "__main__":
    unittest.main()

