import unittest

import numpy as np
import pandas as pd

from cowarch.active import (
    TestSetLeakError,
    assert_pool_excludes_test,
    build_pool_mask,
    fit_seed_model,
    rank_by_uncertainty,
    select_next_batch,
)


def make_frame(n=30):
    return pd.DataFrame(
        {
            "sample_id": [f"s{i:03d}" for i in range(n)],
            "source_id": ["src"] * n,
            "video_id": [f"v{i % 6}" for i in range(n)],
            "crop_path": [f"c{i}.jpg" for i in range(n)],
            "accepted": [True] * n,
            "split": ["train"] * 18 + ["val"] * 6 + ["test"] * 6,
            "label": [""] * n,
        }
    )


class ActiveLearningTests(unittest.TestCase):
    def test_pool_never_contains_test_rows(self):
        frame = make_frame()
        mask = build_pool_mask(frame)
        self.assertTrue(mask.any())
        self.assertFalse(frame.loc[mask, "split"].eq("test").any())

    def test_guard_rejects_a_pool_holding_test_rows(self):
        frame = make_frame()
        with self.assertRaises(TestSetLeakError):
            assert_pool_excludes_test(frame, np.ones(len(frame), dtype=bool))

    def test_labeled_rows_leave_the_pool(self):
        frame = make_frame()
        frame.loc[0, "label"] = "arched"
        mask = build_pool_mask(frame)
        self.assertFalse(mask[0])

    def test_uncertainty_ranking_prefers_the_boundary(self):
        pool = np.array([True, True, True, False])
        probabilities = np.array([0.95, 0.51, 0.10, 0.50])
        chosen = rank_by_uncertainty(probabilities, pool, batch_size=2)
        self.assertEqual(list(chosen), [1, 2])

    def test_few_labels_fall_back_to_nearest_centroid(self):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(30, 8))
        labeled = np.zeros(30, dtype=bool)
        labeled[:4] = True
        model = fit_seed_model(embeddings, labeled, np.array([1, 1, 0, 0]))
        self.assertEqual(model.kind, "nearest_centroid")
        probabilities = model.predict_proba(embeddings)
        self.assertEqual(probabilities.shape, (30,))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_many_labels_upgrade_to_logistic_regression(self):
        rng = np.random.default_rng(1)
        embeddings = rng.normal(size=(40, 8))
        labeled = np.zeros(40, dtype=bool)
        labeled[:20] = True
        model = fit_seed_model(embeddings, labeled, np.array([1] * 10 + [0] * 10))
        self.assertEqual(model.kind, "logistic_regression")

    def test_select_next_batch_stays_out_of_test(self):
        frame = make_frame()
        frame.loc[[0, 1], "label"] = "arched"
        frame.loc[[2, 3], "label"] = "normal"
        rng = np.random.default_rng(2)
        embeddings = rng.normal(size=(len(frame), 12))
        batch, model, probabilities = select_next_batch(frame, embeddings, batch_size=5)
        self.assertEqual(len(batch), 5)
        self.assertFalse(frame.loc[batch, "split"].eq("test").any())
        self.assertTrue(frame.loc[batch, "label"].eq("").all())
        self.assertEqual(probabilities.shape, (len(frame),))
        self.assertIn(model.kind, {"nearest_centroid", "logistic_regression"})

    def test_seed_model_uses_only_train_labels(self):
        frame = make_frame()
        frame.loc[[0, 1], "label"] = "arched"
        frame.loc[[2, 3, 4], "label"] = "normal"
        frame.loc[18:29, "label"] = ["arched", "normal"] * 6
        embeddings = np.random.default_rng(3).normal(size=(len(frame), 10))

        _batch, model, probabilities = select_next_batch(frame, embeddings, batch_size=4)

        self.assertEqual(model.n_labeled, 5)
        self.assertEqual(probabilities.shape, (len(frame),))

    def test_selected_batch_is_train_only(self):
        frame = make_frame()
        frame.loc[[0, 1], "label"] = "arched"
        frame.loc[[2, 3], "label"] = "normal"
        embeddings = np.random.default_rng(4).normal(size=(len(frame), 6))

        batch, _model, _probabilities = select_next_batch(
            frame, embeddings, batch_size=len(frame)
        )

        self.assertTrue(frame.loc[batch, "split"].eq("train").all())

    def test_validation_cannot_be_explicitly_added_to_pool(self):
        frame = make_frame()
        with self.assertRaisesRegex(ValueError, "training-only"):
            build_pool_mask(frame, labelable_splits=("train", "val"))

    def test_test_cannot_be_explicitly_added_to_pool(self):
        frame = make_frame()
        with self.assertRaises(TestSetLeakError):
            build_pool_mask(frame, labelable_splits=("train", "test"))


if __name__ == "__main__":
    unittest.main()
