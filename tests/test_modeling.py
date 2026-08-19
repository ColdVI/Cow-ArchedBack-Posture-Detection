import unittest

import numpy as np

from cowarch.modeling import (
    aggregate_group_predictions,
    binary_metrics,
    cluster_bootstrap_ci,
    cluster_bootstrap_scores,
    evaluate_frame_and_group_metrics,
    group_binary_metrics,
    train_with_validation,
)


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

    def test_group_probabilities_default_to_mean(self):
        grouped = aggregate_group_predictions(
            y_true=np.array([0, 0, 1, 1, 1]),
            probability=np.array([0.1, 0.3, 0.6, 0.8, 1.0]),
            group_ids=np.array(["video_a", "video_a", "video_b", "video_b", "video_b"]),
        )
        self.assertEqual(grouped.group_ids.tolist(), ["video_a", "video_b"])
        self.assertEqual(grouped.y_true.tolist(), [0, 1])
        np.testing.assert_allclose(grouped.probability, [0.2, 0.8])
        self.assertEqual(grouped.frame_counts.tolist(), [2, 3])

    def test_group_metrics_keep_frame_and_group_counts(self):
        metrics = group_binary_metrics(
            y_true=np.array([0, 0, 1, 1]),
            probability=np.array([0.1, 0.2, 0.8, 0.9]),
            group_ids=np.array(["cow_a", "cow_a", "cow_b", "cow_b"]),
            threshold=0.5,
        )
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["n_groups"], 2)
        self.assertEqual(metrics["n_frames"], 4)
        self.assertEqual(metrics["probability_aggregation"], "mean")
        self.assertEqual(metrics["confusion_matrix_tn_fp_fn_tp"], [[1, 0], [0, 1]])

    def test_conflicting_ground_truth_within_group_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting ground-truth labels"):
            aggregate_group_predictions(
                y_true=np.array([0, 1, 1]),
                probability=np.array([0.2, 0.7, 0.9]),
                group_ids=np.array(["same_passage", "same_passage", "other"]),
            )

    def test_cluster_bootstrap_samples_whole_groups_not_frames(self):
        # Cluster sizes 2 and 3 mean a two-cluster draw can contain only 4, 5,
        # or 6 frames. A frame bootstrap would always contain five frames.
        y_true = np.array([0, 0, 1, 1, 1])
        probability = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        groups = np.array(["passage_a", "passage_a", "passage_b", "passage_b", "passage_b"])
        sizes = cluster_bootstrap_scores(
            y_true,
            probability,
            groups,
            metric=lambda sampled_y, _sampled_p: float(len(sampled_y)),
            level="frame",
            n_bootstrap=200,
            random_state=7,
        )
        self.assertEqual(set(sizes), {4.0, 5.0, 6.0})

    def test_cluster_bootstrap_is_reproducible_and_reports_both_levels(self):
        y_true = np.repeat([0, 0, 1, 1], 2)
        probability = np.array([0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95])
        groups = np.repeat(["a", "b", "c", "d"], 2)
        first = cluster_bootstrap_ci(
            y_true, probability, groups, n_bootstrap=250, random_state=11
        )
        second = cluster_bootstrap_ci(
            y_true, probability, groups, n_bootstrap=250, random_state=11
        )
        self.assertEqual(first, second)

        summary = evaluate_frame_and_group_metrics(
            y_true,
            probability,
            groups,
            threshold=0.5,
            n_bootstrap=100,
            random_state=4,
        )
        self.assertEqual(summary["frame"]["n"], 8)
        self.assertEqual(summary["group"]["n"], 4)
        self.assertEqual(summary["cluster_bootstrap"]["frame"]["resampling_unit"], "group")
        self.assertEqual(summary["cluster_bootstrap"]["group"]["resampling_unit"], "group")
        self.assertGreater(summary["cluster_bootstrap"]["frame"]["n_valid_bootstrap"], 0)


if __name__ == "__main__":
    unittest.main()
