import unittest

import numpy as np

from tracker_training.matching import confidence_greedy_match, localization_match
from tracker_training.metrics import evaluate_centroids


class MatchingMetricsTest(unittest.TestCase):
    def test_global_matching_avoids_greedy_crossing(self) -> None:
        predictions = np.asarray([[4.0, 0.0, 0.9], [0.0, 0.0, 0.8]])
        truth = np.asarray([[0.0, 0.0, 10.0, 10.0], [8.0, 0.0, 10.0, 10.0]])
        matches = localization_match(predictions, truth, max_normalized_distance=1.0)
        self.assertEqual(
            {(match.prediction_index, match.ground_truth_index) for match in matches},
            {(0, 1), (1, 0)},
        )

    def test_confidence_matching_marks_duplicate_false(self) -> None:
        predictions = np.asarray([[0.0, 0.0, 0.9], [0.1, 0.0, 0.8]])
        truth = np.asarray([[0.0, 0.0, 10.0, 10.0]])
        labels, matched = confidence_greedy_match(
            predictions, truth, max_normalized_distance=0.1
        )
        np.testing.assert_array_equal(labels, [True, False])
        np.testing.assert_array_equal(matched, [0, -1])

    def test_metrics_report_miss_and_duplicate(self) -> None:
        predictions = [np.asarray([[0.0, 0.0, 0.9], [0.2, 0.0, 0.8]])]
        truth = [
            np.asarray(
                [[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0]]
            )
        ]
        result = evaluate_centroids(predictions, truth, distance_thresholds=(0.1,))
        threshold = result["thresholds"]["0.100"]
        self.assertEqual(threshold["true_positive"], 1)
        self.assertEqual(threshold["false_positive"], 1)
        self.assertEqual(threshold["false_negative"], 1)
        self.assertAlmostEqual(threshold["precision"], 0.5)
        self.assertAlmostEqual(threshold["recall"], 0.5)
        self.assertAlmostEqual(threshold["ap"], 0.5)

    def test_localization_is_permutation_invariant(self) -> None:
        predictions = np.asarray([[1.0, 1.0, 0.7], [21.0, 1.0, 0.6]])
        truth = np.asarray([[0.0, 0.0, 10.0, 10.0], [20.0, 0.0, 10.0, 10.0]])
        first = evaluate_centroids([predictions], [truth])
        second = evaluate_centroids([predictions[::-1]], [truth[::-1]])
        self.assertEqual(first["pixel_error"], second["pixel_error"])
        self.assertEqual(first["normalized_error"], second["normalized_error"])

    def test_empty_ground_truth_has_undefined_ap(self) -> None:
        result = evaluate_centroids([np.empty((0, 3))], [np.empty((0, 4))])
        self.assertIsNone(result["thresholds"]["0.100"]["ap"])


if __name__ == "__main__":
    unittest.main()
