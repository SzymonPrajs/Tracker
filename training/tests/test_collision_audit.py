import unittest

import numpy as np

from tracker_training.collision_audit import audit_collisions, audit_strides


class CollisionAuditTest(unittest.TestCase):
    def test_same_cell_sets_representational_ceiling(self) -> None:
        centers = [np.asarray([[1.0, 1.0], [3.9, 3.9], [8.0, 8.0]])]
        result = audit_collisions(centers, output_stride=4)
        self.assertEqual(result["overflow_targets"], 1)
        self.assertEqual(result["collision_pairs"], 1)
        self.assertEqual(result["max_occupancy"], 2)
        self.assertAlmostEqual(result["theoretical_max_recall"], 2 / 3)

    def test_more_slots_remove_overflow_not_pair_count(self) -> None:
        centers = [np.asarray([[1.0, 1.0], [2.0, 2.0]])]
        result = audit_collisions(centers, output_stride=4, slots_per_cell=2)
        self.assertEqual(result["overflow_targets"], 0)
        self.assertEqual(result["collision_pairs"], 1)
        self.assertEqual(result["theoretical_max_recall"], 1.0)

    def test_stride_sweep_is_monotonic_for_fixture(self) -> None:
        centers = [np.asarray([[1.0, 1.0], [3.0, 3.0], [7.0, 7.0]])]
        results = audit_strides(centers, output_strides=(2, 4, 8))
        overflows = [results[stride]["overflow_targets"] for stride in (2, 4, 8)]
        self.assertEqual(overflows, sorted(overflows))

    def test_empty_dataset_has_no_recall_claim(self) -> None:
        result = audit_collisions([], output_stride=4)
        self.assertIsNone(result["theoretical_max_recall"])


if __name__ == "__main__":
    unittest.main()
