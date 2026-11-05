from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.followup.compare_methods import BudgetUnavailable, motion_indices, uniform_indices


class ComparisonChecks(unittest.TestCase):
    def test_uniform_selection_has_exact_count_and_endpoints(self):
        times = np.arange(101) / 100
        np.testing.assert_array_equal(uniform_indices(times, 4), [0, 25, 50, 75, 100])
        for count in (1, 3, 7, 10):
            indices = uniform_indices(times, count, min_gap_s=0.05)
            self.assertEqual(len(indices) - 1, count)
            np.testing.assert_array_equal(indices[[0, -1]], [0, 100])

    def test_infeasible_uniform_budget_is_not_silently_changed(self):
        with self.assertRaises(BudgetUnavailable):
            uniform_indices(np.linspace(0, 1, 11), 6, min_gap_s=0.2)

    def test_motion_selection_uses_same_ranked_prefix(self):
        order = np.array([6, 2, 4])
        np.testing.assert_array_equal(motion_indices(order, 9, 1), [0, 8])
        np.testing.assert_array_equal(motion_indices(order, 9, 2), [0, 6, 8])
        np.testing.assert_array_equal(motion_indices(order, 9, 3), [0, 2, 6, 8])
        with self.assertRaises(BudgetUnavailable):
            motion_indices(order, 9, 5)


if __name__ == '__main__':
    unittest.main()
