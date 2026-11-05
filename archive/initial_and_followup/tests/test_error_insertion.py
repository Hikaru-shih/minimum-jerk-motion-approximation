from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.followup.error_insertion import insertion_steps


class InsertionChecks(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0, 2, 9)
        peak = 1 - np.abs(self.times - 1)
        self.original = np.column_stack([peak, peak * 0, peak * 0])

    def test_first_insertion_finds_known_peak(self):
        steps = list(insertion_steps(self.original, self.times, max_goals=2))
        self.assertEqual([step[0]['goal_count'] for step in steps], [1, 2])
        self.assertEqual(steps[1][0]['added_frame'], 4)
        np.testing.assert_array_equal(steps[1][1], [0, 4, 8])
        np.testing.assert_allclose(steps[0][2], 0, atol=1e-12)

    def test_each_step_preserves_previous_samples_and_spacing(self):
        steps = list(insertion_steps(self.original, self.times, max_goals=5, min_gap_s=0.25))
        for i, (record, indices, positions, _, _) in enumerate(steps):
            self.assertEqual(len(indices) - 1, record['goal_count'])
            self.assertTrue(np.all(np.diff(self.times[indices]) >= 0.25))
            np.testing.assert_allclose(positions[[0, -1]], self.original[[0, -1]], atol=1e-12)
            if i:
                self.assertTrue(set(steps[i-1][1]).issubset(indices))

    def test_stops_when_spacing_blocks_all_new_candidates(self):
        steps = list(insertion_steps(self.original, self.times, max_goals=5, min_gap_s=1.1))
        self.assertEqual(len(steps), 1)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            list(insertion_steps(self.original, self.times, max_goals=0))


if __name__ == '__main__':
    unittest.main()
