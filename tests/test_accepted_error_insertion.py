from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.followup.accepted_error_insertion import fit, ranked_candidates


class AcceptedInsertionChecks(unittest.TestCase):
    def test_accepts_triangle_peak_and_decreases_position_rmse(self):
        times = np.linspace(0, 2, 9)
        peak = 1 - np.abs(times - 1)
        original = np.column_stack([peak, peak * 0, peak * 0])
        history, trials, snapshots, reason = fit(original, times, max_goals=3)
        self.assertEqual(history[1]['added_frame'], 4)
        self.assertEqual(reason, 'goal_budget')
        self.assertTrue(all(a['position_rmse_m'] > b['position_rmse_m']
                            for a, b in zip(history, history[1:])))
        self.assertEqual(sum(row['accepted'] for row in trials), len(history) - 1)
        for count in range(1, len(history) + 1):
            indices = snapshots[f'indices_{count}']
            self.assertTrue(np.all(np.diff(times[indices]) >= 0.1 - 1e-12))
            np.testing.assert_allclose(snapshots[f'positions_{count}'][[0, -1]],
                                       original[[0, -1]], atol=1e-12)

    def test_all_rejected_trials_leave_initial_state_unchanged(self):
        times = np.arange(9, dtype=float)
        original = np.zeros((9, 3))
        original[4, 0] = 1
        history, trials, snapshots, reason = fit(original, times, max_goals=3)
        self.assertEqual(reason, 'no_improving_candidate')
        self.assertEqual(len(history), 1)
        self.assertEqual(len(trials), 7)
        self.assertEqual(trials[0]['candidate_frame'], 4)
        self.assertFalse(any(row['accepted'] for row in trials))
        np.testing.assert_array_equal(snapshots['indices_1'], [0, 8])
        np.testing.assert_array_equal(snapshots['positions_1'], np.zeros((9, 3)))

    def test_ranking_is_stable_and_obeys_spacing(self):
        times = np.arange(6, dtype=float)
        errors = np.array([100, 9, 8, 8, 7, 100])
        np.testing.assert_array_equal(ranked_candidates(errors, times, [0, 5], 1.5), [2, 3])

    def test_rejected_point_is_reconsidered_after_acceptance(self):
        times = np.arange(5, dtype=float)
        original = np.zeros((5, 3))

        def fake_evaluate(original, times, indices, fraction):
            key = tuple(indices)
            rmse = {(0, 4): 10, (0, 1, 4): 11, (0, 2, 4): 8,
                    (0, 1, 2, 4): 7}[key]
            metrics = {'position_rmse_m': rmse, 'velocity_rmse_m_s': 0,
                       'max_position_error_m': 0}
            return original.copy(), np.zeros(len(indices)-1), np.ones(len(indices)-1), \
                metrics, np.array([0, 9, 8, 7, 0])

        with patch('src.followup.accepted_error_insertion.evaluate', side_effect=fake_evaluate):
            history, trials, _, _ = fit(original, times, max_goals=3)
        self.assertEqual([row['candidate_frame'] for row in trials], [1, 2, 1])
        self.assertEqual([row['accepted'] for row in trials], [False, True, True])
        self.assertEqual([row['added_frame'] for row in history], [None, 2, 1])

    def test_no_eligible_frames_has_separate_stop_reason(self):
        times = np.linspace(0, 2, 9)
        history, trials, _, reason = fit(np.zeros((9, 3)), times, min_gap_s=1.1)
        self.assertEqual(reason, 'no_eligible_source_frame')
        self.assertEqual(len(history), 1)
        self.assertEqual(trials, [])

    def test_tolerance_rejects_tiny_improvement(self):
        times = np.linspace(0, 2, 9)
        peak = (1 - np.abs(times - 1)) * 1e-12
        original = np.column_stack([peak, peak * 0, peak * 0])
        history, _, _, reason = fit(original, times, max_goals=2)
        self.assertEqual(reason, 'no_improving_candidate')
        self.assertEqual(len(history), 1)


if __name__ == '__main__':
    unittest.main()
