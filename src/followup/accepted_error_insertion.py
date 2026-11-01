import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import reconstruct_samples, trajectory_errors
from .error_insertion import plot_history

ROOT = Path(__file__).resolve().parents[2]


def evaluate(original, times, indices, overlap_fraction):
    positions, activation, completion = reconstruct_samples(
        original, times, indices, overlap_fraction
    )
    metrics, errors = trajectory_errors(original, positions, times)
    return positions, activation, completion, metrics, errors


def ranked_candidates(errors, times, indices, min_gap_s):
    eligible = np.ones(len(times), dtype=bool)
    for index in indices:
        eligible &= np.abs(times - times[index]) >= min_gap_s - 1e-12
    eligible[indices] = False
    candidates = np.flatnonzero(eligible)
    return candidates[np.argsort(-errors[candidates], kind='stable')]


def fit(original, times, max_goals=102, min_gap_s=0.1, overlap_fraction=0.25,
        improvement_atol_m=1e-10, progress=None):
    if not isinstance(max_goals, int) or max_goals < 1:
        raise ValueError('Maximum goal count must be a positive integer')
    if not np.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError('Minimum source-time gap must be positive and finite')
    if not np.isfinite(improvement_atol_m) or improvement_atol_m < 0:
        raise ValueError('Improvement tolerance must be nonnegative and finite')

    indices = np.array([0, len(times) - 1])
    if times[-1] - times[0] < min_gap_s - 1e-12:
        raise ValueError('Initial and final samples violate the minimum gap')
    state = evaluate(original, times, indices, overlap_fraction)
    history = [{'goal_count': 1, 'added_frame': None, 'added_source_time_s': None,
                'candidate_rank': None, 'trials_in_round': 0, **state[3]}]
    trials = []
    snapshots = {'timestamps': times}
    save_state(snapshots, 1, indices, state)
    stop_reason = 'goal_budget'

    while len(indices) - 1 < max_goals:
        current_count = len(indices) - 1
        candidates = ranked_candidates(state[4], times, indices, min_gap_s)
        if not len(candidates):
            stop_reason = 'no_eligible_source_frame'
            break

        accepted = False
        for rank, candidate in enumerate(candidates, start=1):
            trial_indices = np.sort(np.r_[indices, candidate])
            trial_state = evaluate(original, times, trial_indices, overlap_fraction)
            improvement = state[3]['position_rmse_m'] - trial_state[3]['position_rmse_m']
            accepted = improvement > improvement_atol_m
            trials.append({
                'from_goal_count': current_count,
                'candidate_rank': rank,
                'candidate_frame': int(candidate),
                'candidate_time_s': float(times[candidate]),
                'current_point_error_m': float(state[4][candidate]),
                'before_position_rmse_m': state[3]['position_rmse_m'],
                'trial_position_rmse_m': trial_state[3]['position_rmse_m'],
                'improvement_m': improvement,
                'accepted': accepted,
            })
            if accepted:
                indices, state = trial_indices, trial_state
                count = len(indices) - 1
                history.append({
                    'goal_count': count,
                    'added_frame': int(candidate),
                    'added_source_time_s': float(times[candidate]),
                    'candidate_rank': rank,
                    'trials_in_round': rank,
                    **state[3],
                })
                save_state(snapshots, count, indices, state)
                if progress is not None:
                    progress(history[-1])
                break
        if not accepted:
            stop_reason = 'no_improving_candidate'
            break

    return history, trials, snapshots, stop_reason


def save_state(snapshots, count, indices, state):
    positions, activation, completion, _, _ = state
    snapshots[f'indices_{count}'] = indices.copy()
    snapshots[f'positions_{count}'] = positions.copy()
    snapshots[f'activation_{count}'] = activation.copy()
    snapshots[f'completion_{count}'] = completion.copy()


def write_csv(path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_progress(record):
    count = record['goal_count']
    if count <= 3 or count % 10 == 0:
        print(f"Accepted {count} goals: position RMSE "
              f"{record['position_rmse_m']*100:.3f} cm; "
              f"candidate rank {record['candidate_rank']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-goals', type=int, default=102)
    parser.add_argument('--min-gap', type=float, default=0.1)
    parser.add_argument('--overlap-fraction', type=float, default=0.25)
    parser.add_argument('--improvement-atol', type=float, default=1e-10,
                        help='Required absolute decrease in position RMSE (m)')
    args = parser.parse_args()
    with np.load(ROOT / 'results/wrist_trajectory.npz') as source:
        original = source['positions'].astype(np.float64)
        times = source['timestamps']

    history, trials, snapshots, stop_reason = fit(
        original, times, args.max_goals, args.min_gap, args.overlap_fraction,
        args.improvement_atol, report_progress,
    )
    output = ROOT / 'results/followup/accepted_error_insertion'
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'history.csv', history, list(history[0]))
    trial_fields = ['from_goal_count', 'candidate_rank', 'candidate_frame',
                    'candidate_time_s', 'current_point_error_m',
                    'before_position_rmse_m', 'trial_position_rmse_m',
                    'improvement_m', 'accepted']
    write_csv(output / 'trials.csv', trials, trial_fields)
    settings = {
        'method': 'First improving candidate in descending current position-error order',
        'tie_break': 'Earlier source frame first',
        'requested_max_goals': args.max_goals,
        'reached_goals': history[-1]['goal_count'],
        'stop_reason': stop_reason,
        'min_source_time_gap_s': args.min_gap,
        'overlap_fraction': args.overlap_fraction,
        'improvement_atol_m': args.improvement_atol,
        'accuracy_threshold': None,
        'acceptance_metric': 'Global 3D position RMSE over all source timestamps',
        'goal_count_convention': 'Excludes initial position; includes final position',
        'total_candidate_trials': len(trials),
        'rejected_trials': sum(not row['accepted'] for row in trials),
        'history': history,
    }
    (output / 'summary.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(output / 'snapshots.npz', **snapshots)
    plot_history(history, output / 'error_vs_goals.png',
                 title='First-improvement insertion: accepted states only')
    print(f"Stopped: {stop_reason}; reached {history[-1]['goal_count']} goals; "
          f"tried {len(trials)} candidates")
    print(f'Final metrics: {history[-1]}')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
