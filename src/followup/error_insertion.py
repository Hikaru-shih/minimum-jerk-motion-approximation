import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .common import (
    NoEligibleGoal,
    reconstruct_samples,
    select_error_candidate,
    trajectory_errors,
)

ROOT = Path(__file__).resolve().parents[2]


def insertion_steps(original, times, max_goals=102, min_gap_s=0.1, overlap_fraction=0.25):
    if not isinstance(max_goals, int) or max_goals < 1:
        raise ValueError('Maximum goal count must be a positive integer')
    if not np.isfinite(min_gap_s) or min_gap_s <= 0:
        raise ValueError('Minimum source-time gap must be positive and finite')

    indices = np.array([0, len(times) - 1])
    if times[-1] - times[0] < min_gap_s - 1e-12:
        raise ValueError('Initial and final source samples violate the minimum gap')
    added = None

    for goal_count in range(1, max_goals + 1):
        positions, activation, completion = reconstruct_samples(
            original, times, indices, overlap_fraction
        )
        metrics, point_errors = trajectory_errors(original, positions, times)
        np.testing.assert_allclose(positions[[0, -1]], original[[0, -1]], rtol=0, atol=1e-12)
        record = {
            'goal_count': goal_count,
            'added_frame': added,
            'added_source_time_s': None if added is None else float(times[added]),
            **metrics,
        }
        yield record, indices.copy(), positions, activation, completion

        if goal_count == max_goals:
            break
        try:
            added = select_error_candidate(point_errors, times, indices, min_gap_s)
        except NoEligibleGoal:
            break
        indices = np.sort(np.r_[indices, added])


def plot_history(history, output, title='Error-driven insertion: every iteration (single method only)'):
    counts = [row['goal_count'] for row in history]
    plots = [
        ('position_rmse_m', 100, 'Position RMSE (cm)'),
        ('velocity_rmse_m_s', 1, 'Velocity RMSE (m/s)'),
        ('max_position_error_m', 100, 'Maximum position error (cm)'),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for axis, (key, scale, label) in zip(axes, plots):
        axis.plot(counts, [row[key] * scale for row in history], linewidth=1.2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel('Goals, excluding initial position and including final position')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-goals', type=int, default=102)
    parser.add_argument('--min-gap', type=float, default=0.1, help='Minimum source-time gap (s)')
    parser.add_argument('--overlap-fraction', type=float, default=0.25)
    args = parser.parse_args()

    with np.load(ROOT / 'results/wrist_trajectory.npz') as source:
        original = source['positions'].astype(np.float64)
        times = source['timestamps']

    history = []
    snapshots = {'timestamps': times}
    saved_counts = {1, 2, 10, 20, 40, 60, 80, args.max_goals}
    previous_rmse = None
    for record, indices, positions, activation, completion in insertion_steps(
        original, times, args.max_goals, args.min_gap, args.overlap_fraction
    ):
        record['position_rmse_increased'] = (
            previous_rmse is not None and record['position_rmse_m'] > previous_rmse
        )
        previous_rmse = record['position_rmse_m']
        history.append(record)
        count = record['goal_count']
        if count in saved_counts:
            snapshots[f'indices_{count}'] = indices
            snapshots[f'positions_{count}'] = positions
            snapshots[f'activation_{count}'] = activation
            snapshots[f'completion_{count}'] = completion
            print(f"Goals {count}: position RMSE {record['position_rmse_m']*100:.3f} cm; "
                  f"velocity RMSE {record['velocity_rmse_m_s']:.5f} m/s", flush=True)

    final_count = history[-1]['goal_count']
    snapshots[f'indices_{final_count}'] = indices
    snapshots[f'positions_{final_count}'] = positions
    snapshots[f'activation_{final_count}'] = activation
    snapshots[f'completion_{final_count}'] = completion
    output = ROOT / 'results/followup/error_insertion'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        'method': 'Largest eligible per-frame 3D position error; sampled positions',
        'requested_max_goals': args.max_goals,
        'reached_goals': final_count,
        'stop_reason': 'goal_budget' if final_count == args.max_goals else 'no_eligible_source_frame',
        'min_source_time_gap_s': args.min_gap,
        'overlap_fraction': args.overlap_fraction,
        'goal_count_convention': 'Excludes initial position; includes final position',
        'accuracy_threshold': None,
        'position_rmse_increase_steps': sum(row['position_rmse_increased'] for row in history),
        'history': history,
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(output / 'snapshots.npz', **snapshots)
    plot_history(history, output / 'error_vs_goals.png')
    print(f"Stopped: {summary['stop_reason']}; "
          f"position RMSE increased on {summary['position_rmse_increase_steps']} steps")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
