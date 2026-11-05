import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .common import reconstruct_samples, trajectory_errors
from .motion_change import motion_scores, select_motion_goals

ROOT = Path(__file__).resolve().parents[2]
MAX_GOALS = 102
MIN_GAP_S = 0.1
OVERLAP_FRACTION = 0.25
METHODS = {
    'fixed_interval': ('Fixed interval (shared timing)', 'tab:blue'),
    'error_first_improvement': ('Error: first improvement', 'tab:orange'),
    'motion_acceleration': ('Motion: raw acceleration magnitude', 'tab:green'),
}


class BudgetUnavailable(ValueError):
    """A selector cannot supply this goal count under the common constraints."""


def uniform_indices(times, goal_count, min_gap_s=MIN_GAP_S):
    if not isinstance(goal_count, int) or goal_count < 1:
        raise ValueError('Goal count must be a positive integer')
    if goal_count >= len(times):
        raise BudgetUnavailable('Not enough source frames')
    indices = np.rint(np.linspace(0, len(times) - 1, goal_count + 1)).astype(int)
    if np.any(np.diff(times[indices]) < min_gap_s - 1e-12):
        raise BudgetUnavailable('Uniform source samples violate the minimum gap')
    return indices


def motion_indices(order, frame_count, goal_count):
    if not isinstance(goal_count, int) or goal_count < 1:
        raise ValueError('Goal count must be a positive integer')
    if len(order) < goal_count - 1:
        raise BudgetUnavailable('Not enough eligible motion peaks')
    return np.sort(np.r_[0, order[:goal_count - 1], frame_count - 1]).astype(int)


def plot_errors(rows, output):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    metrics = [('position_rmse_m', 100, 'Position RMSE (cm)'),
               ('velocity_rmse_m_s', 1, 'Velocity RMSE (m/s)'),
               ('max_position_error_m', 100, 'Maximum position error (cm)')]
    for axis, (key, scale, label) in zip(axes, metrics):
        for method, (name, color) in METHODS.items():
            selected = [r for r in rows if r['method'] == method]
            axis.plot([r['goal_count'] for r in selected],
                      [r[key] * scale for r in selected], label=name, color=color)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].set_title('Matched goal counts: same clip, timing rule, and evaluation')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Goals (initial excluded, final included)')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_positions(times, original, final_positions, output, window=None):
    mask = np.ones(len(times), dtype=bool)
    if window is not None:
        mask = (times >= window[0]) & (times <= window[1])
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for coordinate, axis in enumerate(axes):
        axis.plot(times[mask], original[mask, coordinate], color='black',
                  label='Original', linewidth=1.1)
        for method, values in final_positions.items():
            name, color = METHODS[method]
            axis.plot(times[mask], values[mask, coordinate], label=name, color=color,
                      linewidth=0.9, linestyle='--')
        axis.set_ylabel(f'{"XYZ"[coordinate]} (m)')
        axis.grid(alpha=0.2)
    axes[0].set_title(f'{MAX_GOALS} goals per method' +
                      ('' if window is None else f': {window[0]}-{window[1]} s'))
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    source_path = ROOT / 'results/wrist_trajectory.npz'
    with np.load(source_path) as source:
        original = source['positions'].astype(np.float64)
        times = source['timestamps']
    trajectory_errors(original, original, times)
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=0, atol=1e-12):
        raise ValueError('Uniform frame-index sampling requires uniform timestamps')

    accepted_dir = ROOT / 'results/followup/accepted_error_insertion'
    accepted_summary = json.loads((accepted_dir / 'summary.json').read_text(encoding='utf-8'))
    if (accepted_summary['min_source_time_gap_s'] != MIN_GAP_S
            or accepted_summary['overlap_fraction'] != OVERLAP_FRACTION):
        raise ValueError('Cached error method settings differ; rerun it before comparing')
    accepted_history = {r['goal_count']: r for r in accepted_summary['history']}
    scores, _, _ = motion_scores(original, times)
    _, _, motion_order = select_motion_goals(scores, times, MAX_GOALS, MIN_GAP_S)

    rows, unavailable = [], []
    final_positions = {}
    snapshots = {'timestamps': times}
    snapshot_counts = {1, 2, 10, 20, 40, 60, 80, MAX_GOALS}
    with np.load(accepted_dir / 'snapshots.npz') as cached:
        for count in range(1, MAX_GOALS + 1):
            for method in METHODS:
                try:
                    if method == 'fixed_interval':
                        indices = uniform_indices(times, count)
                    elif method == 'motion_acceleration':
                        indices = motion_indices(motion_order, len(times), count)
                    else:
                        if count not in accepted_history:
                            raise BudgetUnavailable('Error method stopped before this count')
                        indices = cached[f'indices_{count}']
                    if len(indices) != count + 1:
                        raise AssertionError('Mismatched goal count')
                    if np.any(np.diff(times[indices]) < MIN_GAP_S - 1e-12):
                        raise AssertionError('Selected source samples violate minimum gap')
                    positions, activation, completion = reconstruct_samples(
                        original, times, indices, OVERLAP_FRACTION
                    )
                    np.testing.assert_allclose(positions[[0, -1]], original[[0, -1]],
                                               rtol=0, atol=1e-12)
                    metrics, _ = trajectory_errors(original, positions, times)
                    if method == 'error_first_improvement':
                        np.testing.assert_allclose(positions, cached[f'positions_{count}'],
                                                   rtol=0, atol=1e-12)
                        for key, value in metrics.items():
                            np.testing.assert_allclose(value, accepted_history[count][key],
                                                       rtol=0, atol=1e-12)
                    rows.append({'method': method, 'goal_count': count, **metrics})
                    if count in snapshot_counts:
                        prefix = f'{method}_{count}'
                        snapshots[prefix + '_indices'] = indices
                        snapshots[prefix + '_positions'] = positions
                        snapshots[prefix + '_activation'] = activation
                        snapshots[prefix + '_completion'] = completion
                    if count == MAX_GOALS:
                        final_positions[method] = positions
                except BudgetUnavailable as error:
                    unavailable.append({'method': method, 'goal_count': count, 'reason': str(error)})

    output = ROOT / 'results/followup/comparison'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'goal_counts': list(range(1, MAX_GOALS + 1)),
        'goal_count_convention': 'Excludes initial position; includes final position',
        'min_source_time_gap_s': MIN_GAP_S,
        'overlap_fraction': OVERLAP_FRACTION,
        'motion_score': 'Raw acceleration magnitude; no smoothing or error acceptance',
        'fixed_selection': 'Evenly spaced source-frame indices, rounded to nearest integer',
        'fixed_baseline': 'Recomputed with shared timing; not the original stage-one output',
        'error_selection': 'Previously recorded first-improvement positions and scores revalidated',
        'error_candidate_trials': accepted_summary['total_candidate_trials'],
        'evaluation': 'All raw source timestamps, same numerical velocity differentiation',
        'fairness_scope': 'Equal representation size, not equal computation or objective',
        'unavailable': unavailable,
        'metrics': rows,
    }
    (output / 'summary.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(output / 'snapshots.npz', **snapshots)
    plot_errors(rows, output / 'error_vs_goals.png')
    if len(final_positions) == len(METHODS):
        plot_positions(times, original, final_positions, output / 'position_comparison.png')
        plot_positions(times, original, final_positions, output / 'position_detail.png', (17, 23))
    for row in rows:
        if row['goal_count'] in {20, 60, MAX_GOALS}:
            print(row)
    print(f'Saved {len(rows)} evaluated states; unavailable: {len(unavailable)}; {output}')


if __name__ == '__main__':
    main()
