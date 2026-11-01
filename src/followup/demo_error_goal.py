import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .common import reconstruct_samples, select_error_candidate, trajectory_errors

ROOT = Path(__file__).resolve().parents[2]
MIN_GAP_S = 0.1
OVERLAP_FRACTION = 0.25


def plot_step(times, original, before, after, errors_before, errors_after, added, output):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for coordinate, axis in enumerate(axes[:3]):
        axis.plot(times, original[:, coordinate], color='tab:blue', label='Original', linewidth=0.8)
        axis.plot(times, before[:, coordinate], '--', color='tab:orange', label='Before: 1 goal')
        axis.plot(times, after[:, coordinate], color='tab:green', label='After: 2 goals', linewidth=1)
        axis.scatter(times[added], original[added, coordinate], color='black', s=22,
                     label='Added source sample', zorder=4)
        axis.set_ylabel(f'{"XYZ"[coordinate]} (m)')
    axes[0].set_title('First error-driven insertion (not the full method comparison)')
    axes[0].legend(fontsize=8, ncol=2)
    axes[3].plot(times, errors_before * 100, '--', color='tab:orange', label='Before')
    axes[3].plot(times, errors_after * 100, color='tab:green', label='After')
    axes[3].set_ylabel('Position error (cm)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    for axis in axes:
        axis.axvline(times[added], color='black', linewidth=0.7, alpha=0.4)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    with np.load(ROOT / 'results/wrist_trajectory.npz') as data:
        original = data['positions'].astype(np.float64)
        times = data['timestamps']

    before_indices = np.array([0, len(times) - 1])
    before, before_activation, before_completion = reconstruct_samples(
        original, times, before_indices, OVERLAP_FRACTION
    )
    before_metrics, errors_before = trajectory_errors(original, before, times)
    added = select_error_candidate(errors_before, times, before_indices, MIN_GAP_S)

    after_indices = np.sort(np.r_[before_indices, added])
    after, after_activation, after_completion = reconstruct_samples(
        original, times, after_indices, OVERLAP_FRACTION
    )
    after_metrics, errors_after = trajectory_errors(original, after, times)
    for positions in (before, after):
        np.testing.assert_allclose(positions[[0, -1]], original[[0, -1]], rtol=0, atol=1e-12)

    output = ROOT / 'results/followup/first_insertion'
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'purpose': 'One insertion demonstration; not a three-method comparison',
        'selection': 'Largest 3D position error among eligible source frames',
        'min_source_time_gap_s': MIN_GAP_S,
        'overlap_fraction': OVERLAP_FRACTION,
        'goal_count_convention': 'Excludes initial position; includes final position',
        'added_frame': added,
        'added_source_time_s': float(times[added]),
        'before': {'goal_count': 1, **before_metrics},
        'after': {'goal_count': 2, **after_metrics},
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(
        output / 'trajectories.npz', timestamps=times,
        before_positions=before, after_positions=after,
        before_indices=before_indices, after_indices=after_indices,
        before_activation=before_activation, before_completion=before_completion,
        after_activation=after_activation, after_completion=after_completion,
    )
    plot_step(times, original, before, after, errors_before, errors_after, added,
              output / 'comparison.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
