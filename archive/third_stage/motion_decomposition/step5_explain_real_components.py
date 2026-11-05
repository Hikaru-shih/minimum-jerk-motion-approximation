import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'results/step5_real_window_fit'
    output = ROOT / 'results/step5_real_components'
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
    index = summary['best_timing_index']
    with np.load(source / 'fits.npz') as saved:
        times, original = saved['timestamps'].copy(), saved['original'].copy()
        fitted = saved[f'fitted_{index}_positions'].copy()
        activation = saved[f'fitted_{index}_activation'].copy()
        completion = saved[f'fitted_{index}_completion'].copy()
        displacements = saved[f'fitted_{index}_displacements'].copy()
    basis = design_matrix(times, activation, completion)
    contributions = basis[:, :, None] * displacements[None, :, :]
    assert np.allclose(original[0] + contributions.sum(axis=1), fitted, atol=1e-12, rtol=0)
    residual = fitted-original
    error = np.linalg.norm(residual, axis=1)
    peak = int(np.argmax(error))
    masks = [times < activation[0],
             (times >= activation[0]) & (times <= completion[0]),
             (times > completion[0]) & (times < activation[1]),
             times >= activation[1]]
    names = ['Before component 1', 'During component 1', 'Between components', 'During component 2']
    regions = []
    for name, mask in zip(names, masks):
        regions.append({'region': name, 'sample_count': int(mask.sum()),
                        'position_rmse_m': float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None})
    goals = original[0] + np.cumsum(displacements, axis=0)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    for c in range(3):
        axes[c, 0].plot(times, original[:, c]-original[0, c], color='gray', label='Original minus start')
        for k in range(2):
            axes[c, 0].plot(times, contributions[:, k, c], label=f'Component {k+1} contribution')
        axes[c, 0].plot(times, fitted[:, c]-original[0, c], '--', color='black', label='Sum')
        axes[c, 0].set_ylabel(f'{"XYZ"[c]} displacement (m)')
        axes[c, 1].plot(times, 100*residual[:, c], color='tab:red')
        axes[c, 1].axhline(0, color='gray', linewidth=0.6)
        axes[c, 1].set_ylabel(f'{"XYZ"[c]} residual (cm)')
    for k in range(2):
        axes[3, 0].plot(times, basis[:, k], label=f'Progress {k+1}')
    axes[3, 0].set_ylabel('Progress (0 to 1)')
    axes[3, 0].legend(fontsize=8)
    axes[3, 1].plot(times, 100*error, color='black')
    axes[3, 1].scatter(times[peak], 100*error[peak], color='tab:red')
    axes[3, 1].set_ylabel('Position error norm (cm)')
    for axis in axes.flat:
        for a, b, color in zip(activation, completion, ['tab:blue', 'tab:orange']):
            axis.axvspan(a, b, color=color, alpha=0.07)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Selected K=2 fit: additive displacement contributions and remaining errors\n'
                 'Contributions are vectors added to the initial position, not separate wrist paths')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / 'contributions_and_errors.png', dpi=160)
    plt.close(fig)
    result = {'source_best_timing_index': index, 'displacements_m': displacements.tolist(),
              'cumulative_goals_m': goals.tolist(),
              'peak_error_time_s': float(times[peak]), 'peak_error_m': float(error[peak]),
              'peak_error_residual_xyz_m': residual[peak].tolist(),
              'gap_s': float(activation[1]-completion[0]), 'regions': regions,
              'regional_metrics_note': 'Different sample sets and durations; descriptive diagnostics, not matched performance comparisons',
              'new_optimization_performed': False}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
