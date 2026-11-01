import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import trajectory_errors
from .compare_methods import METHODS

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (10, 20, 40, 60, 80, 102)
WINDOW = (17.0, 23.0)


def plot_count_overlays():
    root = ROOT
    out = root / 'results/followup/goal_count_analysis'
    methods = {'fixed_interval': 'Fixed interval', 'error_first_improvement': 'Error driven first improvement', 'motion_acceleration': 'Raw acceleration magnitude'}
    counts = (10, 20, 40, 60, 80, 102)
    colors = {10: 'tab:blue', 20: 'tab:orange', 40: 'tab:green', 60: 'tab:purple', 80: 'tab:brown', 102: 'tab:red'}
    with np.load(root / 'results/wrist_trajectory.npz') as z:
        times = z['timestamps']
        original = z['positions'].astype(float)
    mask = (times >= 17) & (times <= 23)
    with np.load(root / 'results/followup/comparison/snapshots.npz') as z:
        positions = {(m, k): z[f'{m}_{k}_positions'].copy() for m in methods for k in counts}
    velocities = {key: np.gradient(p, times, axis=0, edge_order=2) for (key, p) in positions.items()}
    original_v = np.gradient(original, times, axis=0, edge_order=2)
    limits = []
    for (data, reference) in [(positions, original), (velocities, original_v)]:
        limits.append([])
        for coordinate in range(3):
            a = np.concatenate([reference[mask, coordinate]] + [p[mask, coordinate] for p in data.values()])
            (lo, hi) = (a.min(), a.max())
            pad = max((hi - lo) * 0.07, 0.0001)
            limits[-1].append((lo - pad, hi + pad))
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11, 'legend.fontsize': 10})
    for (method, title) in methods.items():
        for (quantity, data, reference, unit, index) in [('position', positions, original, 'm', 0), ('velocity', velocities, original_v, 'm/s', 1)]:
            (fig, axes) = plt.subplots(3, 2, figsize=(10.5, 6.5), sharex=True)
            for (col, group) in enumerate((counts[:3], counts[3:])):
                for (coordinate, name) in enumerate('XYZ'):
                    ax = axes[coordinate, col]
                    ax.plot(times[mask], reference[mask, coordinate], color='black', linewidth=1.4, label='Original')
                    for count in group:
                        ax.plot(times[mask], data[method, count][mask, coordinate], color=colors[count], linewidth=1.1, label=f'{count} goals', linestyle='--')
                    ax.set_xlim(17, 23)
                    ax.set_ylim(*limits[index][coordinate])
                    ax.set_ylabel(f'{name} ({unit})')
                    ax.grid(alpha=0.2)
                axes[0, col].set_title(', '.join(map(str, group)) + ' goals', fontsize=12)
                axes[0, col].legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
                axes[-1, col].set_xlabel('Time (s)')
            fig.suptitle(title + ' | ' + quantity.capitalize() + ' | 17-23 s', y=0.995, fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.9))
            fig.savefig(out / f'by_count_{method}_{quantity}.png', dpi=220)
            plt.close(fig)


def main():
    folder = ROOT / 'results/followup/comparison'
    output = ROOT / 'results/followup/goal_count_analysis'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / 'results/wrist_trajectory.npz') as source:
        original = source['positions'].astype(float)
        times = source['timestamps']
    with (folder / 'metrics.csv').open(newline='') as stream:
        recorded = {(r['method'], int(r['goal_count'])): r
                    for r in csv.DictReader(stream)}
    velocities = {'original': np.gradient(original, times, axis=0, edge_order=2)}
    positions = {'original': original}
    rows = []
    mask = (times >= WINDOW[0]) & (times <= WINDOW[1])
    with np.load(folder / 'snapshots.npz') as snapshots:
        np.testing.assert_array_equal(times, snapshots['timestamps'])
        for count in COUNTS:
            for method in METHODS:
                key = f'{method}_{count}'
                positions[key] = snapshots[key + '_positions'].copy()
                velocities[key] = np.gradient(positions[key], times, axis=0, edge_order=2)
                metrics, _ = trajectory_errors(original, positions[key], times)
                for name, value in metrics.items():
                    np.testing.assert_allclose(value, float(recorded[method, count][name]),
                                               rtol=0, atol=1e-12)
                local_p = positions[key][mask] - original[mask]
                local_v = velocities[key][mask] - velocities['original'][mask]
                rows.append({'method': method, 'goal_count': count, **metrics,
                             'window_position_rmse_m': float(np.sqrt(np.mean(np.sum(local_p**2, axis=1)))),
                             'window_velocity_rmse_m_s': float(np.sqrt(np.mean(np.sum(local_v**2, axis=1))))})

    limits = []
    for data in (positions, velocities):
        group = []
        for coordinate in range(3):
            values = np.concatenate([array[mask, coordinate] for array in data.values()])
            low, high = values.min(), values.max()
            padding = max((high - low) * 0.07, 1e-4)
            group.append((float(low - padding), float(high + padding)))
        limits.append(group)

    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12,
                         'axes.labelsize': 11, 'legend.fontsize': 10})
    for count in COUNTS:
        fig, axes = plt.subplots(3, 2, figsize=(10.5, 6.3), sharex=True)
        for coordinate, axis_name in enumerate('XYZ'):
            for column, (data, unit) in enumerate(((positions, 'm'), (velocities, 'm/s'))):
                axis = axes[coordinate, column]
                axis.plot(times[mask], data['original'][mask, coordinate], color='black',
                          linewidth=1.35, label='Original')
                for method, (_, color) in METHODS.items():
                    axis.plot(times[mask], data[f'{method}_{count}'][mask, coordinate],
                              color=color, linewidth=1.05, linestyle='--',
                              label={'fixed_interval': 'Fixed interval',
                                     'error_first_improvement': 'Error driven',
                                     'motion_acceleration': 'Acceleration'}[method])
                axis.set_ylim(*limits[column][coordinate])
                axis.set_xlim(*WINDOW)
                axis.set_ylabel(f'{axis_name} ({unit})')
                axis.grid(alpha=0.2)
        axes[0, 0].set_title('Position')
        axes[0, 1].set_title('Velocity')
        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Time (s)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(f'{count} goals per method: 17-23 s', y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        fig.savefig(output / f'comparison_{count:03d}.png', dpi=220)
        plt.close(fig)
    plot_count_overlays()
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    settings = {'goal_counts': COUNTS, 'window_s': WINDOW,
                'input': 'Existing comparison snapshots; no goal selection rerun',
                'velocity': 'numpy.gradient on the full clip before taking the display window; edge_order=2',
                'axis_limits': 'Identical across all six goal counts for each coordinate and quantity',
                'evaluation': 'Global metrics use the full clip; window metrics are separately labelled',
                'position_limits_m': limits[0], 'velocity_limits_m_s': limits[1],
                'records': rows}
    (output / 'summary.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    print('Verified 18 saved trajectories against the original global metrics.')
    print('Created six per-count views, six within-method comparisons, and a separate analysis table.')
    for row in rows:
        print(row['goal_count'], row['method'],
              'window position cm:', round(row['window_position_rmse_m'] * 100, 3),
              'window velocity m/s:', round(row['window_velocity_rmse_m_s'], 5))


if __name__ == '__main__':
    main()
