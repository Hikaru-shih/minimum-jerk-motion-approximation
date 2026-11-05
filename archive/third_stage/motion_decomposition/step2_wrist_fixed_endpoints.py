import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import fit_fixed_endpoints
from .metrics import trajectory_errors

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def plot_comparison(times, original, sampled, fitted, path, window=None):
    positions = (original, sampled, fitted)
    velocities = tuple(np.gradient(p, times, axis=0, edge_order=2) for p in positions)
    labels = ('Original', 'Sampled goals', 'Fitted goals; fixed endpoints')
    colors = ('black', 'tab:blue', 'tab:orange')
    mask = np.ones(len(times), dtype=bool)
    if window is not None:
        mask = (times >= window[0]) & (times <= window[1])
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for coordinate, name in enumerate('XYZ'):
        for column, (arrays, unit) in enumerate(((positions, 'm'), (velocities, 'm/s'))):
            axis = axes[coordinate, column]
            for i, (array, label, color) in enumerate(zip(arrays, labels, colors)):
                axis.plot(times[mask], array[mask, coordinate], color=color,
                          linestyle='-' if i == 0 else '--', linewidth=1, label=label)
            axis.set_ylabel(f'{name} ({unit})')
            axis.grid(alpha=0.2)
    axes[0, 0].set_title('Position')
    axes[0, 1].set_title('Velocity')
    for axis in axes[-1]:
        axis.set_xlabel('Time (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.965))
    scope = 'full clip' if window is None else f'{window[0]}-{window[1]} s'
    fig.suptitle(f'Fixed initial and final positions; same 102 components and times; {scope}', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    source_path = PROJECT_ROOT / 'results/wrist_trajectory.npz'
    snapshot_path = PROJECT_ROOT / 'results/followup/comparison/snapshots.npz'
    metrics_path = PROJECT_ROOT / 'results/followup/comparison/metrics.csv'
    output = ROOT / 'results/step2_wrist_fixed_endpoints'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source_path) as source:
        times = source['timestamps'].copy()
        original = source['positions'].astype(float)
    with np.load(snapshot_path) as snapshot:
        np.testing.assert_array_equal(times, snapshot['timestamps'])
        indices = snapshot['fixed_interval_102_indices'].copy()
        activation = snapshot['fixed_interval_102_activation'].copy()
        completion = snapshot['fixed_interval_102_completion'].copy()
        sampled = snapshot['fixed_interval_102_positions'].copy()

    initial, final = original[0].copy(), original[-1].copy()
    fitted = fit_fixed_endpoints(times, original, initial, final, activation, completion)
    basis = fitted['basis']
    sampled_displacements = np.diff(original[indices], axis=0)
    np.testing.assert_allclose(initial + basis @ sampled_displacements, sampled, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sampled[[0, -1]], original[[0, -1]], rtol=0, atol=1e-12)
    assert len(indices) - 1 == len(activation) == 102
    assert fitted['full_column_rank'] and fitted['free_columns'] == 101
    assert fitted['initial_error_m'] < 1e-12 and fitted['final_error_m'] < 1e-12
    assert fitted['displacement_sum_error_m'] < 1e-12

    delta = final - initial
    other_basis = basis[:, 1:] - basis[:, :1]
    other_target = original - initial - basis[:, :1] * delta
    remaining, _, _, _ = np.linalg.lstsq(other_basis, other_target, rcond=None)
    other_displacements = np.vstack([delta - remaining.sum(axis=0), remaining])
    other_positions = initial + basis @ other_displacements
    np.testing.assert_allclose(fitted['positions'], other_positions, rtol=0, atol=1e-11)
    independent_max_difference = float(np.max(np.abs(fitted['positions'] - other_positions)))

    records = []
    for method, positions in [('sampled_goals', sampled), ('fitted_goals_fixed_endpoints', fitted['positions'])]:
        metrics, _ = trajectory_errors(original, positions, times)
        records.append({
            'method': method, 'goal_count': 102, **metrics,
            'initial_error_m': float(np.linalg.norm(positions[0] - initial)),
            'final_error_m': float(np.linalg.norm(positions[-1] - final)),
        })
    with metrics_path.open(newline='') as stream:
        previous = next(row for row in csv.DictReader(stream)
                        if row['method'] == 'fixed_interval' and row['goal_count'] == '102')
    for metric in ('position_rmse_m', 'velocity_rmse_m_s', 'max_position_error_m'):
        np.testing.assert_allclose(records[0][metric], float(previous[metric]), rtol=0, atol=1e-12)
    assert records[1]['position_rmse_m'] <= records[0]['position_rmse_m'] + 1e-12
    residual = fitted['positions'] - original
    relative_stationarity = float(
        np.linalg.norm(fitted['free_basis'].T @ residual)
        / (np.linalg.norm(fitted['free_basis']) * np.linalg.norm(residual))
    )
    assert relative_stationarity < 1e-10
    shifts = np.linalg.norm(fitted['goals'] - original[indices[1:]], axis=1)
    summary = {
        'scope': 'Position fitting on one real wrist clip with fixed timing and exact endpoint equalities',
        'baseline': 'Saved fixed_interval_102 matched-count reconstruction, not the original stage-one baseline',
        'frames': len(times), 'goal_count': 102,
        'initial_target_m': initial.tolist(), 'final_target_m': final.tolist(),
        'endpoint_target_source': 'First and last observed wrist positions; not independently annotated task targets',
        'fixed': ['initial position', 'final position', 'activation times', 'completion times', 'component count'],
        'fitted': '102 XYZ displacements with their sum fixed; 101 free displacement vectors',
        'objective': 'Unweighted full-clip position squared error',
        'constraint_method': 'Eliminate the last displacement before least squares; no endpoint overwriting or penalty weight',
        'velocity': 'numpy.gradient over all original timestamps, axis=0, edge_order=2; evaluation only',
        'free_rank': fitted['free_rank'], 'free_columns': fitted['free_columns'],
        'condition_number_of_reduced_basis': fitted['condition_number'],
        'condition_number_note': 'Depends on the chosen elimination coordinates; not a parameter-stability test',
        'relative_stationarity_residual': relative_stationarity,
        'displacement_sum_error_m': fitted['displacement_sum_error_m'],
        'independent_first_component_elimination_max_difference_m': independent_max_difference,
        'mean_goal_shift_m': float(np.mean(shifts)), 'maximum_goal_shift_m': float(np.max(shifts)),
        'input_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in [('wrist', source_path), ('snapshots', snapshot_path), ('metrics', metrics_path)]},
        'metrics': records,
        'limitations': [
            'Timing is inherited from the fixed-interval baseline; not fitted',
            'No movement-unit identification or component-count selection',
            'No velocity objective, cancellation penalty, or biomechanical feasibility validation',
            'No generalization or perturbation-stability evaluation',
            'The endpoint velocity is not fitted to the measured endpoint velocity',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(
        output / 'fit.npz', timestamps=times, original=original,
        initial_target=initial, final_target=final, source_indices=indices,
        activation=activation, completion=completion,
        sampled_positions=sampled, fitted_positions=fitted['positions'],
        sampled_goals=original[indices[1:]], fitted_goals=fitted['goals'],
        sampled_displacements=sampled_displacements, fitted_displacements=fitted['displacements'],
        free_singular_values=fitted['singular_values'],
    )
    plot_comparison(times, original, sampled, fitted['positions'], output / 'comparison_full.png')
    plot_comparison(times, original, sampled, fitted['positions'], output / 'comparison_detail.png', (17, 23))
    for record in records:
        print(record['method'],
              f"position RMSE={record['position_rmse_m'] * 100:.6f} cm;",
              f"velocity RMSE={record['velocity_rmse_m_s']:.6f} m/s;",
              f"max error={record['max_position_error_m'] * 100:.6f} cm;",
              f"endpoint errors={record['initial_error_m']:.3e}, {record['final_error_m']:.3e} m")
    print('Free rank:', fitted['free_rank'], '/ 101; reduced-basis condition number:', fitted['condition_number'])
    print('Independent elimination max difference (m):', independent_max_difference)
    print('Relative stationarity residual:', relative_stationarity)
    print('Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
