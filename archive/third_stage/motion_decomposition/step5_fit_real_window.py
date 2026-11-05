import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import fit_fixed_endpoints
from .initialization import choose_best_result, make_initializations
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main():
    source = ROOT / 'results/step5_real_window/window.npz'
    output = ROOT / 'results/step5_real_window_fit'
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as data:
        times, positions = data['timestamps'].copy(), data['positions'].copy()
        reference_velocity = data['velocity_from_full_clip'].copy()
    component_count, minimum_duration = 2, 2 / 120
    starts, selections = make_initializations(times, positions, component_count, minimum_duration)
    fixed_fits, timing_fits, rows, details = [], [], [], []
    for index, start in enumerate(starts):
        if not start['valid']:
            raise ValueError(f"Invalid initialization {start['name']}: {start['invalid_reason']}")
        fixed = fit_fixed_endpoints(times, positions, positions[0], positions[-1],
                                    start['activation'], start['completion'])
        fitted = fit_timing(times, positions, positions[0], positions[-1],
                            start['activation'], start['completion'], minimum_duration)
        fixed_metrics, _ = trajectory_errors(positions, fixed['positions'], times)
        fitted_metrics, _ = trajectory_errors(positions, fitted['positions'], times)
        assert fitted['initial_error_m'] < 1e-12 and fitted['final_error_m'] < 1e-12
        assert fitted_metrics['position_rmse_m'] <= fixed_metrics['position_rmse_m'] + 1e-10
        assert np.all(fitted['activation'] >= times[0])
        assert np.all(fitted['completion'] <= times[-1])
        assert np.all(fitted['completion']-fitted['activation'] >= minimum_duration-1e-12)
        row = {'index': index, 'initialization': start['name'], 'success': fitted['success'],
               **{f'fixed_{key}': value for key, value in fixed_metrics.items()},
               **{f'fitted_{key}': value for key, value in fitted_metrics.items()},
               'initial_error_m': fitted['initial_error_m'], 'final_error_m': fitted['final_error_m'],
               'nfev': fitted['nfev'], 'residual_evaluations': fitted['residual_evaluations'],
               'optimality': fitted['optimality'],
               'timing_jacobian_rank': fitted['timing_jacobian_rank']}
        rows.append(row)
        details.append({'initialization': start, 'result': {key: value for key, value in fitted.items()
                                                         if key != 'positions'}})
        fixed_fits.append(fixed)
        timing_fits.append(fitted)
        print(start['name'], 'success:', fitted['success'], 'position RMSE (cm):',
              100*fixed_metrics['position_rmse_m'], '->', 100*fitted_metrics['position_rmse_m'], flush=True)
    best_fixed = min(range(len(starts)), key=lambda i: (fixed_fits[i]['position_rmse_m'], i))
    best_timing = choose_best_result(timing_fits)
    chosen = timing_fits[best_timing]
    baseline = fixed_fits[best_fixed]
    velocities = component_velocities(times, chosen['activation'], chosen['completion'], chosen['displacements'])
    analytical_velocity = velocities.sum(axis=1)
    summary = {
        'scope': 'Unpenalized real-window timing fit at prescribed K=2; not count selection',
        'input_sha256': source_hash, 'window_s': [times[0], times[-1]], 'samples': len(times),
        'component_count': component_count, 'minimum_duration_s': minimum_duration,
        'settings_reason': 'K=2 is a small overlapping-component debugging case; minimum duration is two sample intervals, not a physiological bound',
        'initialization_settings': {'seed': 23, 'jitters_per_seed': 4, 'jitter_fraction': 0.2, 'overlap_fraction': 0.25},
        'speed_seed_selections': selections,
        'objective': 'Position squared error only; endpoints fixed to observed crop endpoints; no cancellation penalty',
        'algorithm': 'Nonlinear bounded timing least squares; endpoint-constrained linear displacement least squares at every evaluation',
        'comparison_rule': 'Ten paired starts; best fixed-time fit among all ten versus best successful timing fit by position RMSE',
        'best_fixed_index': best_fixed, 'best_timing_index': best_timing,
        'best_fixed_metrics': trajectory_errors(positions, baseline['positions'], times)[0],
        'best_timing_metrics': trajectory_errors(positions, chosen['positions'], times)[0],
        'primary_velocity_metric': 'Same numpy.gradient(edge_order=2) on cropped original and reconstructed positions; endpoint estimates differ from full-clip derivative used in prior inspection',
        'analytical_velocity_vs_full_clip_reference_rmse_m_s': float(np.sqrt(np.mean(np.sum((analytical_velocity-reference_velocity)**2, axis=1)))),
        'analytical_model_endpoint_speeds_m_s': np.linalg.norm(analytical_velocity[[0,-1]], axis=1),
        'full_clip_reference_endpoint_speeds_m_s': np.linalg.norm(reference_velocity[[0,-1]], axis=1),
        'composition_diagnostic_only': composition_metrics(times, velocities),
        'runs': rows, 'details': details,
        'limitations': ['No true component labels or times are available for this motion',
                        'Optimizer termination does not establish global optimality or physiological units',
                        'Velocity is evaluated but not optimized',
                        'Crop endpoint velocities need not be zero, unlike the analytical model',
                        'Fixed component count, short selected window, and numerical settings need further evaluation'],
    }
    (output / 'summary.json').write_text(json.dumps(serializable(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = {'timestamps': times, 'original': positions, 'reference_velocity_full_clip': reference_velocity,
             'selected_fixed_index': best_fixed, 'selected_timing_index': best_timing}
    for i, (fixed, fitted) in enumerate(zip(fixed_fits, timing_fits)):
        for label, result in [('fixed', fixed), ('fitted', fitted)]:
            for key in ('positions', 'displacements'):
                saved[f'{label}_{i}_{key}'] = result[key]
        saved[f'initial_{i}_activation'] = starts[i]['activation']
        saved[f'initial_{i}_completion'] = starts[i]['completion']
        saved[f'fitted_{i}_activation'] = fitted['activation']
        saved[f'fitted_{i}_completion'] = fitted['completion']
    np.savez_compressed(output / 'fits.npz', **saved)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    curves = [('Original', positions, 'black'), ('Best fixed times', baseline['positions'], 'tab:blue'),
              ('Best fitted times', chosen['positions'], 'tab:orange')]
    for label, curve, color in curves:
        velocity = np.gradient(curve, times, axis=0, edge_order=2)
        for c in range(3):
            axes[c, 0].plot(times, curve[:, c], color=color, label=label, linewidth=1.2)
            axes[c, 1].plot(times, velocity[:, c], color=color, label=label, linewidth=1.0)
    for c in range(3):
        axes[c, 0].set_ylabel(f'{"XYZ"[c]} position (m)')
        axes[c, 1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Real wrist, K=2, fixed endpoint positions; no cancellation penalty\nVelocity: identical finite differences on cropped trajectories')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / 'position_velocity_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for c, axis in enumerate(axes):
        for component in range(component_count):
            axis.plot(times, velocities[:, component, c], label=f'Component {component+1}')
        axis.plot(times, analytical_velocity[:, c], color='black', label='Sum')
        axis.plot(times, reference_velocity[:, c], color='gray', alpha=0.6, label='Original (full-clip derivative)')
        axis.set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title('Selected timing fit: analytical component velocities; diagnostics only')
    axes[-1].set_xlabel('Original clip time (s)')
    fig.tight_layout()
    fig.savefig(output / 'component_velocities.png', dpi=160)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    print('Best fixed:', best_fixed, summary['best_fixed_metrics'])
    print('Best timing:', best_timing, summary['best_timing_metrics'])
    print('Selected times:', chosen['activation'], chosen['completion'])


if __name__ == '__main__':
    main()
