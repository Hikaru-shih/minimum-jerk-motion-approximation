import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix, fit_fixed_endpoints
from .initialization import choose_best_result, make_initializations, valid_timing_result
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'results/step5_real_window/window.npz'
    output = ROOT / 'results/step5_count_comparison'
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as data:
        times, original = data['timestamps'].copy(), data['positions'].copy()
    rows, summaries, details, selected = [], [], [], {}
    saved = {'timestamps': times, 'original': original}
    for count in (2, 3, 4):
        starts, candidates = make_initializations(times, original, count, 2/120)
        fits, fixed_fits, count_rows = [], [], []
        for index, start in enumerate(starts):
            if not start['valid']:
                raise ValueError(start['invalid_reason'])
            fixed = fit_fixed_endpoints(times, original, original[0], original[-1],
                                        start['activation'], start['completion'])
            fitted = fit_timing(times, original, original[0], original[-1],
                                start['activation'], start['completion'], 2/120, max_nfev=1000)
            fixed_metrics, _ = trajectory_errors(original, fixed['positions'], times)
            metrics, _ = trajectory_errors(original, fitted['positions'], times)
            reconstructed = original[0] + design_matrix(
                times, fitted['activation'], fitted['completion']) @ fitted['displacements']
            assert np.allclose(reconstructed, fitted['positions'], atol=1e-10, rtol=0)
            assert fitted['initial_error_m'] < 1e-12 and fitted['final_error_m'] < 1e-12
            assert np.all(fitted['activation'] >= times[0])
            assert np.all(fitted['completion'] <= times[-1])
            assert np.all(fitted['completion']-fitted['activation'] >= 2/120-1e-12)
            assert metrics['position_rmse_m'] <= fixed_metrics['position_rmse_m']+1e-10
            velocities = component_velocities(times, fitted['activation'], fitted['completion'], fitted['displacements'])
            composition = composition_metrics(times, velocities)
            opposing = np.zeros(len(times), dtype=bool)
            for i in range(count):
                for j in range(i+1, count):
                    opposing |= np.sum(velocities[:, i]*velocities[:, j], axis=1) < 0
            row = {'component_count': count, 'index': index, 'initialization': start['name'],
                   'success': fitted['success'], 'eligible': valid_timing_result(fitted),
                   **metrics, **{f'fixed_{k}': v for k, v in fixed_metrics.items()},
                   'minimum_duration_s': float(np.min(fitted['completion']-fitted['activation'])),
                   'sum_displacement_norms_m': float(np.linalg.norm(fitted['displacements'], axis=1).sum()),
                   'relative_component_length_excess': composition['relative_component_length_excess'],
                   'opposing_pair_sample_count': int(opposing.sum()),
                   'initial_error_m': fitted['initial_error_m'], 'final_error_m': fitted['final_error_m'],
                   'free_displacement_rank': fitted['free_displacement_rank'],
                   'free_displacement_columns': fitted['free_displacement_columns'],
                   'timing_jacobian_rank': fitted['timing_jacobian_rank'],
                   'timing_parameter_count': fitted['timing_parameter_count'],
                   'nfev': fitted['nfev'], 'residual_evaluations': fitted['residual_evaluations'],
                   'optimality': fitted['optimality']}
            rows.append(row)
            count_rows.append(row)
            fits.append(fitted)
            fixed_fits.append(fixed)
            details.append({'component_count': count, 'index': index, 'initialization': start,
                            'result': {k: v for k, v in fitted.items() if k != 'positions'},
                            'composition': composition})
            prefix = f'k{count}_run{index}'
            for key in ('positions', 'activation', 'completion', 'displacements'):
                saved[f'{prefix}_{key}'] = fitted[key]
            saved[f'{prefix}_fixed_positions'] = fixed['positions']
            print('K', count, start['name'], 'eligible', row['eligible'],
                  'RMSE cm', round(100*metrics['position_rmse_m'], 6), flush=True)
        best = choose_best_result(fits)
        best_fixed = min(range(len(starts)), key=lambda i: (fixed_fits[i]['position_rmse_m'], i))
        selected[count] = fits[best]
        selected_linear = fit_fixed_endpoints(times, original, original[0], original[-1],
                                              fits[best]['activation'], fits[best]['completion'])
        eligible_errors = [r['position_rmse_m'] for r in count_rows if r['eligible']]
        summaries.append({'component_count': count, 'best_index': best,
                          'best_fixed_index': best_fixed, 'best_metrics_and_diagnostics': count_rows[best],
                          'eligible_count': len(eligible_errors),
                          'selected_free_basis_condition_number': selected_linear['condition_number'],
                          'selected_free_basis_singular_values': selected_linear['singular_values'],
                          'eligible_position_rmse_range_m': [min(eligible_errors), max(eligible_errors)],
                          'best_fixed_metrics': trajectory_errors(original, fixed_fits[best_fixed]['positions'], times)[0],
                          'speed_seed_candidates': candidates})
        saved[f'k{count}_selected_index'] = best
        if count == 2:
            prior = json.loads((ROOT / 'results/step5_real_window_fit/summary.json').read_text(encoding='utf-8'))
            assert np.isclose(fits[best]['position_rmse_m'], prior['best_timing_metrics']['position_rmse_m'], atol=1e-10, rtol=0)
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'scope': 'Prescribed K=2,3,4 unpenalized fits on the same 18-21 second window',
               'input_sha256': source_hash, 'objective': 'Position squared error, fixed endpoint positions',
               'settings': {'counts': [2, 3, 4], 'minimum_duration_s': 2/120,
                            'seed': 23, 'jitters_per_seed': 4, 'jitter_fraction': 0.2,
                            'overlap_fraction': 0.25, 'max_nfev_per_run': 1000},
               'fairness_note': 'Same start-generation recipe and ten runs per K, same nfev cap; different dimensions mean different residual-evaluation counts and computational costs',
               'selection': 'Lowest position RMSE among successful finite endpoint-valid results; no cancellation filtering',
               'diagnostic_note': 'Length excess also includes nonparallel motion; opposing pairs alone do not establish unnecessary cancellation',
               'counts': summaries, 'details': details,
               'limitations': ['No automatic count choice', 'Training reconstruction errors only, no held-out validation',
                               'Local optimization, no global-minimum guarantee', 'One short selected window',
                               'Velocity not optimized; endpoint velocity mismatch remains']}
    (output / 'summary.json').write_text(json.dumps(serializable(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output / 'fits.npz', **saved)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, key, scale, label in zip(axes,
            ['position_rmse_m', 'velocity_rmse_m_s', 'max_position_error_m'],
            [100, 1, 100], ['Position RMSE (cm)', 'Velocity RMSE (m/s)', 'Maximum position error (cm)']):
        for count in (2, 3, 4):
            group = [r for r in rows if r['component_count'] == count]
            for row in group:
                axis.scatter(count, scale*row[key], color='gray' if row['eligible'] else 'red', alpha=0.5, s=24)
        axis.plot([2, 3, 4], [scale*s['best_metrics_and_diagnostics'][key] for s in summaries],
                  'o-', color='tab:blue', label='Selected by position RMSE')
        axis.set_xticks([2, 3, 4])
        axis.set_xlabel('Prescribed component count')
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same real window, ten initializations per K; gray = eligible runs, red = ineligible')
    fig.tight_layout()
    fig.savefig(output / 'errors_vs_count.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for name, positions, color in [('Original', original, 'black')]+[
            (f'K={k}', selected[k]['positions'], color) for k, color in zip((2,3,4), ('tab:blue','tab:orange','tab:green'))]:
        velocity = np.gradient(positions, times, axis=0, edge_order=2)
        for c in range(3):
            axes[c, 0].plot(times, positions[:, c], color=color, label=name, linewidth=1.1)
            axes[c, 1].plot(times, velocity[:, c], color=color, label=name, linewidth=1.0)
    for c in range(3):
        axes[c, 0].set_ylabel(f'{"XYZ"[c]} position (m)')
        axes[c, 1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Best position-selected fits at K=2,3,4; no cancellation penalty')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / 'trajectory_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for row, count in enumerate((2, 3, 4)):
        result = selected[count]
        for component, (a, b) in enumerate(zip(result['activation'], result['completion'])):
            axes[row, 0].barh(component+1, b-a, left=a, height=0.5)
        axes[row, 0].set_xlim(times[0], times[-1])
        axes[row, 0].set_yticks(range(1, count+1))
        axes[row, 0].set_ylabel(f'K={count}: component')
        norms = np.linalg.norm(result['displacements'], axis=1)
        axes[row, 1].bar(range(1, count+1), norms)
        axes[row, 1].set_yscale('log')
        axes[row, 1].set_xticks(range(1, count+1))
        axes[row, 1].set_ylabel('Displacement norm (m, log scale)')
        axes[row, 0].grid(axis='x', alpha=0.2)
        axes[row, 1].grid(axis='y', alpha=0.2)
    axes[0, 0].set_title('Fitted activation-completion intervals')
    axes[0, 1].set_title('Each component displacement magnitude')
    axes[-1, 0].set_xlabel('Original clip time (s)')
    axes[-1, 1].set_xlabel('Component index (sorted by activation)')
    fig.tight_layout()
    fig.savefig(output / 'component_diagnostics.png', dpi=160)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    for entry in summaries:
        print('SELECTED', entry['component_count'], entry['best_metrics_and_diagnostics'], flush=True)


if __name__ == '__main__':
    main()
