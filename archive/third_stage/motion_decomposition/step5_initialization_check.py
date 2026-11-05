import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .initialization import choose_best_result, make_initializations, valid_timing_result
from .step1_known_motion import generate_motion
from .step3_speed_minima import speed_profile
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def serialize_start(start):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in start.items()}


def main():
    output = ROOT / 'results/step5_initialization_check'
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(409) / 120.0
    initial = np.array([0.2, -0.1, 0.4])
    true_displacements = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
    true_start, true_end = np.array([0.2, 1.2]), np.array([2.2, 3.2])
    final = initial + true_displacements.sum(axis=0)
    positions = generate_motion(times, initial, true_displacements, true_start, true_end)
    starts, selections = make_initializations(times, positions, 2, 2 / 120)
    fits, details, rows = [], [], []
    for number, start in enumerate(starts):
        if not start['valid']:
            fits.append(None)
            details.append({'initialization': serialize_start(start), 'fit': None})
            continue
        result = fit_timing(times, positions, initial, final, start['activation'],
                            start['completion'], 2 / 120)
        fits.append(result)
        row = {'initialization': number, 'name': start['name'],
               'optimizer_success': result['success'], 'eligible_for_selection': valid_timing_result(result),
               'initial_position_rmse_m': result['initial_position_rmse_m'],
               'position_rmse_m': result['position_rmse_m'],
               'initial_error_m': result['initial_error_m'], 'final_error_m': result['final_error_m']}
        rows.append(row)
        details.append({'initialization': serialize_start(start), 'fit': {
            **row, 'activation': result['activation'].tolist(), 'completion': result['completion'].tolist(),
            'displacements': result['displacements'].tolist(),
            'timing_jacobian_rank': result['timing_jacobian_rank'],
            'free_displacement_rank': result['free_displacement_rank'],
            'message': result['message'], 'nfev': result['nfev'],
            'residual_evaluations': result['residual_evaluations'],
        }})
    selected_index = choose_best_result(fits)
    selected = fits[selected_index]
    validation = []
    for i, result in enumerate(fits):
        if result is None:
            continue
        time_error = float(max(np.max(np.abs(result['activation'] - true_start)),
                               np.max(np.abs(result['completion'] - true_end))))
        displacement_error = float(np.max(np.abs(result['displacements'] - true_displacements)))
        validation.append({'initialization': i, 'maximum_timing_error_s': time_error,
                           'maximum_displacement_component_error_m': displacement_error,
                           'recovered': time_error < 1e-5 and displacement_error < 1e-7})
    chosen_validation = next(row for row in validation if row['initialization'] == selected_index)
    assert selected['position_rmse_m'] <= min(r['position_rmse_m'] for r in rows if r['eligible_for_selection'])

    source_path = PROJECT_ROOT / 'results/wrist_trajectory.npz'
    with np.load(source_path) as source:
        real_times, real_positions = source['timestamps'], source['positions'].astype(float)
    real_starts, real_selections = make_initializations(real_times, real_positions, 102, 2 / 120)
    summary = {
        'scope': 'Generated initialization and selection-rule verification; no cancellation penalty',
        'seed': 23, 'jitters_per_seed': 4, 'jitter_fraction': 0.2,
        'initial_overlap_fraction': 0.25,
        'parameter_note': 'Declared initialization heuristics, not physiological estimates or fitted final constraints',
        'speed_rule': 'Lowest raw local-speed minimum in each nonoverlapping slot around a uniform interior knot; ties earlier; uniform fallback if empty',
        'timing_mapping': 'Expand neighboring knot intervals by 0.25 times the smaller adjacent gap; fixed clip endpoints',
        'selection_rule': 'Successful finite fits with endpoint errors <= 1e-12 m; select lowest full-clip position RMSE, exact ties earlier run',
        'selection_uses_known_parameters': False,
        'rank_note': 'Rank diagnostics recorded, not included as a selection penalty; low error does not imply identification',
        'synthetic': {
            'component_count': 2, 'minimum_duration_s': 2 / 120,
            'unique_initialization_count': len({(s['activation'].tobytes(), s['completion'].tobytes()) for s in starts}),
            'speed_seed_selections': selections, 'runs': details,
            'selected_index': selected_index, 'selected_name': starts[selected_index]['name'],
            'selected_position_rmse_m': selected['position_rmse_m'],
            'selected_activation': selected['activation'].tolist(),
            'selected_completion': selected['completion'].tolist(),
            'validation_after_selection': validation,
            'selected_validation': chosen_validation,
        },
        'real_wrist_preview': {
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'prescribed_component_count': 102,
            'speed_seed_selections': real_selections,
            'uniform_fallback_count': sum(r['fallback_uniform'] for r in real_selections),
            'initializations': [serialize_start(start) for start in real_starts],
            'valid_initialization_count': sum(start['valid'] for start in real_starts),
            'joint_timing_optimization_performed': False,
        },
        'limitations': [
            'K is supplied; not an automatically selected movement count',
            'Uniform slots and lowest-speed samples do not validate movement boundaries',
            'Raw-speed differentiation and window phase can affect seeds',
            'Same-data selection favors the position objective; no velocity or physical-meaning guarantee',
            'No comparison against a cancellation-penalized method yet',
        ],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'selected_synthetic.npz', timestamps=times, original=positions,
                        fitted=selected['positions'], activation=selected['activation'],
                        completion=selected['completion'], displacements=selected['displacements'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _, speed = speed_profile(positions, times)
    axes[0].plot(times, speed, color='black')
    for start, color in [(starts[0], 'tab:blue'), (starts[5], 'tab:orange')]:
        for i, time in enumerate(start['knots'][1:-1]):
            axes[0].axvline(time, color=color, linestyle='--', label=start['seed_type'] if i == 0 else None)
    axes[0].set_ylabel('Raw speed (m/s)')
    axes[0].legend()
    axes[0].set_title('Seed knots are initial guesses, not fixed fitted timing boundaries')
    for i in range(2):
        axes[1].broken_barh([(true_start[i], true_end[i] - true_start[i])],
                           (1.1 + i * 0.25, 0.2), facecolors=('tab:blue', 'tab:orange')[i])
        axes[1].broken_barh([(selected['activation'][i], selected['completion'][i] - selected['activation'][i])],
                           (0.25 + i * 0.25, 0.2), facecolors=('tab:blue', 'tab:orange')[i])
    axes[1].set_yticks([0.475, 1.325], ['Selected fit', 'Known times'])
    axes[1].set_title('Blue/orange bars are components 1/2; vertical offsets show overlap')
    axes[1].set_xlabel('Time (s)')
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'initialization_and_selection.png', dpi=180)
    plt.close(fig)
    print('Synthetic speed-seed choices:', selections)
    for row in rows:
        print(row['name'], 'RMSE (m):', row['position_rmse_m'], 'eligible:', row['eligible_for_selection'])
    print('Selected using position RMSE only:', starts[selected_index]['name'])
    print('Known-answer validation, checked afterward:', chosen_validation)
    print('Recovered runs:', sum(row['recovered'] for row in validation), '/', len(validation))
    print('Real 102-component initialization preview only; valid starts:', sum(r['valid'] for r in real_starts),
          '/', len(real_starts), '; uniform fallback knots:', summary['real_wrist_preview']['uniform_fallback_count'])
    print('Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
