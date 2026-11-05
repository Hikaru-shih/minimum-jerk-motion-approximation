import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from .step1_known_motion import generate_motion
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'results/step4_known_timing'
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(409) / 120.0
    initial = np.array([0.2, -0.1, 0.4])
    true_displacements = np.array([[0.3, 0.1, -0.05], [-0.12, 0.2, 0.08]])
    true_activation = np.array([0.2, 1.2])
    true_completion = np.array([2.2, 3.2])
    final = initial + true_displacements.sum(axis=0)
    original = generate_motion(times, initial, true_displacements, true_activation, true_completion)
    starts = [([0, 1.6], [1.6, 3.4]), ([0.6, 0.9], [2.8, 2.9]),
              ([0.8, 0], [3.4, 1.8]), ([0, 0.4], [3, 3.4]),
              ([0.8, 1.8], [2.8, 3.4])]
    minimum_duration = 2 / 120
    records, details = [], []
    saved = {'timestamps': times, 'original': original,
             'true_activation': true_activation, 'true_completion': true_completion,
             'true_displacements': true_displacements}
    for number, (activation, completion) in enumerate(starts, 1):
        result = fit_timing(times, original, initial, final, activation, completion, minimum_duration)
        time_error = float(max(np.max(np.abs(result['activation'] - true_activation)),
                               np.max(np.abs(result['completion'] - true_completion))))
        displacement_error = float(np.max(np.abs(result['displacements'] - true_displacements)))
        recovered = time_error < 1e-5 and displacement_error < 1e-7
        row = {
            'initialization': number, 'optimizer_success': result['success'],
            'initial_position_rmse_m': result['initial_position_rmse_m'],
            'fitted_position_rmse_m': result['position_rmse_m'],
            'maximum_timing_error_s': time_error,
            'maximum_displacement_component_error_m': displacement_error,
            'recovered_known_parameters': recovered,
            'initial_error_m': result['initial_error_m'], 'final_error_m': result['final_error_m'],
            'timing_jacobian_rank': result['timing_jacobian_rank'],
            'optimizer_optimality': result['optimality'],
            'nfev': result['nfev'], 'residual_evaluations': result['residual_evaluations'],
        }
        records.append(row)
        details.append({**row, 'initial_activation': activation, 'initial_completion': completion,
                        'activation': result['activation'].tolist(), 'completion': result['completion'].tolist(),
                        'displacements': result['displacements'].tolist(),
                        'optimizer_status': result['status'], 'optimizer_message': result['message'],
                        'timing_jacobian_singular_values': result['timing_jacobian_singular_values'].tolist()})
        assert result['initial_error_m'] < 1e-12 and result['final_error_m'] < 1e-12
        assert result['position_rmse_m'] <= result['initial_position_rmse_m'] + 1e-12
        saved[f'positions_{number}'] = result['positions']
        saved[f'activation_{number}'] = result['activation']
        saved[f'completion_{number}'] = result['completion']
        print(number, 'success:', result['success'], 'RMSE:', result['position_rmse_m'],
              'time error (s):', time_error, 'recovered:', recovered)

    ambiguous = []
    constant = np.tile(initial, (len(times), 1))
    for activation, completion in starts[:2]:
        result = fit_timing(times, constant, initial, initial, activation, completion, minimum_duration)
        assert result['position_rmse_m'] < 1e-12 and result['timing_jacobian_rank'] == 0
        ambiguous.append({'activation': result['activation'].tolist(), 'completion': result['completion'].tolist(),
                          'position_rmse_m': result['position_rmse_m'], 'success': result['success'],
                          'timing_jacobian_rank': result['timing_jacobian_rank']})
    summary = {
        'scope': 'Synthetic two-component timing verification only, no cancellation penalty',
        'known_activation': true_activation.tolist(), 'known_completion': true_completion.tolist(),
        'known_displacements': true_displacements.tolist(),
        'objective': 'Full-clip position squared error with fixed initial and final positions',
        'algorithm': 'Bounded nonlinear least squares in time; endpoint-constrained linear least squares in displacement at every evaluation',
        'minimum_duration_s': minimum_duration,
        'minimum_duration_reason': 'Two sample intervals, a numerical safeguard for this example, not a physiological threshold',
        'timing_domain': 'Each activation and completion lies within observed time; duration >= minimum_duration',
        'label_handling': 'No time-order constraint; components are sorted by activation for comparison with known labels',
        'known_answer_used_by_solver': False,
        'noise': None, 'cancellation_penalty': None,
        'timing_recovery_tolerance_s': 1e-5, 'displacement_recovery_tolerance_m': 1e-7,
        'tolerance_note': 'Synthetic recovery checks, not motion-quality thresholds',
        'recovered_initializations': sum(r['recovered_known_parameters'] for r in records),
        'initializations': details,
        'constant_motion_counterexample': ambiguous,
        'limitations': ['Local optimization is not a proof of global uniqueness',
                        'No real-motion segmentation, count selection, velocity objective or biomechanical validation'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / 'fits.npz', **saved)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for coordinate, name in enumerate('XYZ'):
        axes[coordinate].plot(times, original[:, coordinate], color='black', linewidth=2, label='Known trajectory')
        for number in range(1, len(starts) + 1):
            axes[coordinate].plot(times, saved[f'positions_{number}'][:, coordinate], '--',
                                  linewidth=1, label=f'Timing fit {number}')
        axes[coordinate].set_ylabel(f'{name} (m)')
        axes[coordinate].grid(alpha=0.2)
    axes[0].set_title('Synthetic timing fits from five initializations; endpoints fixed')
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output / 'timing_recovery.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5))
    all_intervals = [(true_activation, true_completion)] + [
        (saved[f'activation_{number}'], saved[f'completion_{number}'])
        for number in range(1, len(starts) + 1)
    ]
    for row, (activation, completion) in enumerate(all_intervals):
        for component, color in enumerate(('tab:blue', 'tab:orange')):
            offset = 0.08 if component == 0 else -0.24
            axis.broken_barh([(activation[component], completion[component] - activation[component])],
                            (row + offset, 0.16), facecolors=color)
    axis.set_yticks(range(6), ['Known times'] + [
        f"Initialization {r['initialization']}" + ('' if r['recovered_known_parameters'] else ' (not recovered)')
        for r in records
    ])
    axis.invert_yaxis()
    axis.set_xlim(times[0], times[-1])
    axis.set_xlabel('Activation to completion (s)')
    axis.set_title('Recovered timing intervals; blue/orange = components 1/2')
    axis.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / 'timing_intervals.png', dpi=180)
    plt.close(fig)
    print('Recovered:', summary['recovered_initializations'], '/', len(starts))
    print('Constant-motion example: multiple timings fit equally well; timing Jacobian rank is zero.')
    print('Saved:', output.relative_to(ROOT))


if __name__ == '__main__':
    main()
