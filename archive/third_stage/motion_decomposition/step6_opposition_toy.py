import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .opposition import fit_opposition_fixed_times
from .step1_known_motion import generate_motion
from .step5_fit_real_window import serializable

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'results/step6_opposition_toy'
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(409)/120
    zero = np.zeros(3)
    weights = [0, 1e-6, 1e-4, 1e-2, 1]
    clean = generate_motion(times, zero, [[0.3,0,0]], [0.2], [2.2])
    perturbed = clean.copy()
    perturbed[:, 0] += 0.001*np.sin(2*np.pi*times/times[-1])
    correction = generate_motion(times, zero, [[0.4,0,0], [-0.1,0,0]], [0.2,0.4], [2.2,2.4])
    cases = [('near_identical_times', perturbed, [0.2,0.2001], [2.2,2.2001]),
             ('genuine_opposing_components', correction, [0.2,0.4], [2.2,2.4]),
             ('sequential_return', generate_motion(times, zero, [[0.4,0,0],[-0.1,0,0]], [0.2,1.8], [1.6,3.2]), [0.2,1.8], [1.6,3.2])]
    rows, details, saved = [], [], {'timestamps': times}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for case_index, (name, observed, activation, completion) in enumerate(cases):
        axes[case_index, 0].plot(times, observed[:, 0], color='black', label='Observed')
        for weight in weights:
            result = fit_opposition_fixed_times(times, observed, observed[0], observed[-1],
                                                 activation, completion, weight, 0.3)
            row = {'case': name, 'weight': weight, **{k:v for k,v in result.items()
                    if k not in ('positions', 'displacements')}}
            rows.append(row)
            details.append({**row, 'displacements_m': result['displacements'],
                            'activation_s': activation, 'completion_s': completion})
            assert result['initial_error_m'] < 1e-12 and result['final_error_m'] < 1e-12
            saved[f'{name}_{weight}_positions'] = result['positions']
            axes[case_index, 0].plot(times, result['positions'][:, 0], '--', label=f'lambda={weight:g}')
            print(name, weight, 'RMSE cm', 100*result['position_rmse_m'],
                  'sum |d|', result['sum_displacement_norms_m'], 'success', result['success'], flush=True)
        group = rows[-len(weights):]
        axes[case_index, 1].plot(range(len(weights)), [r['sum_displacement_norms_m'] for r in group], 'o-')
        axes[case_index, 1].set_xticks(range(len(weights)), [f'{w:g}' for w in weights])
        axes[case_index, 1].set_yscale('log')
        axes[case_index, 0].set_title(name.replace('_', ' '))
        axes[case_index, 0].set_ylabel('X position (m)')
        axes[case_index, 1].set_ylabel('Sum displacement norms (m, log)')
        saved[f'{name}_observed'] = observed
    axes[0, 0].legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Penalty weight lambda')
    fig.suptitle('Fixed-time toy checks: suppression, bias, and an allowed sequential return')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / 'toy_comparison.png', dpi=160)
    plt.close(fig)
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'scope': 'Fixed timing synthetic penalty verification only; no real-data penalized timing fit',
               'formula': 'mean ||p_hat-p||^2 + lambda/N * sum_t sum_i<j [(T^2/L)*max(0,-v_i dot v_j)]^2',
               'length_scale_m': 0.3, 'length_scale_reason': 'Prescribed toy scale, not calibrated for real motion',
               'weights': weights, 'timing_optimized': False,
               'source_position': 'Original pairwise dot-product penalty experiment; not a reproduction of Liao and Kirsch equations',
               'cases': details, 'limitations': ['Penalizes genuine simultaneous opposing corrections too',
                    'Nonlinear local displacement solve; no global guarantee',
                    'Weight and length scale not selected for real data',
                    'Discrete sample penalty may miss behavior between samples']}
    (output / 'summary.json').write_text(json.dumps(serializable(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output / 'fits.npz', **saved)


if __name__ == '__main__':
    main()
