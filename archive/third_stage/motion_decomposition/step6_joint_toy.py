import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .initialization import make_initializations, valid_timing_result
from .joint_fit import fit_joint
from .step1_known_motion import generate_motion
from .step5_fit_real_window import serializable
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'results/step6_joint_toy'
    output.mkdir(parents=True, exist_ok=True)
    times = np.arange(409)/120
    cases = [
        ('nonopposing_known', np.array([[.3,.1,0],[.1,.2,.05]]), [.2,1.2], [2.2,3.2]),
        ('opposing_known', np.array([[.4,0,0],[-.1,0,0]]), [.2,.4], [2.2,2.4]),
    ]
    weights = [0, 1e-6, 1e-4]
    rows, details, selections, saved = [], [], [], {'timestamps': times}
    selected_fits = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for case_index, (name, true_d, true_a, true_b) in enumerate(cases):
        original = generate_motion(times, np.zeros(3), true_d, true_a, true_b)
        starts, _ = make_initializations(times, original, 2, 2/120)
        results_by_weight = {weight: [] for weight in weights}
        saved[f'{name}_original'] = original
        for index, start in enumerate(starts):
            projected = fit_timing(times, original, original[0], original[-1],
                                    start['activation'], start['completion'], 2/120)
            for weight in weights:
                fitted = fit_joint(times, original, original[0], original[-1], start['activation'],
                                    start['completion'], 2/120, weight, .3)
                results_by_weight[weight].append(fitted)
                time_error = max(np.max(np.abs(fitted['activation']-true_a)),
                                 np.max(np.abs(fitted['completion']-true_b)))
                d_error = float(np.max(np.abs(fitted['displacements']-true_d)))
                row = {'case': name, 'index': index, 'initialization': start['name'], 'weight': weight,
                       'eligible': valid_timing_result(fitted),
                       **{k:v for k,v in fitted.items() if k not in ('positions','activation','completion','displacements')},
                       'maximum_timing_error_s': float(time_error),
                       'maximum_displacement_coordinate_error_m': d_error,
                       'recovered_known_parameters': bool(time_error < 1e-5 and d_error < 1e-7),
                       'projected_baseline_rmse_m': projected['position_rmse_m']}
                rows.append(row)
                details.append({**row, 'initialization_details': start,
                                'activation': fitted['activation'], 'completion': fitted['completion'],
                                'displacements': fitted['displacements']})
                assert fitted['initial_error_m'] < 1e-12 and fitted['final_error_m'] < 1e-12
                assert fitted['objective_m2'] <= fitted['initial_objective_m2']+1e-10
                saved[f'{name}_w{weight}_run{index}'] = fitted['positions']
                print(name, index, weight, 'success', fitted['success'],
                      'RMSE cm', 100*fitted['position_rmse_m'], flush=True)
        axes[case_index, 0].plot(times, original[:,0], color='black', label='Known X')
        for weight, fits in results_by_weight.items():
            eligible = [i for i, fit in enumerate(fits) if valid_timing_result(fit)]
            if not eligible:
                raise ValueError(f'No eligible result for {name}, weight {weight}')
            best = min(eligible, key=lambda i: (fits[i]['objective_m2'], i))
            selected_fits[(name, weight)] = fits[best]
            group = [r for r in rows if r['case'] == name and r['weight'] == weight]
            selections.append({'case': name, 'weight': weight, 'best_index': best,
                               'eligible_count': len(eligible), 'selected': group[best],
                               'eligible_rmse_range_m': [min(fits[i]['position_rmse_m'] for i in eligible),
                                                         max(fits[i]['position_rmse_m'] for i in eligible)]})
            axes[case_index, 0].plot(times, fits[best]['positions'][:,0], '--', label=f'lambda={weight:g}')
            values = [r['position_rmse_m']*100 for r in group]
            x = weights.index(weight)
            axes[case_index, 1].scatter([x]*len(values), values, alpha=.5)
            axes[case_index, 1].scatter(x, fits[best]['position_rmse_m']*100, marker='x', color='black', s=60)
        axes[case_index, 0].set_title(name.replace('_',' '))
        axes[case_index, 0].set_ylabel('X position (m)')
        axes[case_index, 0].legend(fontsize=8)
        axes[case_index, 1].set_ylabel('Position RMSE (cm)')
        axes[case_index, 1].set_xticks(range(len(weights)), [str(w) for w in weights])
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[-1,0].set_xlabel('Time (s)')
    axes[-1,1].set_xlabel('Penalty weight; black x = objective-selected run')
    fig.suptitle('Joint time/displacement synthetic fits; ten paired starts per weight')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output / 'joint_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (name, true_d, true_a, true_b) in enumerate(cases):
        intervals = [('Known', np.asarray(true_a), np.asarray(true_b))]+[
            (f'lambda={w:g}', selected_fits[(name,w)]['activation'], selected_fits[(name,w)]['completion'])
            for w in weights]
        for setting, (_, a, b) in enumerate(intervals):
            for component, color in enumerate(('tab:blue', 'tab:orange')):
                axes[row,0].barh(3*setting+component, b[component]-a[component],
                                 left=a[component], height=.7, color=color)
        axes[row,0].set_yticks([3*i+.5 for i in range(len(intervals))], [item[0] for item in intervals])
        axes[row,0].invert_yaxis()
        axes[row,0].set_xlim(times[0], times[-1])
        axes[row,0].set_title(name.replace('_',' ')+': component 1/2 intervals')
        for setting, (label, d) in enumerate([('Known',true_d)]+[
                (f'lambda={w:g}',selected_fits[(name,w)]['displacements']) for w in weights]):
            axes[row,1].bar(np.arange(2)+(setting-1.5)*.18, np.linalg.norm(d,axis=1), width=.18, label=label)
        axes[row,1].set_xticks([0,1], ['Component 1','Component 2'])
        axes[row,1].set_ylabel('Displacement norm (m)')
        for axis in axes[row]:
            axis.grid(alpha=.2)
    axes[0,1].legend(fontsize=8)
    axes[-1,0].set_xlabel('Activation to completion (s)')
    fig.tight_layout()
    fig.savefig(output / 'parameter_comparison.png', dpi=160)
    plt.close(fig)
    summary = {'scope': 'Synthetic joint-solver verification only; no real-data penalized fits',
               'weights': weights, 'length_scale_m': .3, 'minimum_duration_s': 2/120,
               'max_nfev': 1000, 'initialization': 'Ten generated starts per case, identical time and linear displacement initializations across weights',
               'selection': 'Lowest total objective among successful finite endpoint-valid runs, within each case and weight; never compare total objective across weights as the same target',
               'control': 'Joint lambda=0 controls for solver change; original variable-projection timing baseline recorded separately',
               'truth_usage': 'Only generate observations and evaluate recovery after fitting; never seed or select using known parameters',
               'known_parameters': [{'case': name, 'activation': a, 'completion': b, 'displacements': d}
                                    for name,d,a,b in cases],
               'selections': selections, 'details': details}
    (output / 'summary.json').write_text(json.dumps(serializable(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (output / 'runs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'fits.npz', **saved)
    for selection in selections:
        print('SELECTED', selection, flush=True)


if __name__ == '__main__':
    main()
