import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .goal_count_selection import chord_knee
from .step4_diagnose_initialization import component_velocities
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'results/step9_count_comparison'
    output = ROOT/'results/step12_count_evidence'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in ('summary.json', 'fits.npz')}
    previous = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    sensitivity, metrics, components = [], [], []
    counts = np.arange(1, 7)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    with np.load(source/'fits.npz') as arrays:
        for w, window in enumerate(('18_21', '30_33')):
            rows = [r for r in previous['details'] if r['window'] == window and r['weight'] == 1e-6]
            selected = [next(s for s in previous['selections'] if s['window'] == window
                             and s['weight'] == 1e-6 and s['component_count'] == k) for k in counts]
            for k, selection in zip(counts, selected):
                eligible = [r for r in rows if r['component_count'] == k and r['eligible']]
                metrics.append({'window': window, 'K': int(k),
                                'selected_position_cm': 100*selection['selected']['position_rmse_m'],
                                'selected_velocity_m_s': selection['selected']['velocity_rmse_m_s'],
                                'eligible_position_min_cm': 100*min(r['position_rmse_m'] for r in eligible),
                                'eligible_position_max_cm': 100*max(r['position_rmse_m'] for r in eligible)})
            recipes = [('all_10', lambda r: True),
                       ('uniform_family', lambda r: r['initialization'].startswith('uniform')),
                       ('speed_family', lambda r: not r['initialization'].startswith('uniform'))]
            recipes += [(f'omit_start_{i}', lambda r, i=i: r['index'] != i) for i in range(10)]
            for recipe, keep in recipes:
                best = [min((r for r in rows if r['component_count'] == k and r['eligible'] and keep(r)),
                            key=lambda r: (r['objective_m2'], r['index'])) for k in counts]
                for upper in (3, 4, 5, 6):
                    for metric in ('objective_m2', 'position_mse_m2'):
                        costs = [r['objective_m2'] if metric == 'objective_m2' else r['position_rmse_m']**2
                                 for r in best[:upper]]
                        knee = chord_knee(counts[:upper], costs)
                        sensitivity.append({'window': window, 'recipe': recipe, 'upper_K': upper,
                                            'metric': metric, 'selected_K': knee['selected_count'],
                                            'reason': knee['reason'], 'costs': costs,
                                            'selected_run_indices': [r['index'] for r in best[:upper]]})
            own = [r for r in metrics if r['window'] == window]
            axes[w, 0].fill_between(counts, [r['eligible_position_min_cm'] for r in own],
                                   [r['eligible_position_max_cm'] for r in own], alpha=.2,
                                   label='Eligible start range (not confidence interval)')
            axes[w, 0].plot(counts, [r['selected_position_cm'] for r in own], 'o-', label='Lowest-objective fit')
            axes[w, 1].plot(counts, [r['selected_velocity_m_s'] for r in own], 'o-')
            axes[w, 0].set_ylabel('Position RMSE (cm)')
            axes[w, 1].set_ylabel('Velocity RMSE (m/s)')
            for axis in axes[w]:
                axis.set_title(window.replace('_', '-')+' s; lambda=1e-6')
                axis.set_xlabel('K'); axis.set_xticks(counts); axis.grid(alpha=.2)
            axes[w, 0].legend(fontsize=8)
            times = arrays[f'w{window}_timestamps']
            original = arrays[f'w{window}_original']
            original_speed = np.linalg.norm(np.gradient(original, times, axis=0, edge_order=2), axis=1)
            fig2, panels = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
            for col, k in enumerate((2, 3, 4)):
                index = selected[k-1]['best_index']
                prefix = f'w{window}_k{k}_lambda1e-06_run{index}'
                a, b, d = [arrays[f'{prefix}_{key}'] for key in ('activation', 'completion', 'displacements')]
                v = component_velocities(times, a, b, d)
                panels[0, col].plot(times, original_speed, color='0.65', linewidth=.8, label='Original finite-difference speed')
                for i in range(k):
                    panels[0, col].plot(times, np.linalg.norm(v[:, i], axis=1), label=f'Component {i+1}')
                    panels[1, col].plot([a[i], b[i]], [i+1, i+1], linewidth=5)
                    components.append({'window': window, 'K': k, 'index': int(index), 'component': i+1,
                                       'activation_s': float(a[i]), 'completion_s': float(b[i]),
                                       'displacement_norm_m': float(np.linalg.norm(d[i])),
                                       'dx_m': float(d[i,0]), 'dy_m': float(d[i,1]), 'dz_m': float(d[i,2])})
                panels[0, col].plot(times, np.linalg.norm(v.sum(axis=1), axis=1), 'k--', label='Combined speed')
                panels[0, col].set_title(f'K={k}; position {metrics[-6+k-1]["selected_position_cm"]:.3f} cm')
                panels[0, col].set_ylabel('Analytical speed (m/s)')
                panels[0, col].legend(fontsize=7)
                panels[1, col].set_ylabel('Component interval'); panels[1, col].set_yticks(range(1,k+1))
                panels[1, col].set_xlabel('Clip time (s)')
                for axis in panels[:, col]: axis.grid(alpha=.2)
            fig2.suptitle('Selected K=2,3,4 fits: overlapping intervals are allowed; speed norms do not show direction')
            fig2.tight_layout(rect=(0,0,1,.95))
            fig2.savefig(output/f'components_{window}.png', dpi=150)
            plt.close(fig2)
    fig.tight_layout(); fig.savefig(output/'curves_and_start_spread.png', dpi=150); plt.close(fig)
    result = {'scope': 'Read-only inspection of existing step9 fits; no new fitting or selector',
              'weight': 1e-6, 'source_hashes': hashes, 'metrics': metrics,
              'sensitivity': sensitivity, 'components': components,
              'limits': ['Omitting saved starts is not independent replication or a confidence interval',
                         'K=1 starts are identical', 'No extrapolation beyond K=6',
                         'Component speed norms do not establish movement-unit meaning',
                         'Later extra-start diagnostics are not substituted into original score curves']}
    (output/'summary.json').write_text(json.dumps(serializable(result), indent=2)+'\n', encoding='utf-8')
    write_csv(output/'metrics.csv', metrics)
    write_csv(output/'components.csv', components)
    write_csv(output/'knee_sensitivity.csv', [{k:v for k,v in r.items() if k not in ('costs','selected_run_indices')}
                                             for r in sensitivity])
    for name, digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    for r in sensitivity:
        if r['recipe'] in ('all_10','uniform_family','speed_family') and r['metric']=='objective_m2':
            print(r['window'], r['recipe'], r['upper_K'], r['selected_K'], flush=True)


if __name__ == '__main__':
    main()
