import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .initialization import make_initializations, valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0, 1e-6, 1e-4)
COUNTS = (2, 3, 4)


def plot_saved_diagnostics():
    output = ROOT / 'results/step6_joint_real'
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['details']
    with np.load(output / 'fits.npz') as data:
        times = data['timestamps'].copy()
    selected = {}
    for entry in summary['selections']:
        if entry['best_index'] is not None:
            selected[(entry['component_count'],entry['weight'])] = next(
                r for r in rows if r['component_count'] == entry['component_count']
                and r['weight'] == entry['weight'] and r['index'] == entry['best_index'])
    fig, axes = plt.subplots(3,2,figsize=(12,9))
    for row, weight in enumerate(WEIGHTS):
        group = [r for r in rows if r['component_count'] == 4 and r['weight'] == weight]
        for entry in group:
            color = 'tab:blue' if entry['eligible'] else 'tab:red'
            axes[row,0].scatter(entry['sum_displacement_norms_m'],100*entry['position_rmse_m'],color=color)
        axes[row,0].set_xscale('log')
        axes[row,0].set_ylabel('Position RMSE (cm)')
        axes[row,0].set_title(f'K=4, lambda={weight:g}; blue=eligible, red=ineligible')
        if (4,weight) in selected:
            result = selected[(4,weight)]
            for component, (a,b) in enumerate(zip(result['activation'],result['completion'])):
                axes[row,1].barh(component+1,b-a,left=a,height=.5)
            axes[row,1].set_yticks(range(1,5))
            axes[row,1].set_xlim(times[0],times[-1])
            axes[row,1].set_ylabel('Selected component')
        else:
            axes[row,1].text(.5,.5,'No eligible result',ha='center',transform=axes[row,1].transAxes)
        for axis in axes[row]:
            axis.grid(alpha=.2)
    axes[-1,0].set_xlabel('Sum displacement norms (m, log scale)')
    axes[-1,1].set_xlabel('Activation to completion (s)')
    fig.tight_layout()
    fig.savefig(output / 'k4_diagnostics.png',dpi=160)
    plt.close(fig)


def main():
    source = ROOT / 'results/step5_real_window/window.npz'
    baseline_path = ROOT / 'results/step5_count_comparison/summary.json'
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source, baseline_path)}
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    output = ROOT / 'results/step6_joint_real'
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source) as data:
        times, original = data['timestamps'].copy(), data['positions'].copy()
    rows, details, selections = [], [], []
    selected = {}
    saved = {'timestamps': times, 'original': original}
    for count in COUNTS:
        starts, candidates = make_initializations(times, original, count, 2/120)
        for weight in WEIGHTS:
            group, fits = [], []
            for index, start in enumerate(starts):
                if not start['valid']:
                    raise ValueError(start['invalid_reason'])
                result = fit_joint(times, original, original[0], original[-1],
                                   start['activation'], start['completion'], 2/120,
                                   weight, .3, max_nfev=1000)
                metrics, _ = trajectory_errors(original, result['positions'], times)
                velocity = component_velocities(times, result['activation'], result['completion'], result['displacements'])
                composition = composition_metrics(times, velocity)
                reconstructed = original[0]+design_matrix(times, result['activation'], result['completion']) @ result['displacements']
                assert np.allclose(reconstructed, result['positions'], atol=1e-9, rtol=0)
                assert result['initial_error_m'] < 1e-12 and result['final_error_m'] < 1e-12
                assert result['objective_m2'] <= result['initial_objective_m2']+1e-10
                assert np.all(result['activation'] >= times[0]) and np.all(result['completion'] <= times[-1])
                assert np.all(result['completion']-result['activation'] >= 2/120-1e-12)
                row = {'component_count': count, 'weight': weight, 'index': index,
                       'initialization': start['name'], 'eligible': valid_timing_result(result),
                       **{k:v for k,v in result.items() if k not in ('positions','activation','completion','displacements')},
                       **metrics,
                       'relative_component_length_excess': composition['relative_component_length_excess'],
                       'minimum_duration_s': float(np.min(result['completion']-result['activation']))}
                rows.append(row)
                group.append(row)
                fits.append(result)
                details.append({**row, 'initialization_details': start,
                                'activation': result['activation'], 'completion': result['completion'],
                                'displacements': result['displacements'], 'composition': composition})
                prefix = f'k{count}_w{weight}_run{index}'
                for key in ('positions','activation','completion','displacements'):
                    saved[f'{prefix}_{key}'] = result[key]
                print('K', count, 'lambda', weight, 'run', index, 'eligible', row['eligible'],
                      'RMSE cm', round(100*result['position_rmse_m'],6),
                      'sum |d|', round(result['sum_displacement_norms_m'],6), flush=True)
                with (output / 'runs.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            eligible = [i for i,r in enumerate(group) if r['eligible']]
            best = min(eligible, key=lambda i: (group[i]['objective_m2'],i)) if eligible else None
            selections.append({'component_count': count, 'weight': weight, 'eligible_count': len(eligible),
                               'best_index': best, 'selected': group[best] if best is not None else None,
                               'all_run_position_rmse_range_m': [min(r['position_rmse_m'] for r in group), max(r['position_rmse_m'] for r in group)],
                               'speed_seed_candidates': candidates})
            if best is not None:
                selected[(count,weight)] = fits[best]
    summary = {'scope': 'Same real 18-21 s window, prescribed K=2,3,4, paired joint solver with and without opposition penalty',
               'source_hashes': hashes, 'counts': COUNTS, 'weights': WEIGHTS,
               'settings': {'length_scale_m': .3, 'minimum_duration_s': 2/120, 'max_nfev': 1000,
                            'seed': 23, 'jitters_per_seed': 4, 'jitter_fraction': .2, 'overlap_fraction': .25},
               'scale_weight_reason': 'Prespecified toy settings retained for sensitivity comparison; not calibrated or selected as best for real motion',
               'selection': 'Within each K and weight, lowest total objective among successful finite endpoint-valid results; all runs retained',
               'fairness': 'Same times and initial linear displacements across weights within each K; no warm starts; same nfev cap does not imply equal runtime',
               'velocity_metric': 'Same numpy.gradient(edge_order=2) on cropped original and reconstructed positions',
               'original_variable_projection_baselines': baseline['counts'],
               'selections': selections, 'details': details,
               'limitations': ['No true movement-unit labels for real motion', 'No automatic count or weight selection',
                               'Training-window errors only', 'Penalty can suppress genuine opposing corrections',
                               'Pair count changes with K; same weight is not equal effective regularization',
                               'Local convergence and nonzero endpoint reference velocities remain limitations']}
    (output / 'summary.json').write_text(json.dumps(serializable(summary), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output / 'fits.npz', **saved)
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for axis, key, scale, label in zip(axes,
            ('position_rmse_m','velocity_rmse_m_s','sum_displacement_norms_m'),
            (100,1,1), ('Position RMSE (cm)','Velocity RMSE (m/s)','Sum displacement norms (m, log)')):
        for weight in WEIGHTS:
            points = [s for s in selections if s['weight'] == weight and s['selected'] is not None]
            axis.plot([s['component_count'] for s in points], [scale*s['selected'][key] for s in points],
                      'o-', label=f'Joint lambda={weight:g}')
        axis.plot(COUNTS, [scale*x['best_metrics_and_diagnostics'][key] for x in baseline['counts']],
                  'x--', color='black', label='Original variable projection')
        axis.set_xticks(COUNTS)
        axis.set_xlabel('Prescribed component count')
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[2].set_yscale('log')
    axes[0].legend(fontsize=7)
    fig.suptitle('Same wrist window; each joint result selected by its own objective')
    fig.tight_layout()
    fig.savefig(output / 'error_and_size_comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
    for label, curve, color in [('Original', original, 'black')]+[
            (f'K=4, lambda={w:g}',selected[(4,w)]['positions'],color)
            for w,color in zip(WEIGHTS,('tab:blue','tab:orange','tab:green')) if (4,w) in selected]:
        velocity = np.gradient(curve,times,axis=0,edge_order=2)
        for c in range(3):
            axes[c,0].plot(times,curve[:,c],color=color,label=label,linewidth=1.1)
            axes[c,1].plot(times,velocity[:,c],color=color,label=label,linewidth=1.0)
    for c in range(3):
        axes[c,0].set_ylabel(f'{"XYZ"[c]} position (m)')
        axes[c,1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.tight_layout()
    fig.savefig(output / 'k4_trajectory_comparison.png',dpi=160)
    plt.close(fig)
    plot_saved_diagnostics()
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    for entry in selections:
        print('SELECTED',entry['component_count'],entry['weight'],entry['selected'],flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics-only', action='store_true', help='Plot saved K=4 diagnostics without fitting')
    args = parser.parse_args()
    if args.diagnostics_only:
        plot_saved_diagnostics()
    else:
        main()
