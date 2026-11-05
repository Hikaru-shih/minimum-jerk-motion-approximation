import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .stability import compare_aligned
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]


def analyze_saved(source, output, prefix_template='k4_w1e-06_run{index}', window_label='18-21 s'):
    output.mkdir(parents=True,exist_ok=True)
    hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('summary.json','fits.npz')}
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    selection = next(s for s in summary['selections'] if s.get('component_count',4) == 4 and s['weight'] == 1e-6)
    reference_index = selection['best_index']
    entries = sorted([r for r in summary['details'] if r.get('component_count',4) == 4 and r['weight'] == 1e-6],key=lambda r:r['index'])
    assert [r['index'] for r in entries] == list(range(10))
    fits = []
    with np.load(source/'fits.npz') as saved:
        times = saved['timestamps'].copy()
        for row in entries:
            prefix = prefix_template.format(index=row['index'])
            fits.append({key:saved[f'{prefix}_{key}'].copy() for key in ('activation','completion','displacements','positions')})
    rows, components = [], []
    for index,fit in enumerate(fits):
        compared = compare_aligned(times,fits[reference_index],fit)
        order = compared.pop('alignment')
        rows.append({'index':index,'initialization':entries[index]['initialization'],
                     'eligible':entries[index]['eligible'],'position_rmse_m':entries[index]['position_rmse_m'],
                     'objective_m2':entries[index]['objective_m2'],
                     'objective_gap_to_selected_m2':entries[index]['objective_m2']-selection['selected']['objective_m2'],
                     **compared})
        for k,original_index in enumerate(order):
            d = fit['displacements'][original_index]
            components.append({'run_index':index,'aligned_component':k+1,'original_component':int(original_index)+1,
                               'activation_s':fit['activation'][original_index],'completion_s':fit['completion'][original_index],
                               'dx_m':d[0],'dy_m':d[1],'dz_m':d[2]})
    pairs = []
    trajectory_matrix = np.zeros((len(fits),len(fits)))
    signature_matrix = np.zeros_like(trajectory_matrix)
    for i,j in combinations(range(len(fits)),2):
        compared = compare_aligned(times,fits[i],fits[j])
        compared.pop('alignment')
        pairs.append({'run_i':i,'run_j':j,**compared})
        trajectory_matrix[i,j] = trajectory_matrix[j,i] = compared['trajectory_difference_rmse_m']
        signature_matrix[i,j] = signature_matrix[j,i] = compared['velocity_signature_rmse_m_s']
    write_csv(output/'runs_vs_selected.csv',serializable(rows))
    write_csv(output/'aligned_components.csv',serializable(components))
    write_csv(output/'all_pairs.csv',serializable(pairs))
    fig,axes = plt.subplots(4,2,figsize=(12,11),sharex=True)
    for k in range(4):
        group = [r for r in components if r['aligned_component'] == k+1]
        x = [r['run_index'] for r in group]
        axes[k,0].plot(x,[r['activation_s'] for r in group],'o-',label='Activation')
        axes[k,0].plot(x,[r['completion_s'] for r in group],'o-',label='Completion')
        for c,color in zip('xyz',('tab:blue','tab:orange','tab:green')):
            axes[k,1].plot(x,[100*r[f'd{c}_m'] for r in group],'o-',color=color,label=c.upper())
        axes[k,0].set_ylabel(f'Component {k+1}: time (s)')
        axes[k,1].set_ylabel(f'Component {k+1}: displacement (cm)')
    axes[0,0].legend(fontsize=8)
    axes[0,1].legend(fontsize=8,ncol=3)
    for axis in axes.flat:
        axis.set_xticks(range(10))
        axis.axvline(reference_index,color='gray',linestyle=':',alpha=.5)
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel('Initialization index (0-4 uniform, 5-9 speed seed)')
    fig.suptitle(f'{window_label}, K=4, lambda=1e-6: components aligned to selected run by velocity signature')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'aligned_parameters.png',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5))
    for axis,matrix,title in zip(axes,(100*trajectory_matrix,signature_matrix),
            ('Trajectory difference RMSE (cm)','Matched component velocity difference (m/s)')):
        plotted = axis.imshow(matrix,cmap='viridis',origin='lower',vmin=0)
        axis.set_title(title,fontsize=11,pad=12)
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        axis.set_xlabel('Run index')
        axis.set_ylabel('Run index')
        fig.colorbar(plotted,ax=axis,shrink=.8)
    fig.subplots_adjust(left=.06,right=.96,bottom=.13,top=.85,wspace=.35)
    fig.savefig(output/'pairwise_differences.png',dpi=160)
    plt.close(fig)
    result = {'scope':f'{window_label}: saved K=4 lambda=1e-6 ten-initialization stability only; no new fitting',
              'source_hashes':hashes,'reference_run':reference_index,
              'matching_rule':'Enumerate 24 permutations, minimize mean over samples of sum over components of squared analytical XYZ velocity differences; ties use lexicographic order',
              'matching_limitations':'Numerical correspondence, not biological identity; weak components may have poorly determined timings',
              'pairwise_metric_note':'Every pair is matched independently; parameter plot uses selected run as reference',
              'runs':rows,'all_pairs':pairs,'aligned_components':components,
              'ranges':{key:[min(r[key] for r in rows),max(r[key] for r in rows)] for key in
                        ('position_rmse_m','maximum_activation_difference_s','maximum_completion_difference_s',
                         'maximum_displacement_vector_difference_m','trajectory_difference_rmse_m','velocity_signature_rmse_m_s')},
              'pairwise_maxima':{key:max(r[key] for r in pairs) for key in
                                ('maximum_activation_difference_s','maximum_completion_difference_s',
                                 'maximum_displacement_vector_difference_m','trajectory_difference_rmse_m',
                                 'maximum_trajectory_difference_m','velocity_signature_rmse_m_s')},
              'limitations':['Ten starts are deterministic optimization probes, not independent participants or clips',
                             'No significance test, automatic clustering threshold or proof of uniqueness',
                             'Repeated matching cannot resolve genuine splitting/merging ambiguity'],
              'new_optimization_performed':False}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    print('Selected run:',reference_index)
    for row in rows:
        print(row)
    return result


def main():
    analyze_saved(ROOT/'results/step6_joint_real',ROOT/'results/step7_initialization_stability')


if __name__ == '__main__':
    main()
