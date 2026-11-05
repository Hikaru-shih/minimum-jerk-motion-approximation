import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix, fit_fixed_endpoints
from .initialization import make_initializations, valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]
COUNTS = tuple(range(1,7))
WEIGHTS = (0,1e-6)
WINDOWS = ((18,21),(30,33))
SETTINGS = {'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,
            'seed':23,'jitters_per_seed':4,'jitter_fraction':.2,'overlap_fraction':.25}


def load_existing(source, label):
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    assert all(summary['settings'][key] == value for key,value in SETTINGS.items())
    cache = {}
    with np.load(source/'fits.npz') as saved:
        for row in summary['details']:
            count,weight,index = row.get('component_count',4),row['weight'],row['index']
            if weight not in WEIGHTS:
                continue
            prefix = f'k{count}_w{weight}_run{index}' if label == 'first' else f'w{weight}_run{index}'
            fit = {key:saved[f'{prefix}_{key}'].copy() for key in ('positions','activation','completion','displacements')}
            fit.update({key:value for key,value in row.items() if key not in fit})
            cache[(count,weight,index)] = fit
    return cache


def main():
    output = ROOT/'results/step9_count_comparison'
    output.mkdir(parents=True,exist_ok=True)
    source = ROOT.parents[1]/'results/wrist_trajectory.npz'
    source_paths = [source]+[ROOT/f'results/{name}/{file}' for name in ('step6_joint_real','step8_second_window') for file in ('summary.json','fits.npz')]
    hashes = {str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    protocol = {'windows_s':WINDOWS,'component_counts':COUNTS,'weights':WEIGHTS,'settings':SETTINGS,
                'scope':'Fixed finite count comparison, not automatic count selection',
                'count_reason':'Start with one component, extend previous 2-4 comparison through six; six is an experimental cap, not inferred movement count',
                'positive_weight_reason':'1e-6 is a provisional controlled condition previously examined, not a selected optimal weight',
                'selection':'Within each window,K,weight choose eligible fit with lowest own total objective; no fallback to ineligible fits',
                'fairness':'Same ten start-generation recipe and nfev cap; parameter dimension, pair count and residual evaluation cost vary with K',
                'single_component_note':'All ten generated K=1 initializations are identical; compute once per weight and reuse, not ten independent attempts',
                'input_hashes':hashes}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    caches = [load_existing(ROOT/'results/step6_joint_real','first'),
              load_existing(ROOT/'results/step8_second_window','second')]
    with np.load(source) as data:
        full_times,full_positions = data['timestamps'].copy(),data['positions'].astype(float)
    rows,details,selections,fixed_rows = [],[],[],[]
    saved = {}
    selected_curves = {}
    for window,cache in zip(WINDOWS,caches):
        label = f'{window[0]}_{window[1]}'
        mask = (full_times>=window[0])&(full_times<=window[1])
        times,positions = full_times[mask],full_positions[mask]
        assert len(times)==361
        saved[f'w{label}_timestamps'],saved[f'w{label}_original'] = times,positions
        for count in COUNTS:
            starts,_ = make_initializations(times,positions,count,2/120)
            signatures = [(s['activation'].tobytes(),s['completion'].tobytes()) for s in starts]
            fixed = []
            for index,start in enumerate(starts):
                if not start['valid']:
                    raise ValueError(start['invalid_reason'])
                base = fit_fixed_endpoints(times,positions,positions[0],positions[-1],start['activation'],start['completion'])
                metrics,_ = trajectory_errors(positions,base['positions'],times)
                fixed.append({'window':label,'component_count':count,'index':index,**metrics})
            best_fixed = min(fixed,key=lambda r:(r['position_rmse_m'],r['index']))
            fixed_rows.append(best_fixed)
            for weight in WEIGHTS:
                group,fits = [],[]
                identical = {}
                for index,start in enumerate(starts):
                    key = (count,weight,index)
                    if key in cache:
                        fit,origin = cache[key],'prior_experiment'
                        for field in ('activation','completion'):
                            np.testing.assert_array_equal(start[field],fit['initialization_details'][field])
                    elif signatures[index] in identical:
                        fit,origin = identical[signatures[index]],'identical_initialization'
                    else:
                        fit = fit_joint(times,positions,positions[0],positions[-1],start['activation'],start['completion'],
                                        2/120,weight,.3,max_nfev=1000)
                        origin = 'new_solve'
                    identical[signatures[index]] = fit
                    metrics,_ = trajectory_errors(positions,fit['positions'],times)
                    composition = composition_metrics(times,component_velocities(times,fit['activation'],fit['completion'],fit['displacements']))
                    assert np.allclose(positions[0]+design_matrix(times,fit['activation'],fit['completion'])@fit['displacements'],fit['positions'],atol=1e-9,rtol=0)
                    assert fit['initial_error_m']<1e-12 and fit['final_error_m']<1e-12
                    assert fit['objective_m2']<=fit['initial_objective_m2']+1e-10
                    assert np.all(fit['activation']>=times[0]) and np.all(fit['completion']<=times[-1])
                    assert np.all(fit['completion']-fit['activation']>=2/120-1e-12)
                    row = {'window':label,'component_count':count,'weight':weight,'index':index,
                           'initialization':start['name'],'origin':origin,'eligible':valid_timing_result(fit),
                           **metrics,'objective_m2':fit['objective_m2'],'opposition_cost_m2':fit['opposition_cost_m2'],
                           'sum_displacement_norms_m':fit['sum_displacement_norms_m'],
                           'relative_component_length_excess':composition['relative_component_length_excess'],
                           'initial_error_m':fit['initial_error_m'],'final_error_m':fit['final_error_m'],
                           'status':fit['status'],'message':fit['message'],'nfev':fit['nfev'],
                           'residual_evaluations':fit['residual_evaluations'],'optimality':fit['optimality'],
                           'parameter_count':fit['parameter_count'],'joint_jacobian_rank':fit['joint_jacobian_rank']}
                    rows.append(row)
                    group.append(row)
                    fits.append(fit)
                    details.append({**row,'activation':fit['activation'],'completion':fit['completion'],
                                    'displacements':fit['displacements'],'initialization_details':start})
                    prefix=f'w{label}_k{count}_lambda{weight}_run{index}'
                    for field in ('activation','completion','displacements','positions'):
                        saved[f'{prefix}_{field}']=fit[field]
                    write_csv(output/'runs.csv',serializable(rows))
                    print(label,'K',count,'lambda',weight,'run',index,origin,'eligible',row['eligible'],
                          'RMSE cm',round(100*metrics['position_rmse_m'],6),'sum|d|',round(row['sum_displacement_norms_m'],6),flush=True)
                eligible = [i for i,r in enumerate(group) if r['eligible']]
                best = min(eligible,key=lambda i:(group[i]['objective_m2'],i)) if eligible else None
                entry = {'window':label,'component_count':count,'weight':weight,'best_index':best,
                         'eligible_count':len(eligible),'distinct_initialization_count':len(set(signatures)),
                         'eligible_distinct_initialization_count':len({signatures[i] for i in eligible}),
                         'selected':group[best] if best is not None else None,
                         'all_position_rmse_range_m':[min(r['position_rmse_m'] for r in group),max(r['position_rmse_m'] for r in group)],
                         'eligible_position_rmse_range_m':[min(group[i]['position_rmse_m'] for i in eligible),max(group[i]['position_rmse_m'] for i in eligible)] if eligible else None}
                selections.append(entry)
                if best is not None:
                    selected_curves[(label,count,weight)]=fits[best]['positions']
                (output/'progress.json').write_text(json.dumps(serializable(selections),indent=2)+'\n',encoding='utf-8')
    result={**protocol,'selections':selections,'fixed_time_baselines':fixed_rows,'details':details,
            'accounting':{origin:sum(r['origin']==origin for r in rows) for origin in ('prior_experiment','identical_initialization','new_solve')},
            'baseline_note':'Best position fit among the same initial-time candidates with timing fixed, displacement fitted; not copied-goal old baseline',
            'limitations':['One clip, two selected windows','Finite-budget local optimization; gaps if no eligible fit',
                           'Same lambda across K does not mean same pair-normalized penalty',
                           'No ground-truth component count, no automatic count selection',
                           'Finite-difference velocity evaluation, fixed position endpoint mismatch remains']}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(output/'selected_metrics.csv',serializable([e['selected'] for e in selections if e['selected'] is not None]))
    write_csv(output/'fixed_time_baselines.csv',serializable(fixed_rows))
    np.savez_compressed(output/'fits.npz',**saved)
    plot_results(output,result,selected_curves,saved)
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()==digest
    print('ACCOUNTING',result['accounting'],flush=True)
    for entry in selections:
        print('SELECTED',entry,flush=True)


def plot_results(output,result,selected_curves,saved):
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,window in enumerate(WINDOWS):
        label=f'{window[0]}_{window[1]}'
        for axis,key,scale,title in zip(axes[row],('position_rmse_m','velocity_rmse_m_s','sum_displacement_norms_m'),
                (100,1,1),('Position RMSE (cm)','Velocity RMSE (m/s)','Sum displacement norms (m, log)')):
            for weight,color in zip(WEIGHTS,('tab:blue','tab:orange')):
                group=[s for s in result['selections'] if s['window']==label and s['weight']==weight]
                axis.plot(COUNTS,[scale*s['selected'][key] if s['selected'] else np.nan for s in group],
                          'o-',color=color,label=f'Joint lambda={weight:g}')
                for entry in result['details']:
                    if entry['window']==label and entry['weight']==weight and not entry['eligible']:
                        axis.scatter(entry['component_count'],scale*entry[key],marker='x',color='red',s=15,alpha=.4)
            if key!='sum_displacement_norms_m':
                base=[r for r in result['fixed_time_baselines'] if r['window']==label]
                axis.plot(COUNTS,[scale*r[key] for r in base],'--',color='gray',label='Fixed initial times, fitted displacement')
            axis.set_title(f'{window[0]}-{window[1]} s: {title}',fontsize=10)
            axis.set_xticks(COUNTS)
            axis.set_xlabel('Prescribed component count K')
            axis.grid(alpha=.2)
        axes[row,2].set_yscale('log')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('K=1-6 comparison; red x = ineligible last iterates, not selected results')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/'errors_and_size_vs_count.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,window in enumerate(WINDOWS):
        label=f'{window[0]}_{window[1]}'
        for weight,color in zip(WEIGHTS,('tab:blue','tab:orange')):
            group=[s for s in result['selections'] if s['window']==label and s['weight']==weight]
            axes[row,0].plot(COUNTS,[s['eligible_distinct_initialization_count'] for s in group],'o-',color=color,label=f'lambda={weight:g}')
            for entry in group:
                if entry['eligible_position_rmse_range_m'] is not None:
                    low,high=entry['eligible_position_rmse_range_m']
                    axes[row,1].vlines(entry['component_count']+(-.05 if weight==0 else .05),100*low,100*high,color=color,linewidth=4,alpha=.7)
                    axes[row,1].scatter([entry['component_count']+(-.05 if weight==0 else .05)]*2,
                                        [100*low,100*high],color=color,marker='_',s=55)
        axes[row,0].set_ylabel(f'{window[0]}-{window[1]} s\nEligible distinct starts')
        axes[row,1].set_ylabel('Eligible position RMSE range (cm)')
    axes[0,0].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(COUNTS)
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel('Prescribed K; K=1 has one distinct start')
    fig.tight_layout()
    fig.savefig(output/'convergence_and_spread.png',dpi=160)
    plt.close(fig)
    for window in WINDOWS:
        label=f'{window[0]}_{window[1]}'
        times,original=saved[f'w{label}_timestamps'],saved[f'w{label}_original']
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves=[('Original',original,'black')]+[(f'K={k}, lambda=1e-6',selected_curves[(label,k,1e-6)],color)
                for k,color in ((2,'tab:blue'),(4,'tab:orange'),(6,'tab:green')) if (label,k,1e-6) in selected_curves]
        for name,p,color in curves:
            v=np.gradient(p,times,axis=0,edge_order=2)
            for c in range(3):
                axes[c,0].plot(times,p[:,c],color=color,label=name,linewidth=1)
                axes[c,1].plot(times,v[:,c],color=color,label=name,linewidth=1)
        for c in range(3):
            axes[c,0].set_ylabel(f'{"XYZ"[c]} position (m)')
            axes[c,1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
        for axis in axes.flat:
            axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        for axis in axes[-1]:
            axis.set_xlabel('Original clip time (s)')
        fig.suptitle(f'{window[0]}-{window[1]} s: prespecified K=2,4,6 displayed; all counts saved')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(output/f'trajectories_{label}.png',dpi=160)
        plt.close(fig)


def plot_saved():
    output=ROOT/'results/step9_count_comparison'
    result=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with np.load(output/'fits.npz') as data:
        saved={key:data[key].copy() for key in data.files}
    curves={}
    for entry in result['selections']:
        if entry['best_index'] is not None:
            label,count,weight,index=entry['window'],entry['component_count'],entry['weight'],entry['best_index']
            curves[(label,count,weight)]=saved[f'w{label}_k{count}_lambda{weight}_run{index}_positions']
    plot_results(output,result,curves,saved)


if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument('--plots-only',action='store_true',help='Redraw saved results without fitting')
    args=parser.parse_args()
    if args.plots_only:
        plot_saved()
    else:
        main()
