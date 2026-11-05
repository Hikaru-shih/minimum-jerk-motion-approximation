import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix, fit_fixed_endpoints
from .goal_count_selection import chord_knee
from .initialization import make_initializations, valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv
from .step9_count_comparison import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/step14_extended_counts'
FIELDS = ('activation', 'completion', 'displacements', 'positions')


def fit_task(task):
    window, k, weight, index, start, t, p = task
    begun = time.perf_counter()
    fit = fit_joint(t, p, p[0], p[-1], start['activation'], start['completion'],
                    2/120, weight, .3, max_nfev=1000)
    np.testing.assert_allclose(p[0]+design_matrix(t, fit['activation'], fit['completion'])@fit['displacements'],
                               fit['positions'], atol=1e-9, rtol=0)
    assert fit['initial_error_m']<1e-12 and fit['final_error_m']<1e-12
    assert fit['objective_m2']<=fit['initial_objective_m2']+1e-10
    assert np.all(fit['activation']>=t[0]) and np.all(fit['completion']<=t[-1])
    assert np.all(fit['completion']-fit['activation']>=2/120-1e-12)
    metrics, _ = trajectory_errors(p, fit['positions'], t)
    composition = composition_metrics(t, component_velocities(t, fit['activation'], fit['completion'], fit['displacements']))
    row = {'window': window, 'component_count': k, 'weight': weight, 'index': index,
           'initialization': start['name'], 'origin': 'extension_solve', 'eligible': valid_timing_result(fit),
           **metrics, **{key:value for key,value in fit.items() if key not in FIELDS and key!='position_rmse_m'},
           'relative_component_length_excess': composition['relative_component_length_excess'],
           'elapsed_seconds': time.perf_counter()-begun, 'initialization_details': start,
           **{key:fit[key] for key in FIELDS if key!='positions'}}
    return serializable(row), {key:fit[key] for key in FIELDS}


def main(workers, plots_only):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if plots_only:
        result = json.loads((OUTPUT/'summary.json').read_text())
        with np.load(OUTPUT/'fits.npz') as z: arrays = dict(z)
        plot_results(result, arrays)
        return
    source = ROOT/'results/step9_count_comparison'
    hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('summary.json','fits.npz')}
    previous = json.loads((source/'summary.json').read_text())
    assert previous['settings']==SETTINGS
    protocol = {'scope':'Prespecified finite extension through K=12, not a final count selector',
                'settings':SETTINGS, 'counts':list(range(1,13)), 'weights':[0,1e-6],
                'source_hashes':hashes, 'new_counts':list(range(7,13)),
                'planned_new_solves':240, 'selection':'Lowest own objective among eligible fits; no fallback',
                'upper_limit_reason':'User-authorized finite computational boundary, not physiological evidence',
                'fairness':'Same ten start recipe and per-fit nfev cap; dimension and actual compute grow with K',
                'score_limit':'IID Gaussian BIC-style screen only; temporal dependence and degeneracy remain',
                'workers':workers}
    protocol_path = OUTPUT/'protocol.json'
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        assert all(old[key]==protocol[key] for key in ('settings','counts','weights','source_hashes'))
    else:
        protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    checkpoint = OUTPUT/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    details = previous['details'].copy()
    fixed = previous['fixed_time_baselines'].copy()
    with np.load(source/'fits.npz') as z: arrays = dict(z)
    tasks = []
    for window in ('18_21','30_33'):
        t, p = arrays[f'w{window}_timestamps'], arrays[f'w{window}_original']
        for k in range(7,13):
            starts, _ = make_initializations(t,p,k,2/120)
            bases = []
            for i,start in enumerate(starts):
                assert start['valid']
                fit = fit_fixed_endpoints(t,p,p[0],p[-1],start['activation'],start['completion'])
                metrics, _ = trajectory_errors(p,fit['positions'],t)
                bases.append({'window':window,'component_count':k,'index':i,**metrics})
            fixed.append(min(bases,key=lambda r:(r['position_rmse_m'],r['index'])))
            for weight in (0,1e-6):
                for i,start in enumerate(starts):
                    prefix = f'w{window}_k{k}_lambda{weight}_run{i}'
                    path = checkpoint/(prefix+'.json')
                    if path.exists() and path.with_suffix('.npz').exists():
                        row = json.loads(path.read_text())
                        with np.load(path.with_suffix('.npz')) as z:
                            arrays.update({prefix+'_'+key:z[key].copy() for key in FIELDS})
                        details.append(row)
                    else:
                        tasks.append((window,k,weight,i,start,t,p))
    print('Pending fits',len(tasks),'workers',workers,flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_task,task):task[:4] for task in tasks}
        for future in as_completed(futures):
            row, fit = future.result()
            window,k,weight,i = futures[future]
            prefix = f'w{window}_k{k}_lambda{weight}_run{i}'
            np.savez_compressed(checkpoint/(prefix+'.npz'), **fit)
            (checkpoint/(prefix+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            arrays.update({prefix+'_'+key:value for key,value in fit.items()})
            details.append(row)
            completed = len(details)-240
            print('DONE',completed,'/240',window,k,weight,i,'eligible',row['eligible'],
                  'cm',round(100*row['position_rmse_m'],5),'seconds',round(row['elapsed_seconds'],1),flush=True)
            (OUTPUT/'progress.json').write_text(json.dumps({'completed_extension_records':completed,
                'planned_extension_records':240,'last':row},indent=2)+'\n')
    details.sort(key=lambda r:(r['window'],r['component_count'],r['weight'],r['index']))
    selections, screens = [], []
    for window in ('18_21','30_33'):
        for weight in (0,1e-6):
            for k in range(1,13):
                group = [r for r in details if r['window']==window and r['weight']==weight and r['component_count']==k]
                assert len(group)==10
                eligible = [r for r in group if r['eligible']]
                best = min(eligible,key=lambda r:(r['objective_m2'],r['index'])) if eligible else None
                selections.append({'window':window,'weight':weight,'component_count':k,'selected':best,
                                   'eligible_count':len(eligible),'best_index':best['index'] if best else None})
            group = [r for r in selections if r['window']==window and r['weight']==weight]
            for upper in (6,8,10,12):
                subset = group[:upper]
                knee = chord_knee(range(1,upper+1),[r['selected']['objective_m2'] for r in subset]) if all(r['selected'] for r in subset) else {'selected_count':None,'reason':'Missing eligible fit'}
                scores = []
                for r in subset:
                    if r['selected']:
                        prefix = f'w{window}_k{r["component_count"]}_lambda{weight}_run{r["best_index"]}'
                        e = (arrays[prefix+'_positions']-arrays[f'w{window}_original'])[1:-1]
                        n = e.size
                        scores.append((float(n*np.log(np.sum(e*e)/n)+(5*r['component_count']-2)*np.log(n)),r['component_count']))
                screens.append({'window':window,'weight':weight,'upper_K':upper,'knee_K':knee['selected_count'],
                                'knee_reason':knee['reason'],'bic_style_K':min(scores)[1] if scores else None,
                                'missing_counts':sum(r['selected'] is None for r in subset)})
    result = {**protocol,'details':details,'selections':selections,'fixed_time_baselines':fixed,'screens':screens}
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
    columns = ('window','component_count','weight','index','eligible','position_rmse_m','velocity_rmse_m_s',
               'max_position_error_m','objective_m2','opposition_cost_m2','sum_displacement_norms_m',
               'relative_component_length_excess','status','nfev','residual_evaluations','parameter_count','joint_jacobian_rank')
    write_csv(OUTPUT/'selected_metrics.csv',[{key:r['selected'][key] for key in columns} for r in selections if r['selected']])
    write_csv(OUTPUT/'runs.csv',[{key:r[key] for key in columns} for r in details])
    write_csv(OUTPUT/'screens.csv',screens)
    np.savez_compressed(OUTPUT/'fits.npz',**arrays)
    plot_results(result,arrays)
    for name,digest in hashes.items(): assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    print('COMPLETE',len(details),'records',len(selections),'groups',flush=True)


def plot_results(result,arrays):
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    for w,window in enumerate(('18_21','30_33')):
        for weight,color in ((0,'tab:blue'),(1e-6,'tab:orange')):
            group = [r for r in result['selections'] if r['window']==window and r['weight']==weight]
            for c,(key,scale) in enumerate((('position_rmse_m',100),('velocity_rmse_m_s',1),('sum_displacement_norms_m',1))):
                axes[w,c].plot(range(1,13),[scale*r['selected'][key] if r['selected'] else np.nan for r in group],
                               'o-',color=color,label=f'lambda={weight:g}')
                for r in result['details']:
                    if r['window']==window and r['weight']==weight and not r['eligible']:
                        axes[w,c].scatter(r['component_count'],scale*r[key],marker='x',color='red',s=12,alpha=.3)
            for r in group:
                eligible = [x for x in result['details'] if x['window']==window and x['weight']==weight and x['component_count']==r['component_count'] and x['eligible']]
                if eligible:
                    axes[w,0].vlines(r['component_count']+(-.04 if weight==0 else .04),
                                     100*min(x['position_rmse_m'] for x in eligible),100*max(x['position_rmse_m'] for x in eligible),color=color,alpha=.4)
        for c,label in enumerate(('Position RMSE (cm)','Velocity RMSE (m/s)','Sum displacement norms (m, log)')):
            axes[w,c].set_title(window.replace('_','-')+' s: '+label,fontsize=10)
            axes[w,c].set_xticks(range(1,13)); axes[w,c].set_xlabel('Prescribed K'); axes[w,c].grid(alpha=.2)
        axes[w,2].set_yscale('log')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Finite K=1-12 extension; red x: ineligible iterates; vertical ranges: eligible starts')
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(OUTPUT/'errors_and_size.png',dpi=150); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4))
    for axis,window in zip(axes,('18_21','30_33')):
        for weight in (0,1e-6):
            group = [r for r in result['selections'] if r['window']==window and r['weight']==weight]
            axis.plot(range(1,13),[r['eligible_count'] if r['component_count']>1 else int(r['eligible_count']>0) for r in group],
                      'o-',label=f'lambda={weight:g}')
        axis.set_title(window.replace('_','-')+' s'); axis.set_xticks(range(1,13)); axis.set_ylim(-.3,10.5)
        axis.set_xlabel('Prescribed K; K=1 identical starts counted once')
        axis.set_ylabel('Eligible starts'); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT/'eligible_starts.png',dpi=150); plt.close(fig)
    for window in ('18_21','30_33'):
        t,original = arrays[f'w{window}_timestamps'],arrays[f'w{window}_original']
        fig,axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves = [('Original',original,'black')]
        for k,color in ((6,'tab:blue'),(9,'tab:orange'),(12,'tab:green')):
            row = next(r for r in result['selections'] if r['window']==window and r['weight']==1e-6 and r['component_count']==k)
            if row['selected']:
                curves.append((f'K={k}',arrays[f'w{window}_k{k}_lambda1e-06_run{row["best_index"]}_positions'],color))
        for name,p,color in curves:
            v = np.gradient(p,t,axis=0,edge_order=2)
            for c in range(3):
                axes[c,0].plot(t,p[:,c],label=name,color=color,linewidth=1)
                axes[c,1].plot(t,v[:,c],label=name,color=color,linewidth=1)
        for c in range(3):
            axes[c,0].set_ylabel('XYZ'[c]+' position (m)'); axes[c,1].set_ylabel('XYZ'[c]+' velocity (m/s)')
        for axis in axes.flat: axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        for axis in axes[-1]: axis.set_xlabel('Clip time (s)')
        fig.suptitle(window.replace('_','-')+' s; lambda=1e-6; displayed K=6,9,12 chosen before fitting')
        fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(OUTPUT/f'trajectories_{window}.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--plots-only',action='store_true')
    args = parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    main(args.workers,args.plots_only)
