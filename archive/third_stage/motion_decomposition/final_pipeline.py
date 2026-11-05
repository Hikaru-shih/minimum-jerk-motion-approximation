"""Speed-valley count proposal, fitted timing/positions, and uniform controls."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix, fit_fixed_endpoints
from .goal_count_selection import detection_speed, speed_valley_knots, valley_initializations
from .initialization import make_initializations, times_from_knots, valid_timing_result
from .metrics import trajectory_errors
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv
from .timing_fit import fit_timing

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/final_pipeline'


def solve(task):
    label, k, index, start, t, p = task
    fit = fit_timing(t, p, p[0], p[-1], start['activation'], start['completion'],
                     2/120, max_nfev=1000)
    return label, k, index, start['name'], fit


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT/'checkpoints'
    cache.mkdir(exist_ok=True)
    source = ROOT.parents[1]/'results/wrist_trajectory.npz'
    with np.load(source) as z:
        ft, fp = z['timestamps'].astype(float), z['positions'].astype(float)
    protocol = {
        'method': 'Prominent speed lobes propose count; intervening speed minima seed times; free timing with endpoint-constrained linear LS at every residual evaluation',
        'counts': list(range(1,13)), 'windows': [[18,21],[30,33]],
        'filter_samples':17, 'polynomial_order':3, 'relative_prominence':.1,
        'minimum_peak_gap_s':.15, 'minimum_duration_s':2/120,
        'initial_overlap_fraction':.25, 'starts_per_count':4, 'max_nfev':1000,
        'objective':'Raw XYZ position squared error only; no opposition penalty',
        'selection':'Count from nominal detector, never adjusted from fitted error; lowest RMSE successful endpoint-valid fit at that count',
        'initializations':'Uniform base/jitter and motion seed base/jitter; at proposed K use detected-valley seed, at other K use speed-minimum slots; seed 23',
        'fairness':'Identical target, endpoints and prescribed K. Four nonlinear timing starts versus one exact linear fixed-time solve; unequal compute. Starts depend on K; no global-optimum claim.',
        'scope':'Two 3 s windows from the same original clip; not full-clip conversion or biological unit identification',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256':{n:hashlib.sha256((ROOT/'motion_decomposition'/n).read_bytes()).hexdigest()
                       for n in ('final_pipeline.py','timing_fit.py','fixed_times.py','goal_count_selection.py','initialization.py')},
    }
    pp = OUT/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        pp.write_text(json.dumps(protocol,indent=2)+'\n')
    rows, runs, saved, tasks, selected, sensitivity = [], [], {}, [], {}, []
    full_speed = {w:detection_speed(ft,fp,w) for w in (9,17,25)}
    windows = {}
    for left,right in protocol['windows']:
        label = f'{left}_{right}'
        mask = (ft>=left)&(ft<=right)
        t,p = ft[mask],fp[mask]
        windows[label] = (t,p)
        for name,value in [('timestamps',t),('original',p),('speed',full_speed[17][mask])]:
            saved[f'{label}_{name}'] = value
        det = speed_valley_knots(t,full_speed[17][mask],.1,.15)
        selected[label] = serializable(det)
        for w in (9,17,25):
            for prominence in (.05,.1,.2):
                d = speed_valley_knots(t,full_speed[w][mask],prominence,.15)
                sensitivity.append({'window':label,'filter_samples':w,'prominence':prominence,'K':d['component_count']})
        for k in protocol['counts']:
            a,b = times_from_knots(np.linspace(t[0],t[-1],k+1),.25)
            goals = np.column_stack([np.interp(np.linspace(t[0],t[-1],k+1)[1:],t,p[:,c]) for c in range(3)])
            d = np.diff(np.vstack([p[0],goals]),axis=0)
            fits = [('uniform_sampled',dict(activation=a,completion=b,displacements=d,positions=p[0]+design_matrix(t,a,b)@d)),
                    ('uniform_fitted',dict(activation=a,completion=b,**fit_fixed_endpoints(t,p,p[0],p[-1],a,b)))]
            for method,fit in fits:
                add_result(rows,saved,label,k,method,t,p,fit)
            if k == det['component_count']:
                starts = [valley_initializations(det['knots'])[i] for i in (0,1,5,6)]
            else:
                starts,_ = make_initializations(t,p,k,2/120,jitters_per_seed=1)
            for index,start in enumerate(starts):
                tasks.append((label,k,index,start,t,p))
    results = []
    pending = []
    for task in tasks:
        label,k,index,*_ = task
        cp = cache/f'{label}_k{k}_run{index}.json'
        if cp.exists():
            item=json.loads(cp.read_text())
            for name in ('activation','completion','displacements','positions'):
                item[4][name] = np.asarray(item[4][name])
            results.append(item)
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=6) as executor:
        for future in as_completed([executor.submit(solve,task) for task in pending]):
            item=future.result()
            label,k,index,_,fit=item
            (cache/f'{label}_k{k}_run{index}.json').write_text(json.dumps(serializable(item),allow_nan=False))
            results.append(item)
            print(label,k,index,fit['success'],fit['position_rmse_m'],flush=True)
    for label,(t,p) in windows.items():
        for k in protocol['counts']:
            group=[r for r in results if r[0]==label and r[1]==k]
            for _,_,index,name,fit in group:
                runs.append({'window':label,'K':k,'index':index,'initialization':name,
                             'eligible':valid_timing_result(fit),**{key:value for key,value in fit.items()
                             if key not in ('positions','activation','completion','displacements','timing_jacobian_singular_values')}})
            eligible=[r for r in group if valid_timing_result(r[4])]
            if eligible:
                best=min(eligible,key=lambda r:(r[4]['position_rmse_m'],r[2]))
                add_result(rows,saved,label,k,'adaptive_fitted',t,p,best[4],best[2])
            else:
                rows.append({'window':label,'K':k,'method':'adaptive_fitted','eligible':False})
    summary={'protocol':protocol,'selected':selected,'rows':rows,'runs':runs,'sensitivity':sensitivity}
    (OUT/'summary.json').write_text(json.dumps(serializable(summary),indent=2,allow_nan=False)+'\n')
    write_csv(OUT/'metrics.csv',rows)
    write_csv(OUT/'runs.csv',serializable(runs))
    np.savez_compressed(OUT/'fits.npz',**saved)
    plot(summary,saved)
    print('FINAL',[(r['window'],r['K'],r['method'],r.get('position_rmse_m')) for r in rows
                   if r['K']==selected[r['window']]['component_count']],flush=True)


def add_result(rows,saved,label,k,method,t,p,fit,index=None):
    reconstructed=p[0]+design_matrix(t,fit['activation'],fit['completion'])@fit['displacements']
    np.testing.assert_allclose(reconstructed,fit['positions'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(reconstructed[[0,-1]],p[[0,-1]],atol=1e-12,rtol=0)
    metrics,_=trajectory_errors(p,reconstructed,t)
    rows.append({'window':label,'K':k,'method':method,'eligible':True,'best_start':index,**metrics,
                 'sum_displacement_norms_m':float(np.linalg.norm(fit['displacements'],axis=1).sum())})
    for name in ('activation','completion','displacements','positions'):
        saved[f'{label}_{method}_k{k}_{name}']=fit[name]


def plot(s,z):
    colors={'uniform_sampled':'tab:gray','uniform_fitted':'tab:blue','adaptive_fitted':'tab:orange'}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,label in enumerate(s['selected']):
        for method,color in colors.items():
            rows=[r for r in s['rows'] if r['window']==label and r['method']==method]
            for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
                axes[i,j].plot([r['K'] for r in rows],[r.get(key,np.nan)*scale for r in rows],'o-',label=method,color=color)
        for j in range(2):
            axes[i,j].axvline(s['selected'][label]['component_count'],color='k',linestyle=':')
            axes[i,j].set(title=label+' s',xlabel='Prescribed component count K')
            axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel('Position RMSE (cm)'); axes[i,1].set_ylabel('Velocity RMSE (m/s)')
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'errors_vs_count.png',dpi=150);plt.close(fig)
    for label,det in s['selected'].items():
        k=det['component_count'];t=z[f'{label}_timestamps'];p=z[f'{label}_original']
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves=[('original',p,'black')]+[(m,z[f'{label}_{m}_k{k}_positions'],c) for m,c in colors.items()
                                        if f'{label}_{m}_k{k}_positions' in z]
        for name,curve,c in curves:
            v=np.gradient(curve,t,axis=0,edge_order=2)
            for j in range(3):
                axes[j,0].plot(t,curve[:,j],color=c,label=name,linewidth=1)
                axes[j,1].plot(t,v[:,j],color=c,linewidth=1)
        for j in range(3):
            axes[j,0].set_ylabel('XYZ'[j]+' position (m)');axes[j,1].set_ylabel('XYZ'[j]+' velocity (m/s)')
        axes[0,0].legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Clip time (s)')
        fig.suptitle(f'{label} s: speed-valley proposed K={k}; raw target, fixed endpoints')
        fig.tight_layout();fig.savefig(OUT/f'trajectories_{label}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    for ax,(label,det) in zip(axes,s['selected'].items()):
        t=z[f'{label}_timestamps'];v=z[f'{label}_speed']
        ax.plot(t,v,label='Detection speed (17-sample SG)')
        peaks=det['peak_frames'];ax.scatter(t[peaks],v[peaks],color='green',label='Retained peaks')
        for knot in det['knots'][1:-1]:ax.axvline(knot,color='red',linestyle=':')
        ax.set(title=f'{label} s: K={det["component_count"]}',ylabel='Speed (m/s)',xlabel='Clip time (s)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'count_selection.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
