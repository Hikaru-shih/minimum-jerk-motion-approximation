import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .initialization import times_from_knots
from .metrics import trajectory_errors
from .opposition import fit_opposition_fixed_times, opposition_residuals
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = 1e-6
METHODS = ('uniform_sampled_goals', 'uniform_fitted_displacements', 'adaptive_timing')


def main():
    source = ROOT/'results/step14_extended_counts'
    output = ROOT/'results/uniform_baseline_comparison'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('summary.json','fits.npz')}
    summary = json.loads((source/'summary.json').read_text())
    rows, saved = [], {}
    protocol = {'scope':'Same-window, same-K baseline comparison; no new count selector',
                'windows':['18_21','30_33'],'counts':list(range(1,13)), 'weight':WEIGHT,
                'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,
                'uniform_times':'K equal intervals; extend internal sides by 0.25 interval, clamp to crop',
                'sampled_goals':'Linear interpolation of observed XYZ at uniform interval ends',
                'fitted_control':'Same fixed timing, endpoint-constrained fitted displacement with same opposition objective',
                'fairness_limit':'Adaptive fits have ten timing starts; fixed-time control has one linear-LS displacement start; not equal compute or proof of global optimum',
                'crop_limit':'All methods independently reconstruct each 3 s crop with observed endpoints; not a crop of the original full-clip reconstruction',
                'source_hashes':hashes}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with np.load(source/'fits.npz') as z:
        for window in protocol['windows']:
            t,p = z[f'w{window}_timestamps'],z[f'w{window}_original']
            saved[f'{window}_timestamps'],saved[f'{window}_original'] = t,p
            for k in range(1,13):
                knots = np.linspace(t[0],t[-1],k+1)
                a,b = times_from_knots(knots,.25)
                goals = np.column_stack([np.interp(knots[1:],t,p[:,c]) for c in range(3)])
                d = np.diff(np.vstack([p[0],goals]),axis=0)
                copied = p[0]+design_matrix(t,a,b)@d
                fixed = fit_opposition_fixed_times(t,p,p[0],p[-1],a,b,WEIGHT,.3,max_nfev=1000)
                selection = next(r for r in summary['selections'] if r['window']==window and r['weight']==WEIGHT and r['component_count']==k)
                index = selection['best_index']
                prefix = f'w{window}_k{k}_lambda1e-06_run{index}'
                fits = [(METHODS[0],a,b,d,copied,True,0,'No optimization'),
                        (METHODS[1],a,b,fixed['displacements'],fixed['positions'],fixed['success'],fixed['nfev'],fixed['message']),
                        (METHODS[2],z[prefix+'_activation'],z[prefix+'_completion'],z[prefix+'_displacements'],z[prefix+'_positions'],True,selection['selected']['nfev'],'Reused selected fit')]
                for method,activation,completion,displacements,fitted,success,nfev,message in fits:
                    metrics,_ = trajectory_errors(p,fitted,t)
                    q = opposition_residuals(t,activation,completion,displacements,.3)
                    penalty = float(q@q/len(t))
                    np.testing.assert_allclose(p[0]+design_matrix(t,activation,completion)@displacements,fitted,atol=1e-12)
                    np.testing.assert_allclose(fitted[[0,-1]],p[[0,-1]],atol=1e-12,rtol=0)
                    row = {'window':window,'K':k,'method':method,'eligible':bool(success),**metrics,
                           'opposition_cost_m2':penalty,'objective_m2':metrics['position_rmse_m']**2+WEIGHT*penalty,
                           'sum_displacement_norms_m':float(np.linalg.norm(displacements,axis=1).sum()),
                           'nfev':nfev,'message':message}
                    rows.append(row)
                    for name,value in (('positions',fitted),('activation',activation),('completion',completion),('displacements',displacements)):
                        saved[f'{window}_{method}_k{k}_{name}'] = value
    result = {**protocol,'rows':rows,'new_fixed_time_control_solves':24,
              'selected_counts':{'18_21':3,'30_33':4},
              'selected_count_source':'Existing total-objective knee at upper K=12, not chosen from baseline scores'}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
    write_csv(output/'metrics.csv',rows)
    np.savez_compressed(output/'fits.npz',**saved)
    plot_results(output,result,saved)
    for name,digest in hashes.items(): assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for row in rows:
        if row['K'] in (result['selected_counts'][row['window']],6): print(row,flush=True)


def plot_results(output,result,saved):
    colors = ('tab:gray','tab:blue','tab:orange')
    names = ('Uniform times, sampled goals','Uniform times, fitted displacement','Fitted timing and displacement')
    fig,axes = plt.subplots(2,2,figsize=(12,8))
    for r,window in enumerate(result['windows']):
        for method,color,name in zip(METHODS,colors,names):
            group = [x for x in result['rows'] if x['window']==window and x['method']==method]
            for c,key,scale in ((0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)):
                axes[r,c].plot(range(1,13),[scale*x[key] if x['eligible'] else np.nan for x in group],'o-',color=color,label=name)
        for c in range(2):
            axes[r,c].set_title(window.replace('_','-')+' s'); axes[r,c].set_xticks(range(1,13)); axes[r,c].set_xlabel('Same prescribed K')
            axes[r,c].axvline(result['selected_counts'][window],color='black',linestyle=':',alpha=.5)
            axes[r,c].grid(alpha=.2)
        axes[r,0].set_ylabel('Position RMSE (cm)'); axes[r,1].set_ylabel('Velocity RMSE (m/s)')
    axes[0,0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'errors_vs_count.png',dpi=160); plt.close(fig)
    for window,k in result['selected_counts'].items():
        t,p = saved[f'{window}_timestamps'],saved[f'{window}_original']
        fig,axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves = [('Original',p,'black')]+[(name,saved[f'{window}_{method}_k{k}_positions'],color) for method,color,name in zip(METHODS,colors,names)]
        for name,fitted,color in curves:
            v = np.gradient(fitted,t,axis=0,edge_order=2)
            for c in range(3):
                axes[c,0].plot(t,fitted[:,c],label=name,color=color,linewidth=1)
                axes[c,1].plot(t,v[:,c],label=name,color=color,linewidth=1)
        for c in range(3):
            axes[c,0].set_ylabel('XYZ'[c]+' position (m)'); axes[c,1].set_ylabel('XYZ'[c]+' velocity (m/s)')
        for axis in axes.flat: axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        for axis in axes[-1]: axis.set_xlabel('Clip time (s)')
        fig.suptitle(window.replace('_','-')+f' s: same K={k}, same endpoints and raw target')
        fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(output/f'trajectories_{window}.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    main()
