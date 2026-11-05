import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .final_pipeline import add_result
from .fixed_times import fit_fixed_endpoints
from .initialization import times_from_knots
from .metrics import trajectory_errors
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv
from .valley_fit import fit_valley_units

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/mainline'


def main():
    OUT.mkdir(exist_ok=True)
    source=ROOT/'results/final_pipeline'
    previous=json.loads((source/'summary.json').read_text())
    protocol={'count_rule':previous['protocol'],'timing_bounds':
              'For each boundary, r=min adjacent lobe widths (edge: one width). Onset in [left-.45r,left+.20r]; completion in [right-.20r,right+.45r], clipped to crop. Four starts seed23. Positions via endpoint-constrained linear LS for each trial time.',
              'reason':'Associate one component with each detected speed lobe; optimize timing locally with overlap. These are explicit heuristic association bounds, not inferred physiological limits.',
              'objective':'Raw position squared error; no opposition penalty',
              'count_sweep':'Repeat detector with prominence from .01 to .40, fixed 17-sample smoothing and .15 s separation; each unique valley partition is fitted. Uniform controls at same K. Different K also changes detected partition; not a pure K-only causal comparison.',
              'failure_policy':'Keep failed starts and missing fits; nominal prominence .10 is not retuned to minimize error.',
              'source_sha256':hashlib.sha256((source/'summary.json').read_bytes()).hexdigest(),
              'code_sha256':{n:hashlib.sha256((ROOT/'motion_decomposition'/n).read_bytes()).hexdigest() for n in ('valley_fit.py','complete_mainline.py')}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    z=dict(np.load(source/'fits.npz'));rows=[];saved={};details={};sweep=[]
    from .goal_count_selection import speed_valley_knots
    for label,det in previous['selected'].items():
        t,p,speed=[z[f'{label}_{n}'] for n in ('timestamps','original','speed')]
        for n,v in [('timestamps',t),('original',p),('speed',speed)]:saved[f'{label}_{n}']=v
        candidates={}
        for prominence in sorted(set([.10,*np.arange(.01,.401,.01).round(2).tolist()])):
            detection=speed_valley_knots(t,speed,float(prominence),.15)
            key=tuple(detection['boundary_frames'].tolist())
            candidates.setdefault(key,{'detection':detection,'prominences':[]})['prominences'].append(float(prominence))
        nominal_key=tuple(det['boundary_frames'])
        for number,(key,candidate) in enumerate(candidates.items()):
            d=candidate['detection'];k=d['component_count']
            result=fit_valley_units(t,p,d['knots'])
            name=f'{label}_partition{number}'
            details[name]={'detection':d,'prominences':candidate['prominences'],**result}
            if result['best'] is None:
                sweep.append({'window':label,'partition':number,'K':k,'prominences':str(candidate['prominences']),'eligible':False})
                continue
            fit=result['best'];metrics,_=trajectory_errors(p,fit['positions'],t)
            sweep.append({'window':label,'partition':number,'K':k,'prominences':str(candidate['prominences']),
                          'nominal':key==nominal_key,'eligible':True,**metrics,
                          'sum_displacement_norms_m':float(np.linalg.norm(fit['displacements'],axis=1).sum())})
            for n in ('positions','activation','completion','displacements'):saved[f'{name}_{n}']=fit[n]
            if key==nominal_key:
                add_result(rows,saved,label,k,'valley_fitted',t,p,fit,fit['index'])
                details[label]={'nominal_partition':number,'detection':d,'fit':fit}
            print(label,'partition',number,'K',k,'rmse',metrics['position_rmse_m'],flush=True)
        for k in sorted({d['detection']['component_count'] for d in candidates.values()}):
            for method in ('uniform_sampled','uniform_fitted'):
                r=next((r for r in previous['rows'] if r['window']==label and r['K']==k and r['method']==method),None)
                if r is None:
                    a,b=times_from_knots(np.linspace(t[0],t[-1],k+1),.25)
                    f=fit_fixed_endpoints(t,p,p[0],p[-1],a,b)
                    if method=='uniform_sampled':
                        from .fixed_times import design_matrix
                        g=np.column_stack([np.interp(np.linspace(t[0],t[-1],k+1)[1:],t,p[:,c]) for c in range(3)])
                        dd=np.diff(np.vstack([p[0],g]),axis=0)
                        f={'positions':p[0]+design_matrix(t,a,b)@dd,'displacements':dd}
                    f.update(activation=a,completion=b)
                    add_result(rows,saved,label,k,method,t,p,f)
                else:
                    rows.append(r)
                    for n in ('positions','activation','completion','displacements'):saved[f'{label}_{method}_k{k}_{n}']=z[f'{label}_{method}_k{k}_{n}']
    summary={'protocol':protocol,'rows':rows,'sweep':sweep,'details':details,'detector_sensitivity':previous['sensitivity']}
    (OUT/'summary.json').write_text(json.dumps(serializable(summary),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'fits.npz',**saved);write_csv(OUT/'metrics.csv',rows);write_csv(OUT/'count_sweep.csv',sweep)
    draw(summary,saved)


def draw(s,z):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,label in enumerate(('18_21','30_33')):
        for method in ('uniform_sampled','uniform_fitted'):
            rr=sorted([r for r in s['rows'] if r['window']==label and r['method']==method],key=lambda r:r['K'])
            for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
                axes[i,j].plot([r['K'] for r in rr],[r[key]*scale for r in rr],'o-',label=method)
        rr=[r for r in s['sweep'] if r['window']==label and r['eligible']]
        for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
            axes[i,j].scatter([r['K'] for r in rr],[r[key]*scale for r in rr],color='tab:green',label='valley partitions')
            for r in rr:
                if r['nominal']:axes[i,j].scatter(r['K'],r[key]*scale,marker='*',s=150,color='black',label='nominal prominence 0.10')
            axes[i,j].set(title=label+' s',xlabel='Number of components K');axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel('Position RMSE (cm)');axes[i,1].set_ylabel('Velocity RMSE (m/s)')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'errors_vs_count.png',dpi=160);plt.close(fig)
    for label in ('18_21','30_33'):
        t,p=z[f'{label}_timestamps'],z[f'{label}_original'];d=s['details'][label];k=d['detection']['component_count']
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for name,curve,color in [('original',p,'black')]+[(m,z[f'{label}_{m}_k{k}_positions'],c) for m,c in [('uniform_sampled','tab:gray'),('uniform_fitted','tab:blue'),('valley_fitted','tab:green')]]:
            v=np.gradient(curve,t,axis=0,edge_order=2)
            for j in range(3):
                axes[j,0].plot(t,curve[:,j],label=name,color=color,linewidth=1);axes[j,1].plot(t,v[:,j],color=color,linewidth=1)
        for j in range(3):axes[j,0].set_ylabel('XYZ'[j]+' position (m)');axes[j,1].set_ylabel('XYZ'[j]+' velocity (m/s)')
        for ax in axes.flat:ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Clip time (s)')
        axes[0,0].legend(fontsize=8);fig.suptitle(f'{label} s nominal K={k}');fig.tight_layout();fig.savefig(OUT/f'trajectories_{label}.png',dpi=160);plt.close(fig)
        from .step4_diagnose_initialization import component_velocities
        vv=component_velocities(t,d['fit']['activation'],d['fit']['completion'],d['fit']['displacements'])
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot(t,z[f'{label}_speed'],color='black',label='Detection speed')
        for knot in d['detection']['knots'][1:-1]:axes[0].axvline(knot,color='red',linestyle=':')
        for j in range(k):
            axes[1].plot(t,np.linalg.norm(vv[:,j,:],axis=1),label=f'Component {j+1}')
        axes[0].set_ylabel('Speed (m/s)');axes[1].set_ylabel('Component speed (m/s)');axes[1].set_xlabel('Clip time (s)')
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/f'components_{label}.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
