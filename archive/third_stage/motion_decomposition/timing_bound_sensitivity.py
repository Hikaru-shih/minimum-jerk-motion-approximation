"""Prespecified local timing-bound sensitivity at the nominal detected counts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from .fixed_times import fit_fixed_endpoints
from .metrics import trajectory_errors
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'results/timing_bound_sensitivity'
SETTINGS = (('narrow',.30,.10),('nominal',.45,.20),('wide',.60,.30))


def fit_variant(t, p, knots, outward, inward):
    t,p,knots = (np.asarray(x,dtype=float) for x in (t,p,knots))
    if not 0 < inward < .5 or not outward >= .25:
        raise ValueError('Inward fraction must be between 0 and .5; outward at least .25 for shared initial timing')
    gaps = np.diff(knots)
    radius = np.r_[gaps[0],np.minimum(gaps[:-1],gaps[1:]),gaps[-1]]
    lower = np.r_[np.maximum(t[0],knots[:-1]-outward*radius[:-1]),knots[1:]-inward*radius[1:]]
    upper = np.r_[knots[:-1]+inward*radius[:-1],np.minimum(t[-1],knots[1:]+outward*radius[1:])]
    base = np.r_[np.maximum(t[0],knots[:-1]-.25*radius[:-1]),np.minimum(t[-1],knots[1:]+.25*radius[1:])]
    k = len(gaps)
    rng = np.random.default_rng(23)
    runs = []
    for index in range(4):
        start = base.copy() if index == 0 else lower+rng.uniform(.05,.95,2*k)*(upper-lower)
        def residual(x):
            fit=fit_fixed_endpoints(t,p,p[0],p[-1],x[:k],x[k:])
            return (fit['positions']-p).ravel()
        result=least_squares(residual,start,bounds=(lower,upper),jac='3-point',x_scale='jac',
                             ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
        a,b=result.x[:k],result.x[k:]
        fit=fit_fixed_endpoints(t,p,p[0],p[-1],a,b)
        metrics,_=trajectory_errors(p,fit['positions'],t)
        eligible=bool(result.success and np.isfinite(fit['positions']).all()
                      and fit['initial_error_m']<=1e-12 and fit['final_error_m']<=1e-12)
        runs.append({'index':index,'activation':a,'completion':b,
                     'displacements':fit['displacements'],'positions':fit['positions'],
                     'initial_timing':start,'success':bool(result.success),'eligible':eligible,
                     'status':int(result.status),'message':str(result.message),'nfev':int(result.nfev),
                     'free_rank':fit['free_rank'],'free_columns':fit['free_columns'],
                     'condition_number':fit['condition_number'],**metrics,
                     'sum_displacement_norms_m':float(np.linalg.norm(fit['displacements'],axis=1).sum()),
                     'bound_hits':int(np.sum(np.minimum(result.x-lower,upper-result.x)<=1e-6*(upper-lower)))})
    good=[r for r in runs if r['eligible']]
    return {'lower_bounds':lower,'upper_bounds':upper,'runs':runs,
            'best':min(good,key=lambda r:(r['position_rmse_m'],r['index'])) if good else None}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/mainline'
    previous=json.loads((source/'summary.json').read_text())
    arrays=dict(np.load(source/'fits.npz'))
    protocol={'purpose':'Timing-bound sensitivity only; fixed nominal detector, K, knots and raw targets',
              'settings':[{'name':name,'outward':out,'inward':inside} for name,out,inside in SETTINGS],
              'choice_reason':'Narrower and wider nested neighborhoods around the existing nominal setting, selected before new fits; not tuned for best results',
              'starts':4,'seed':23,'max_nfev':1000,'objective':'Raw XYZ position least squares, no opposition penalty',
              'initialization':'Identical boundary extension .25 for start 0; same normalized random draws for starts 1..3, mapped into each feasible box',
              'scope':'18-21 s K4 and 30-33 s K3 only; not all count partitions, filter settings, full clip or multiple subjects',
              'selection':'Lowest position RMSE among successful endpoint-valid fits; no replacing nominal setting by whichever setting scores best',
              'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('summary.json','fits.npz')},
              'code_sha256':{n:hashlib.sha256((ROOT/'motion_decomposition'/n).read_bytes()).hexdigest() for n in ('timing_bound_sensitivity.py','fixed_times.py')}}
    protocol_path=OUTPUT/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==protocol
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    details={};rows=[];runs=[];saved={}
    for label in ('18_21','30_33'):
        t,p=arrays[f'{label}_timestamps'],arrays[f'{label}_original']
        knots=np.asarray(previous['details'][label]['detection']['knots'])
        for name,value in [('timestamps',t),('original',p),('knots',knots)]:saved[f'{label}_{name}']=value
        for name,outward,inward in SETTINGS:
            result=fit_variant(t,p,knots,outward,inward)
            details[f'{label}_{name}']=result
            for fit in result['runs']:
                runs.append({'window':label,'setting':name,'K':len(knots)-1,
                             **{key:value for key,value in fit.items() if key not in ('positions','activation','completion','displacements','initial_timing')}})
            best=result['best']
            if best is None:
                rows.append({'window':label,'setting':name,'eligible':False})
                continue
            if name=='nominal':
                old=previous['details'][label]['fit']
                np.testing.assert_allclose(best['positions'],old['positions'],atol=1e-10,rtol=0)
                assert best['index']==old['index']
            row={'window':label,'setting':name,'outward':outward,'inward':inward,'K':len(knots)-1,
                 'eligible':True,'best_start':best['index'],
                 **{key:best[key] for key in ('position_rmse_m','velocity_rmse_m_s','max_position_error_m','sum_displacement_norms_m','bound_hits','condition_number')}}
            rows.append(row)
            for field in ('activation','completion','displacements','positions'):saved[f'{label}_{name}_{field}']=best[field]
            print(row,flush=True)
    summary={'protocol':protocol,'rows':rows,'runs':runs,'details':details}
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),indent=2,allow_nan=False)+'\n')
    write_csv(OUTPUT/'metrics.csv',rows);write_csv(OUTPUT/'runs.csv',runs)
    np.savez_compressed(OUTPUT/'fits.npz',**saved)
    draw(summary,saved)


def draw(s,z):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,label in enumerate(('18_21','30_33')):
        rows=[r for r in s['rows'] if r['window']==label and r['eligible']]
        for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
            axes[i,j].plot([r['setting'] for r in rows],[r[key]*scale for r in rows],'o-')
            axes[i,j].set_title(f'{label} s, same K={rows[0]["K"]}')
            axes[i,j].grid(alpha=.2)
        axes[i,0].set_ylabel('Position RMSE (cm)');axes[i,1].set_ylabel('Velocity RMSE (m/s)')
    fig.tight_layout();fig.savefig(OUTPUT/'sensitivity.png',dpi=160);plt.close(fig)
    for label in ('18_21','30_33'):
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        t=z[f'{label}_timestamps'];p=z[f'{label}_original']
        curves=[('original',p,'black')]+[(name,z[f'{label}_{name}_positions'],c) for (name,_,_),c in zip(SETTINGS,('tab:blue','tab:orange','tab:green')) if f'{label}_{name}_positions' in z]
        for name,curve,color in curves:
            v=np.gradient(curve,t,axis=0,edge_order=2)
            for j in range(3):axes[j,0].plot(t,curve[:,j],color=color,label=name,linewidth=1);axes[j,1].plot(t,v[:,j],color=color,linewidth=1)
        for j in range(3):axes[j,0].set_ylabel('XYZ'[j]+' position (m)');axes[j,1].set_ylabel('XYZ'[j]+' velocity (m/s)')
        for axis in axes.flat:axis.grid(alpha=.2)
        for axis in axes[-1]:axis.set_xlabel('Clip time (s)')
        axes[0,0].legend(fontsize=8);fig.suptitle(f'{label} s timing-bound sensitivity')
        fig.tight_layout();fig.savefig(OUTPUT/f'trajectories_{label}.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
