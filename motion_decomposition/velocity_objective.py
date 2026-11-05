"""Isolated position/velocity objective experiment at nominal speed partitions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .fixed_times import design_matrix, fit_fixed_endpoints
from .io_utils import serializable, write_csv
from .metrics import trajectory_errors

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/velocity_objective'
WEIGHTS = (0.0, 0.01, 0.1, 1.0)


def scales(t, p):
    velocity = np.gradient(p, t, axis=0, edge_order=2)
    length = float(np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2, axis=1))))
    speed = float(np.sqrt(np.mean(np.sum(velocity**2, axis=1))))
    return max(length, 1e-12), max(speed, 1e-12)


def joint_residual(t, p, reconstructed, weight, length, speed):
    position = ((reconstructed-p)/length).ravel()
    if weight == 0:
        return position
    velocity = np.gradient(reconstructed-p, t, axis=0, edge_order=2)
    return np.r_[position, np.sqrt(weight)*velocity.ravel()/speed]


def fit_joint_fixed(t, p, a, b, weight, length, speed):
    if not np.isfinite(weight) or weight < 0 or length <= 0 or speed <= 0:
        raise ValueError('Nonnegative finite weight and positive scales required')
    if weight == 0:
        return fit_fixed_endpoints(t, p, p[0], p[-1], a, b)
    basis = design_matrix(t, a, b)
    if not np.all(basis[0] == 0) or not np.all(basis[-1] == 1):
        raise ValueError('Components must preserve crop endpoints')
    delta = p[-1]-p[0]
    free = basis[:, :-1]-basis[:, -1:]
    target = p-p[0]-basis[:, -1:]*delta
    matrix = np.vstack([free/length, np.sqrt(weight)*np.gradient(free, t, axis=0, edge_order=2)/speed])
    rhs = np.vstack([target/length, np.sqrt(weight)*np.gradient(target, t, axis=0, edge_order=2)/speed])
    if free.shape[1]:
        d, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=None)
    else:
        d, rank, singular = np.empty((0, 3)), 0, np.empty(0)
    displacements = np.vstack([d, delta-d.sum(axis=0)])
    return {'positions':p[0]+basis@displacements, 'displacements':displacements,
            'free_rank':int(rank), 'singular_values':singular}


def fit_joint(t, p, knots, weight):
    length, speed = scales(t, p)
    gaps = np.diff(knots)
    radius = np.r_[gaps[0], np.minimum(gaps[:-1], gaps[1:]), gaps[-1]]
    lower = np.r_[np.maximum(t[0],knots[:-1]-.45*radius[:-1]), knots[1:]-.20*radius[1:]]
    upper = np.r_[knots[:-1]+.20*radius[:-1], np.minimum(t[-1],knots[1:]+.45*radius[1:])]
    base = np.r_[np.maximum(t[0],knots[:-1]-.25*radius[:-1]), np.minimum(t[-1],knots[1:]+.25*radius[1:])]
    rng = np.random.default_rng(23)
    k = len(gaps)
    runs = []
    for index in range(4):
        x = base.copy() if index == 0 else lower+rng.uniform(.05,.95,2*k)*(upper-lower)
        def residual(v):
            f = fit_joint_fixed(t,p,v[:k],v[k:],weight,length,speed)
            return joint_residual(t,p,f['positions'],weight,length,speed)
        opt = least_squares(residual,x,bounds=(lower,upper),jac='3-point',x_scale='jac',
                            ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
        fit = fit_joint_fixed(t,p,opt.x[:k],opt.x[k:],weight,length,speed)
        errors,_ = trajectory_errors(p,fit['positions'],t)
        endpoint_error = float(np.max(np.linalg.norm(fit['positions'][[0,-1]]-p[[0,-1]],axis=1)))
        objective = float(np.sum(residual(opt.x)**2)/len(t))
        runs.append({'index':index,'success':bool(opt.success),'status':int(opt.status),
                     'message':str(opt.message),'nfev':int(opt.nfev),'objective':objective,
                     'eligible':bool(opt.success and np.isfinite(objective) and endpoint_error<1e-10),
                     'endpoint_error_m':endpoint_error,'activation':opt.x[:k],'completion':opt.x[k:],
                     'initial_timing':x,'sum_displacement_norms_m':float(np.linalg.norm(fit['displacements'],axis=1).sum()),
                     **fit,**errors})
    good = [r for r in runs if r['eligible']]
    return {'length_scale_m':length,'speed_scale_m_s':speed,'lower_bounds':lower,'upper_bounds':upper,
            'runs':runs,'best':min(good,key=lambda r:(r['objective'],r['index'])) if good else None}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUTPUT.mkdir(parents=True,exist_ok=True)
    source = ROOT/'results/mainline'
    summary = json.loads((source/'summary.json').read_text())
    data = dict(np.load(source/'fits.npz'))
    protocol = {'weights':WEIGHTS,'objective':'mean(||position_error||^2)/L^2 + lambda*mean(||velocity_error||^2)/V^2',
                'scales':'L=RMS centered raw position; V=RMS raw velocity; floor 1e-12; fixed per window across weights',
                'derivative':'numpy.gradient edge_order=2 for both linear basis and raw targets; same as evaluation',
                'controls':'Nominal K and knots, bounds .45/.20, four identical initial timings seed23, max_nfev1000, fixed position endpoints, no opposition penalty',
                'selection':'Lowest joint objective among successful endpoint-valid starts; no preferred weight selected after results',
                'scope':'Exploratory in-sample comparison; no held-out validation or physiological claim',
                'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('summary.json','fits.npz')},
                'numpy':np.__version__}
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows, details, saved = [], {}, {}
    for label in ('18_21','30_33'):
        t,p = data[f'{label}_timestamps'],data[f'{label}_original']
        knots = np.asarray(summary['details'][label]['detection']['knots'])
        saved.update({f'{label}_timestamps':t,f'{label}_original':p})
        fig,axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves = [('original',p)]
        for weight in WEIGHTS:
            result = fit_joint(t,p,knots,weight)
            key = f'{label}_lambda{weight:g}'
            details[key] = result
            best = result['best']
            if best is None:
                rows.append({'window':label,'weight':weight,'eligible':False})
                print(key,'NO ELIGIBLE FIT',flush=True)
                continue
            metrics,_ = trajectory_errors(p,best['positions'],t)
            zero_difference = float(np.max(np.abs(best['positions']-np.asarray(summary['details'][label]['fit']['positions'])))) if weight==0 else None
            rows.append({'window':label,'weight':weight,'K':len(knots)-1,'eligible':True,**metrics,
                         'objective':best['objective'],'endpoint_error_m':best['endpoint_error_m'],
                         'sum_displacement_norms_m':best['sum_displacement_norms_m'],
                         'successful_starts':sum(r['eligible'] for r in result['runs']),
                         'zero_control_max_difference_m':zero_difference})
            for name in ('positions','activation','completion','displacements'):saved[f'{key}_{name}']=best[name]
            curves.append((f'lambda={weight:g}',best['positions']))
            print(key,metrics,'successful',rows[-1]['successful_starts'],flush=True)
        for name,curve in curves:
            v = np.gradient(curve,t,axis=0,edge_order=2)
            for j in range(3):
                axes[j,0].plot(t,curve[:,j],label=name,linewidth=1)
                axes[j,1].plot(t,v[:,j],linewidth=1)
        for j in range(3):
            axes[j,0].set_ylabel('XYZ'[j]+' position (m)')
            axes[j,1].set_ylabel('XYZ'[j]+' velocity (m/s)')
        for ax in axes.flat:ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('Clip time (s)')
        axes[0,0].legend(fontsize=8)
        fig.suptitle(label+' position / velocity objective comparison')
        fig.tight_layout();fig.savefig(OUTPUT/f'trajectories_{label}.png',dpi=150);plt.close(fig)
    write_csv(OUTPUT/'metrics.csv',rows)
    np.savez_compressed(OUTPUT/'fits.npz',**saved)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable({'protocol':protocol,'rows':rows,'details':details}),indent=2,allow_nan=False)+'\n')
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest


if __name__ == '__main__':
    main()
