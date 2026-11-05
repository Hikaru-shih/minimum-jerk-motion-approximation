"""Fit components associated with detected speed lobes, allowing local overlap."""
import numpy as np
from scipy.optimize import least_squares

from .fixed_times import fit_fixed_endpoints


def fit_valley_units(times, positions, knots, seed=23, starts=4):
    times, positions, knots = map(lambda x: np.asarray(x,dtype=float), (times,positions,knots))
    if (times.ndim!=1 or positions.shape!=(len(times),3) or knots.ndim!=1
            or len(knots)<2 or not np.isfinite(knots).all() or np.any(np.diff(knots)<=0)
            or knots[0]!=times[0] or knots[-1]!=times[-1]):
        raise ValueError('Increasing knots must cover the observation interval')
    gaps=np.diff(knots)
    k=len(gaps)
    # Each onset/end stays near its own lobe boundary. Adjacent supports may overlap.
    radius=np.r_[gaps[0],np.minimum(gaps[:-1],gaps[1:]),gaps[-1]]
    alo=np.maximum(times[0],knots[:-1]-.45*radius[:-1])
    ahi=knots[:-1]+.20*radius[:-1]
    blo=knots[1:]-.20*radius[1:]
    bhi=np.minimum(times[-1],knots[1:]+.45*radius[1:])
    lower=np.r_[alo,blo];upper=np.r_[ahi,bhi]
    a=np.maximum(times[0],knots[:-1]-.25*radius[:-1])
    b=np.minimum(times[-1],knots[1:]+.25*radius[1:])
    rng=np.random.default_rng(seed)
    results=[]
    for i in range(starts):
        x=np.r_[a,b] if i==0 else lower+rng.uniform(.05,.95,2*k)*(upper-lower)
        def residual(v):
            fit=fit_fixed_endpoints(times,positions,positions[0],positions[-1],v[:k],v[k:])
            return (fit['positions']-positions).ravel()
        opt=least_squares(residual,x,bounds=(lower,upper),jac='3-point',x_scale='jac',
                          ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=1000)
        fit=fit_fixed_endpoints(times,positions,positions[0],positions[-1],opt.x[:k],opt.x[k:])
        results.append({'index':i,'activation':opt.x[:k],'completion':opt.x[k:],
                        'displacements':fit['displacements'],'positions':fit['positions'],
                        'success':bool(opt.success),'status':int(opt.status),'nfev':int(opt.nfev),
                        'message':str(opt.message),'position_rmse_m':fit['position_rmse_m'],
                        'initial_error_m':fit['initial_error_m'],'final_error_m':fit['final_error_m'],
                        'free_rank':fit['free_rank'],'free_columns':fit['free_columns'],
                        'condition_number':fit['condition_number']})
    eligible=[f for f in results if f['success'] and np.isfinite(f['position_rmse_m'])]
    return {'lower_bounds':lower,'upper_bounds':upper,'runs':results,
            'best':min(eligible,key=lambda f:(f['position_rmse_m'],f['index'])) if eligible else None}
