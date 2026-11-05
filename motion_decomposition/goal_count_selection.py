import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .initialization import times_from_knots


def chord_knee(counts, costs):
    counts, costs = np.asarray(counts,dtype=float),np.asarray(costs,dtype=float)
    if counts.ndim!=1 or costs.shape!=counts.shape or len(counts)<3:
        raise ValueError('At least three matching counts and costs required')
    if (not np.isfinite(counts).all() or not np.isfinite(costs).all() or np.any(costs<0)
            or np.any(np.diff(counts)<=0) or np.any(counts<1) or np.any(counts!=np.floor(counts))):
        raise ValueError('Counts must increase, costs must be finite and nonnegative')
    tolerance=1e-12*max(float(np.max(costs)),1e-15)
    if np.any(np.diff(costs)>tolerance):
        return {'selected_count':None,'reason':'Nonmonotone costs; no envelope or smoothing applied'}
    span=costs[0]-costs[-1]
    if span<=tolerance:
        return {'selected_count':None,'reason':'Flat cost curve'}
    x=(counts-counts[0])/(counts[-1]-counts[0])
    y=(costs-costs[-1])/span
    distances=((1-x)-y)/np.sqrt(2)
    best=int(np.argmax(distances[1:-1]))+1
    if distances[best]<=1e-10:
        return {'selected_count':None,'reason':'No positive interior knee','normalized_counts':x,
                'normalized_costs':y,'chord_distances':distances}
    return {'selected_count':int(counts[best]),'reason':'Maximum positive perpendicular distance below endpoint chord; exact tie selects smaller K',
            'normalized_counts':x,'normalized_costs':y,'chord_distances':distances}


def detection_speed(times,positions,window_length):
    times,positions=np.asarray(times,dtype=float),np.asarray(positions,dtype=float)
    if times.ndim!=1 or positions.shape!=(len(times),3) or len(times)<window_length:
        raise ValueError('Expected timestamps and positions (N,3) covering filter window')
    if not np.isfinite(times).all() or not np.isfinite(positions).all() or np.any(np.diff(times)<=0):
        raise ValueError('Finite positions and strictly increasing times required')
    if not isinstance(window_length,int) or window_length<5 or window_length%2!=1:
        raise ValueError('Odd filter window at least five required')
    dt=float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times),dt,atol=1e-10,rtol=1e-9):
        raise ValueError('Uniform sampling required')
    velocity=savgol_filter(positions,window_length=window_length,polyorder=3,deriv=1,delta=dt,axis=0,mode='interp')
    return np.linalg.norm(velocity,axis=1)


def speed_valley_knots(times,speed,relative_prominence=.1,minimum_peak_gap_s=.15):
    times,speed=np.asarray(times,dtype=float),np.asarray(speed,dtype=float)
    if times.ndim!=1 or speed.shape!=times.shape or len(times)<3:
        raise ValueError('Matching times and speed vectors required')
    if not np.isfinite(times).all() or not np.isfinite(speed).all() or np.any(speed<0) or np.any(np.diff(times)<=0):
        raise ValueError('Finite nonnegative speed and increasing timestamps required')
    if not 0<relative_prominence<1 or not np.isfinite(minimum_peak_gap_s) or minimum_peak_gap_s<=0:
        raise ValueError('Prominence in (0,1) and positive peak gap required')
    dt=float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times),dt,atol=1e-10,rtol=1e-9):
        raise ValueError('Uniform sampling required')
    peaks,properties=find_peaks(speed,prominence=relative_prominence*float(speed.max()),
                                distance=max(1,int(np.ceil(minimum_peak_gap_s/dt-1e-9))))
    valleys=[]
    for left,right in zip(peaks[:-1],peaks[1:]):
        valleys.append(int(left+np.argmin(speed[left:right+1])))
    indices=np.array([0,*valleys,len(times)-1],dtype=int)
    if np.any(np.diff(indices)<=0):
        raise ValueError('Detected boundaries must increase strictly')
    return {'knots':times[indices],'boundary_frames':indices,'peak_frames':peaks,
            'peak_prominences':properties['prominences'],'component_count':len(indices)-1,
            'no_peak_fallback':len(peaks)==0,
            'rule':'One speed lobe per retained prominent peak, interior boundaries at lowest sample between adjacent peaks; crop endpoints cover edges; no peaks falls back to K=1'}


def valley_initializations(knots,seed=23):
    knots=np.asarray(knots,dtype=float)
    uniform=np.linspace(knots[0],knots[-1],len(knots))
    rng=np.random.default_rng(seed)
    starts=[]
    for family,base in [('uniform',uniform),('detected_valleys',knots)]:
        gaps=np.diff(base)
        radius=.2*np.minimum(gaps[:-1],gaps[1:])
        for repetition in range(5):
            perturbed=base.copy()
            if repetition:
                perturbed[1:-1]+=rng.uniform(-1,1,len(radius))*radius
            a,b=times_from_knots(perturbed,.25)
            starts.append({'name':f'{family}_{repetition}','activation':a,'completion':b,'knots':perturbed})
    return starts
