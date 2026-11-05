import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .step4_diagnose_initialization import component_velocities
from .step5_fit_real_window import serializable

ROOT = Path(__file__).resolve().parents[1]
COLORS = ('tab:blue','tab:orange','tab:green','tab:red')


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT/'results/step6_joint_real'
    output = ROOT/'results/step6_real_components'
    output.mkdir(parents=True,exist_ok=True)
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in ('summary.json','fits.npz')}
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    selection = next(s for s in summary['selections'] if s['component_count'] == 4 and s['weight'] == 1e-6)
    index = selection['best_index']
    prefix = f'k4_w1e-06_run{index}'
    with np.load(source/'fits.npz') as data:
        times, original = data['timestamps'].copy(), data['original'].copy()
        a,b,d,fitted = [data[f'{prefix}_{key}'].copy() for key in ('activation','completion','displacements','positions')]
    basis = design_matrix(times,a,b)
    contributions = basis[:,:,None]*d[None,:,:]
    velocity = component_velocities(times,a,b,d)
    assert np.allclose(original[0]+contributions.sum(axis=1),fitted,atol=1e-12,rtol=0)
    goals = original[0]+np.cumsum(d,axis=0)
    residual = fitted-original
    error = np.linalg.norm(residual,axis=1)
    peak = int(np.argmax(error))
    components = [{'component':k+1,'activation_s':a[k],'completion_s':b[k],
                   'duration_s':b[k]-a[k],'dx_m':d[k,0],'dy_m':d[k,1],'dz_m':d[k,2],
                   'displacement_norm_m':np.linalg.norm(d[k]),
                   'goal_x_m':goals[k,0],'goal_y_m':goals[k,1],'goal_z_m':goals[k,2],
                   'nearest_sampled_reconstructed_goal_distance_m':np.min(np.linalg.norm(fitted-goals[k],axis=1))}
                  for k in range(4)]
    pairs, pair_curves = [], []
    for i in range(4):
        for j in range(i+1,4):
            dot = np.sum(velocity[:,i]*velocity[:,j],axis=1)
            q = np.maximum(0,-dot)
            pair_cost = np.sum(((times[-1]-times[0])**2/.3*q)**2)/len(times)
            pairs.append({'component_i':i+1,'component_j':j+1,
                          'direction_angle_degrees':np.degrees(np.arccos(np.clip(np.dot(d[i],d[j])/(np.linalg.norm(d[i])*np.linalg.norm(d[j])),-1,1))),
                          'overlap_duration_s':max(0,min(b[i],b[j])-max(a[i],a[j])),
                          'opposing_sample_count':int(np.sum(dot<0)),
                          'unweighted_opposition_cost_m2':pair_cost})
            pair_curves.append((i,j,q))
    assert np.isclose(sum(p['unweighted_opposition_cost_m2'] for p in pairs),selection['selected']['opposition_cost_m2'],rtol=1e-10,atol=1e-12)
    write_csv(output/'components.csv',serializable(components))
    write_csv(output/'pairs.csv',serializable(pairs))
    fig,axes = plt.subplots(4,2,figsize=(12,10),sharex=True)
    for c in range(3):
        axes[c,0].plot(times,original[:,c]-original[0,c],color='gray',label='Original minus start')
        for k,color in enumerate(COLORS):
            axes[c,0].plot(times,contributions[:,k,c],color=color,label=f'Component {k+1}')
        axes[c,0].plot(times,fitted[:,c]-original[0,c],'--',color='black',label='Sum')
        axes[c,0].set_ylabel(f'{"XYZ"[c]} displacement (m)')
        axes[c,1].plot(times,100*residual[:,c],color='tab:red')
        axes[c,1].axhline(0,color='gray',linewidth=.6)
        axes[c,1].set_ylabel(f'{"XYZ"[c]} residual (cm)')
    for k,color in enumerate(COLORS):
        axes[3,0].plot(times,basis[:,k],color=color,label=f'Progress {k+1}')
    axes[3,0].set_ylabel('Progress (0 to 1)')
    axes[3,0].legend(fontsize=7,ncol=2)
    axes[3,1].plot(times,100*error,color='black')
    axes[3,1].scatter(times[peak],100*error[peak],color='tab:red')
    axes[3,1].set_ylabel('Position error norm (cm)')
    axes[0,0].legend(fontsize=7,ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Saved K=4, lambda=1e-6 solution: additive displacement contributions and residuals')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'contributions_and_errors.png',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(4,1,figsize=(11,10),sharex=True)
    reference_velocity = np.gradient(original,times,axis=0,edge_order=2)
    for c in range(3):
        for k,color in enumerate(COLORS):
            axes[c].plot(times,velocity[:,k,c],color=color,label=f'Component {k+1}')
        axes[c].plot(times,velocity.sum(axis=1)[:,c],'--',color='black',label='Analytical sum')
        axes[c].plot(times,reference_velocity[:,c],color='gray',alpha=.5,label='Original finite difference')
        axes[c].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    for i,j,q in pair_curves:
        if np.any(q>0):
            axes[3].plot(times,q,label=f'Pair {i+1}-{j+1}')
    axes[3].set_ylabel('max(0, -v_i dot v_j)\n(m^2/s^2)')
    axes[3].legend(fontsize=8)
    axes[0].legend(fontsize=7,ncol=3)
    for axis in axes:
        axis.grid(alpha=.2)
    axes[-1].set_xlabel('Original clip time (s)')
    fig.suptitle('Component velocities and simultaneous opposition; diagnostic, not a new fit')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'velocities_and_opposition.png',dpi=160)
    plt.close(fig)
    result = {'source_hashes':hashes,'selected_run_index':index,'weight':1e-6,
              'selection_reason':'Inspect previously selected small-weight solution, not selecting a final weight',
              'components':components,'pairs':pairs,'peak_position_error_time_s':times[peak],
              'peak_position_error_m':error[peak],'peak_residual_xyz_m':residual[peak],
              'unweighted_opposition_cost_m2':sum(p['unweighted_opposition_cost_m2'] for p in pairs),
              'new_optimization_performed':False,
              'goal_distance_note':'Distance to nearest saved reconstructed sample, not proof of continuous-path intersection',
              'velocity_note':'Analytical component sum is diagnostic; published comparison RMSE uses finite differences for both trajectories'}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name,digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    print(json.dumps(serializable(result),indent=2))


if __name__ == '__main__':
    main()
