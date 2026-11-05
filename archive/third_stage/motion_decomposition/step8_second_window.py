import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .initialization import make_initializations, valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv
from .step7_weight_sensitivity import WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def main():
    source = PROJECT_ROOT/'results/wrist_trajectory.npz'
    previous_path = ROOT/'results/step7_weight_sensitivity/summary.json'
    output = ROOT/'results/step8_second_window'
    output.mkdir(parents=True,exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    previous_digest = hashlib.sha256(previous_path.read_bytes()).hexdigest()
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    assert previous['length_scale_m']==.3 and tuple(previous['weights'])==WEIGHTS
    settings = {'component_count':4,'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,
                'seed':23,'jitters_per_seed':4,'jitter_fraction':.2,'overlap_fraction':.25}
    protocol = {'window_s':[30,33],'source_sha256':digest,'weights':WEIGHTS,'settings':settings,
                'reason':'Contains previously inspected 31-33 s movement; prepend one second for the same 3 s duration as the first window; selected before these fits',
                'selection':'Lowest own total objective among eligible results per weight; no cross-weight winner',
                'scope':'Another window of the same clip, not another subject, clip or independent generalization test'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    with np.load(source) as data:
        full_times, full_positions = data['timestamps'].astype(float),data['positions'].astype(float)
    frames = np.flatnonzero((full_times>=30)&(full_times<=33))
    times,original = full_times[frames],full_positions[frames]
    reference_velocity = np.gradient(full_positions,full_times,axis=0,edge_order=2)[frames]
    assert len(times)==361 and np.all(np.diff(frames)==1)
    assert np.allclose(np.diff(times),1/120,atol=1e-12,rtol=0)
    starts,candidates = make_initializations(times,original,4,2/120)
    rows,details,selections = [],[],[]
    selected = {}
    saved = {'timestamps':times,'original':original,'source_frames':frames,
             'reference_velocity_full_clip':reference_velocity}
    for weight in WEIGHTS:
        group,fits = [],[]
        for index,start in enumerate(starts):
            if not start['valid']:
                raise ValueError(start['invalid_reason'])
            fit = fit_joint(times,original,original[0],original[-1],start['activation'],start['completion'],
                            2/120,weight,.3,max_nfev=1000)
            metrics,_ = trajectory_errors(original,fit['positions'],times)
            composition = composition_metrics(times,component_velocities(times,fit['activation'],fit['completion'],fit['displacements']))
            assert np.allclose(original[0]+design_matrix(times,fit['activation'],fit['completion'])@fit['displacements'],fit['positions'],atol=1e-9,rtol=0)
            assert fit['initial_error_m']<1e-12 and fit['final_error_m']<1e-12
            assert fit['objective_m2']<=fit['initial_objective_m2']+1e-10
            assert np.all(fit['activation']>=times[0]) and np.all(fit['completion']<=times[-1])
            assert np.all(fit['completion']-fit['activation']>=2/120-1e-12)
            row = {'weight':weight,'index':index,'initialization':start['name'],'eligible':valid_timing_result(fit),
                   **metrics,**{k:v for k,v in fit.items() if k not in ('positions','activation','completion','displacements')},
                   'relative_component_length_excess':composition['relative_component_length_excess']}
            rows.append(row)
            group.append(row)
            fits.append(fit)
            details.append({**row,'initialization_details':start,'activation':fit['activation'],
                            'completion':fit['completion'],'displacements':fit['displacements']})
            prefix = f'w{weight}_run{index}'
            for key in ('activation','completion','displacements','positions'):
                saved[f'{prefix}_{key}'] = fit[key]
            write_csv(output/'runs.csv',serializable(rows))
            print(weight,index,'eligible',row['eligible'],'RMSE cm',100*fit['position_rmse_m'],
                  'sum |d|',fit['sum_displacement_norms_m'],flush=True)
        eligible = [i for i,row in enumerate(group) if row['eligible']]
        best = min(eligible,key=lambda i:(group[i]['objective_m2'],i)) if eligible else None
        selections.append({'weight':weight,'eligible_count':len(eligible),'best_index':best,
                           'selected':group[best] if best is not None else None,
                           'all_position_rmse_range_m':[min(r['position_rmse_m'] for r in group),max(r['position_rmse_m'] for r in group)]})
        if best is not None:
            selected[weight] = fits[best]
    summary = {**protocol,'previous_summary_sha256':previous_digest,'samples':len(times),
               'source_frame_indices_inclusive':[int(frames[0]),int(frames[-1])],
               'coordinate_ranges_m':np.ptp(original,axis=0),
               'reference_endpoint_speeds_m_s':np.linalg.norm(reference_velocity[[0,-1]],axis=1),
               'raw_speed_range_m_s':[float(np.linalg.norm(reference_velocity,axis=1).min()),float(np.linalg.norm(reference_velocity,axis=1).max())],
               'speed_seed_candidates':candidates,'selections':selections,'details':details,
               'first_window_selections':previous['selections'],
               'limitations':['Two windows from one clip only','Fixed K and numerical settings not physiological thresholds',
                              'No automatic weight or count selection','No movement-unit ground truth',
                              'Endpoint velocity mismatch and finite-budget local optimization remain']}
    (output/'summary.json').write_text(json.dumps(serializable(summary),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(output/'fits.npz',**saved)
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    for row,entries,label in [(0,previous['selections'],'First window: 18-21 s'),(1,selections,'Second window: 30-33 s')]:
        for axis,key,scale,title in zip(axes[row],('position_rmse_m','velocity_rmse_m_s','sum_displacement_norms_m'),
                (100,1,1),('Position RMSE (cm)','Velocity RMSE (m/s)','Sum displacement norms (m, log)')):
            y = [scale*s['selected'][key] if s['selected'] is not None else np.nan for s in entries]
            axis.plot(range(len(WEIGHTS)),y,'o-')
            axis.set_xticks(range(len(WEIGHTS)),[f'{w:g}' for w in WEIGHTS],rotation=40)
            axis.set_title(label+'\n'+title,fontsize=10)
            axis.grid(alpha=.2)
        axes[row,2].set_yscale('log')
    for axis in axes[-1]:
        axis.set_xlabel('Weight (categorical spacing)')
    fig.tight_layout()
    fig.savefig(output/'two_window_tradeoffs.png',dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
    curves = [('Original',original,'black')]+[(f'lambda={w:g}',selected[w]['positions'],color)
              for w,color in [(0,'tab:blue'),(1e-6,'tab:orange'),(1e-4,'tab:green')] if w in selected]
    for label,curve,color in curves:
        velocity = np.gradient(curve,times,axis=0,edge_order=2)
        for c in range(3):
            axes[c,0].plot(times,curve[:,c],color=color,label=label,linewidth=1.1)
            axes[c,1].plot(times,velocity[:,c],color=color,label=label,linewidth=1)
    for c in range(3):
        axes[c,0].set_ylabel(f'{"XYZ"[c]} position (m)')
        axes[c,1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Original clip time (s)')
    fig.suptitle('Second window: same K=4, settings and three prespecified displayed weights')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'position_velocity_comparison.png',dpi=160)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    assert hashlib.sha256(previous_path.read_bytes()).hexdigest()==previous_digest
    for entry in selections:
        print('SELECTED',entry,flush=True)


if __name__=='__main__':
    main()
