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
from .stability import compare_aligned
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = (0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4)


def main():
    source = ROOT/'results/step6_joint_real'
    window = ROOT/'results/step5_real_window/window.npz'
    output = ROOT/'results/step7_weight_sensitivity'
    output.mkdir(parents=True,exist_ok=True)
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source/'summary.json',source/'fits.npz',window)}
    previous = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    assert previous['settings'] == {'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,
                                    'seed':23,'jitters_per_seed':4,'jitter_fraction':.2,'overlap_fraction':.25}
    with np.load(window) as data:
        times,original = data['timestamps'].copy(),data['positions'].copy()
    starts,_ = make_initializations(times,original,4,2/120)
    cached = {}
    reference_index = next(s['best_index'] for s in previous['selections'] if s['component_count']==4 and s['weight']==1e-6)
    with np.load(source/'fits.npz') as data:
        for row in previous['details']:
            if row['component_count'] == 4:
                prefix = f"k4_w{row['weight']}_run{row['index']}"
                fit = {key:data[f'{prefix}_{key}'].copy() for key in ('activation','completion','displacements','positions')}
                fit.update({k:v for k,v in row.items() if k not in fit})
                cached[(row['weight'],row['index'])] = fit
    reference = cached[(1e-6,reference_index)]
    rows,details,selections = [],[],[]
    saved = {'timestamps':times,'original':original}
    selected = {}
    for weight in WEIGHTS:
        group,fits = [],[]
        for index,start in enumerate(starts):
            key = (weight,index)
            reused = key in cached
            if reused:
                result = cached[key]
                for field in ('activation','completion'):
                    np.testing.assert_array_equal(start[field],result['initialization_details'][field])
            else:
                result = fit_joint(times,original,original[0],original[-1],start['activation'],start['completion'],
                                   2/120,weight,.3,max_nfev=1000)
            metrics,_ = trajectory_errors(original,result['positions'],times)
            composition = composition_metrics(times,component_velocities(times,result['activation'],result['completion'],result['displacements']))
            assert np.allclose(original[0]+design_matrix(times,result['activation'],result['completion'])@result['displacements'],result['positions'],atol=1e-9,rtol=0)
            assert result['initial_error_m']<1e-12 and result['final_error_m']<1e-12
            assert result['objective_m2']<=result['initial_objective_m2']+1e-10
            compared = compare_aligned(times,reference,result)
            alignment = compared.pop('alignment')
            row = {'weight':weight,'index':index,'initialization':start['name'],'reused_saved_fit':reused,
                   'eligible':valid_timing_result(result),**metrics,
                   'opposition_cost_m2':result['opposition_cost_m2'],'objective_m2':result['objective_m2'],
                   'sum_displacement_norms_m':result['sum_displacement_norms_m'],
                   'relative_component_length_excess':composition['relative_component_length_excess'],
                   'status':result['status'],'message':result['message'],'nfev':result['nfev'],
                   'residual_evaluations':result['residual_evaluations'],'optimality':result['optimality'],**compared}
            rows.append(row)
            group.append(row)
            fits.append(result)
            details.append({**row,'activation':result['activation'],'completion':result['completion'],
                            'displacements':result['displacements'],'alignment_to_reference':alignment})
            prefix=f'w{weight}_run{index}'
            for field in ('activation','completion','displacements','positions'):
                saved[f'{prefix}_{field}']=result[field]
            write_csv(output/'runs.csv',serializable(rows))
            print(weight,index,'reused',reused,'eligible',row['eligible'],'RMSE cm',100*metrics['position_rmse_m'],flush=True)
        eligible=[i for i,r in enumerate(group) if r['eligible']]
        best=min(eligible,key=lambda i:(group[i]['objective_m2'],i)) if eligible else None
        selections.append({'weight':weight,'effective_weight_per_m2':weight/.3**2,'eligible_count':len(eligible),
                           'best_index':best,'selected':group[best] if best is not None else None,
                           'all_position_rmse_range_m':[min(r['position_rmse_m'] for r in group),max(r['position_rmse_m'] for r in group)]})
        if best is not None:
            selected[weight]=fits[best]
    result={'scope':'K=4 same-window weight sensitivity; no final weight selection',
            'source_hashes':hashes,'weights':WEIGHTS,'length_scale_m':.3,
            'scale_identity':'For fixed T, penalty strength depends on lambda/L^2; scaling L by c and lambda by c^2 leaves objective unchanged',
            'settings':previous['settings'],'reuse':'30 existing fits at 0,1e-6,1e-4; 50 new fits at prespecified intermediate weights, same ten starts and no warm starts',
            'reference':{'weight':1e-6,'index':reference_index},
            'selection':'Lowest own total objective among eligible results within each weight, no cross-weight winner',
            'selections':selections,'details':details,
            'limitations':['Fixed arbitrary length scale and one clip window','Local finite-budget optimization',
                           'Not a proof of continuity, uniqueness or physiological interpretation',
                           'No automatic weight or component-count selection']}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(output/'fits.npz',**saved)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    keys=('position_rmse_m','velocity_rmse_m_s','sum_displacement_norms_m','opposition_cost_m2')
    scales=(100,1,1,1)
    labels=('Position RMSE (cm)','Velocity RMSE (m/s)','Sum displacement norms (m, log)','Unweighted opposition cost (m^2, log)')
    for axis,key,scale,label in zip(axes.flat,keys,scales,labels):
        for row in rows:
            axis.scatter(WEIGHTS.index(row['weight']),scale*row[key],color='gray' if row['eligible'] else 'red',alpha=.4,s=18)
        axis.plot(range(len(WEIGHTS)),[scale*s['selected'][key] if s['selected'] else np.nan for s in selections],'o-',color='tab:blue')
        axis.set_xticks(range(len(WEIGHTS)),[f'{w:g}' for w in WEIGHTS],rotation=35)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[1,0].set_yscale('log')
    axes[1,1].set_yscale('log')
    for axis in axes[-1]:
        axis.set_xlabel('Penalty weight (categorical spacing)')
    fig.suptitle('K=4 weight sensitivity: gray eligible, red ineligible; blue selected by own objective')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'weight_tradeoffs.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(12,10),sharex=True)
    for weight,fit in selected.items():
        match=compare_aligned(times,reference,fit)['alignment']
        x=WEIGHTS.index(weight)
        for k,j in enumerate(match):
            axes[k,0].scatter(x,fit['activation'][j],color='tab:blue')
            axes[k,0].scatter(x,fit['completion'][j],color='tab:orange')
            for c,color in enumerate(('tab:blue','tab:orange','tab:green')):
                axes[k,1].scatter(x,100*fit['displacements'][j,c],color=color)
    for k in range(4):
        axes[k,0].set_ylabel(f'C{k+1}: activation/finish (s)')
        axes[k,1].set_ylabel(f'C{k+1}: XYZ displacement (cm)')
    for axis in axes.flat:
        axis.set_xticks(range(len(WEIGHTS)),[f'{w:g}' for w in WEIGHTS],rotation=35)
        axis.grid(alpha=.2)
    fig.suptitle('Objective-selected fits aligned to lambda=1e-6 reference\nLeft blue=start orange=finish; right blue=X orange=Y green=Z')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(output/'selected_parameters.png',dpi=160)
    plt.close(fig)
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for row in selections:
        print('SELECTED',row,flush=True)


if __name__=='__main__':
    main()
