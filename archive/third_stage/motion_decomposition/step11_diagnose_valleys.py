import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .goal_count_selection import detection_speed,speed_valley_knots,valley_initializations
from .initialization import make_initializations,valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step1_known_motion import generate_motion
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT=Path(__file__).resolve().parents[1]
WEIGHT=1e-6


def synthetic_cases():
    full_times=np.arange(-60,421)/120
    mask=(full_times>=0)&(full_times<=3)
    cases=[]
    for offset in (1.1,.75,.45,.2,0):
        a=np.array([.25,.25+offset])
        b=a+.9
        d=np.array([[.18,.04,0],[.14,.03,0]])
        clean=generate_motion(full_times,np.zeros(3),d,a,b)
        cases.append({'name':f'two_offset_{offset:g}','category':'two_components',
                      'activation':a,'completion':b,'displacements':d,
                      'clean_full':clean,'observed_full':clean.copy(),
                      'generating_count':2,'count_interpretation':'Generating count, not guaranteed uniquely identifiable',
                      'overlap_fraction':max(0,1-offset/.9),'perturbation_amplitude_m':0.,'perturbation_hz':0.})
    a,b,d=np.array([.25]),np.array([2.75]),np.array([[.3,.1,.05]])
    clean=generate_motion(full_times,np.zeros(3),d,a,b)
    for frequency,amplitude in [(0,0)]+[(f,am) for f in (3,6) for am in (.0005,.001,.002)]:
        observed=clean.copy()
        perturbation=np.zeros(len(full_times))
        perturbation[mask]=amplitude*np.sin(2*np.pi*frequency*full_times[mask])*np.sin(np.pi*full_times[mask]/3)**2
        observed[:,0]+=perturbation
        cases.append({'name':f'one_wiggle_{frequency:g}hz_{amplitude:g}m','category':'single_with_perturbation',
                      'activation':a.copy(),'completion':b.copy(),'displacements':d.copy(),
                      'clean_full':clean.copy(),'observed_full':observed,
                      'generating_count':1,'count_interpretation':'Count of clean base only; perturbed observations are not exactly one minimum-jerk component',
                      'overlap_fraction':0.,'perturbation_amplitude_m':amplitude,'perturbation_hz':frequency})
    assert len(cases)==12 and mask.sum()==361
    return full_times,mask,cases


def fit_group(times,observed,starts):
    fits=[]
    identical={}
    for start in starts:
        signature=(np.asarray(start['activation']).tobytes(),np.asarray(start['completion']).tobytes())
        if signature in identical:
            fit=identical[signature]
            origin='identical_initialization'
        else:
            fit=fit_joint(times,observed,observed[0],observed[-1],start['activation'],start['completion'],2/120,WEIGHT,.3,max_nfev=1000)
            identical[signature]=fit
            origin='new_solve'
        assert np.allclose(observed[0]+design_matrix(times,fit['activation'],fit['completion'])@fit['displacements'],fit['positions'],atol=1e-9,rtol=0)
        assert fit['initial_error_m']<1e-12 and fit['final_error_m']<1e-12
        assert np.all(fit['activation']>=times[0]) and np.all(fit['completion']<=times[-1])
        assert np.all(fit['completion']-fit['activation']>=2/120-1e-12)
        assert fit['objective_m2']<=fit['initial_objective_m2']+1e-10
        fits.append((fit,origin))
    eligible=[i for i,(fit,_) in enumerate(fits) if valid_timing_result(fit)]
    best=min(eligible,key=lambda i:(fits[i][0]['objective_m2'],i)) if eligible else None
    return fits,best


def main():
    output=ROOT/'results/step11_valley_diagnostic'
    output.mkdir(parents=True,exist_ok=True)
    prior_paths=[ROOT/'results/step10_count_selection/summary.json',ROOT/'results/step9_count_comparison/summary.json',ROOT/'results/step9_count_comparison/fits.npz']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prior_paths}
    full_times,mask,cases=synthetic_cases()
    protocol={'scope':'Fixed-B diagnostics, not parameter tuning or a new count selector',
              'detector':{'savgol_window':17,'polyorder':3,'relative_prominence':.1,'minimum_peak_gap_s':.15},
              'fit':{'weight':WEIGHT,'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,'seed':23},
              'synthetic_cases':[{k:v for k,v in case.items() if not k.endswith('_full')} for case in cases],
              'real_probe':'Six merged-interval starts from saved K=4 fit, still fit K=3; diagnostic access to higher-K solution, not fair method score',
              'source_hashes':hashes,'limitations':['Selected synthetic cases, not an estimated general success rate',
                  'Generating count may not be identifiable under overlap','Perturbations are controlled additions, not a validated noise model']}
    (output/'protocol.json').write_text(json.dumps(serializable(protocol),indent=2)+'\n',encoding='utf-8')
    times=full_times[mask]
    rows,runs,details=[],[],[]
    saved={'timestamps':times}
    for case in cases:
        name=case['name']
        clean,observed=case['clean_full'][mask],case['observed_full'][mask]
        speed=detection_speed(full_times,case['observed_full'],17)[mask]
        detected=speed_valley_knots(times,speed,.1,.15)
        generating_count=case['generating_count']
        common,_=make_initializations(times,observed,generating_count,2/120)
        groups=[('B_selected_count',valley_initializations(detected['knots'])),('known_base_count_control',common)]
        row={'case':name,'category':case['category'],'generating_count':generating_count,
             'detected_count':detected['component_count'],'count_interpretation':case['count_interpretation'],
             'overlap_fraction':case['overlap_fraction'],'perturbation_amplitude_m':case['perturbation_amplitude_m'],
             'perturbation_hz':case['perturbation_hz'],'boundary_times_s':detected['knots'],
             'clean_observed_rmse_m':trajectory_errors(clean,observed,times)[0]['position_rmse_m']}
        saved[f'{name}_observed'],saved[f'{name}_clean'],saved[f'{name}_speed']=observed,clean,speed
        for method,starts in groups:
            fits,best=fit_group(times,observed,starts)
            for index,(fit,origin) in enumerate(fits):
                observed_metrics,_=trajectory_errors(observed,fit['positions'],times)
                clean_metrics,_=trajectory_errors(clean,fit['positions'],times)
                record={'case':name,'method':method,'index':index,'initialization':starts[index]['name'],
                        'origin':origin,'eligible':valid_timing_result(fit),**observed_metrics,
                        'clean_position_rmse_m':clean_metrics['position_rmse_m'],
                        **{k:v for k,v in fit.items() if k not in ('positions','activation','completion','displacements','position_rmse_m')}}
                runs.append(record)
                details.append({**record,'activation':fit['activation'],'completion':fit['completion'],
                                'displacements':fit['displacements'],'initialization_details':starts[index]})
                saved[f'{name}_{method}_run{index}']=fit['positions']
            row[method]={'best_index':best,'eligible_count':sum(valid_timing_result(fit) for fit,_ in fits),
                         'selected':next(r for r in runs if r['case']==name and r['method']==method and r['index']==best) if best is not None else None}
        rows.append(row)
        (output/'progress.json').write_text(json.dumps(serializable(rows),indent=2)+'\n',encoding='utf-8')
        print(name,'base K',generating_count,'B K',detected['component_count'],
              'B RMSE cm',100*row['B_selected_count']['selected']['position_rmse_m'] if row['B_selected_count']['selected'] else None,flush=True)
    previous=json.loads(prior_paths[1].read_text(encoding='utf-8'))
    b_summary=json.loads(prior_paths[0].read_text(encoding='utf-8'))
    real_b=next(r for r in b_summary['comparisons'] if r['window']=='30_33' and r['method']=='B_valleys' and r['weight']==WEIGHT)
    selection4=next(r for r in previous['selections'] if r['window']=='30_33' and r['component_count']==4 and r['weight']==WEIGHT)
    with np.load(prior_paths[2]) as data:
        real_times,real_positions=data['w30_33_timestamps'].copy(),data['w30_33_original'].copy()
        prefix=f'w30_33_k4_lambda1e-06_run{selection4["best_index"]}'
        activation,completion=data[f'{prefix}_activation'].copy(),data[f'{prefix}_completion'].copy()
        reference4=data[f'{prefix}_positions'].copy()
    starts=[]
    for i,j in combinations(range(4),2):
        keep=[k for k in range(4) if k not in (i,j)]
        a=np.array([*activation[keep],min(activation[i],activation[j])])
        b=np.array([*completion[keep],max(completion[i],completion[j])])
        order=np.argsort(a,kind='stable')
        starts.append({'name':f'merge_intervals_{i+1}_{j+1}','activation':a[order],'completion':b[order]})
    probes,best_probe=fit_group(real_times,real_positions,starts)
    real_rows=[]
    for index,(fit,origin) in enumerate(probes):
        metrics,_=trajectory_errors(real_positions,fit['positions'],real_times)
        real_rows.append({'index':index,'initialization':starts[index]['name'],'eligible':valid_timing_result(fit),**metrics,
                          **{k:v for k,v in fit.items() if k not in ('positions','activation','completion','displacements','position_rmse_m')}})
        saved[f'real_probe_{index}']=fit['positions']
        details.append({'case':'real_second_window_probe','method':'higher_K_informed_diagnostic',**real_rows[-1],
                        'activation':fit['activation'],'completion':fit['completion'],'displacements':fit['displacements'],
                        'initialization_details':starts[index]})
    saved['real_timestamps'],saved['real_original'],saved['real_reference4']=real_times,real_positions,reference4
    result={**protocol,'synthetic_results':rows,'synthetic_runs':runs,'details':details,
            'real_diagnostic':{'previous_B_K3':real_b,'previous_K4':selection4['selected'],
                               'probes':real_rows,'best_probe_index':best_probe,
                               'selection':'Lowest objective eligible among six probes; no changes to original method score',
                               'inference_limit':'Finding a better K3 solution demonstrates search dependence; not finding one does not prove K3 insufficient globally'}}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(output/'synthetic_runs.csv',serializable(runs))
    write_csv(output/'real_probes.csv',serializable(real_rows))
    np.savez_compressed(output/'fits.npz',**saved)
    plot_results(output,result,saved)
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    print('REAL DIAGNOSTIC',result['real_diagnostic'],flush=True)


def plot_results(output,result,saved):
    fig,axes=plt.subplots(6,2,figsize=(13,17),sharex=True)
    times=saved['timestamps']
    for axis,case in zip(axes.flat,result['synthetic_results']):
        name=case['case']
        axis.plot(times,saved[f'{name}_speed'],color='black')
        detection=speed_valley_knots(times,saved[f'{name}_speed'],.1,.15)
        peaks=detection['peak_frames']
        axis.scatter(times[peaks],saved[f'{name}_speed'][peaks],color='tab:green',s=15)
        for knot in detection['knots'][1:-1]:
            axis.axvline(knot,color='tab:red',linestyle=':',alpha=.6)
        axis.set_title(name.replace('_',' ')+f'\nbase K={case["generating_count"]}, B K={case["detected_count"]}',fontsize=9)
        axis.set_ylabel('Detection speed (m/s)')
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(output/'synthetic_detection.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    x=np.arange(len(result['synthetic_results']))
    axes[0].plot(x,[r['generating_count'] for r in result['synthetic_results']],'o-',label='Clean generating count')
    axes[0].plot(x,[r['detected_count'] for r in result['synthetic_results']],'o-',label='B detected count')
    for method,color in [('B_selected_count','tab:orange'),('known_base_count_control','tab:blue')]:
        y=[100*r[method]['selected']['position_rmse_m'] if r[method]['selected'] else np.nan for r in result['synthetic_results']]
        axes[1].plot(x,y,'o-',color=color,label=method)
    for axis in axes:
        axis.set_xticks(x,[str(i) for i in x])
        axis.set_xlabel('Case index; names in summary.json and detection figure')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel('Component count; perturbed case base is not exact truth')
    axes[1].set_ylabel('Observed position RMSE (cm)')
    fig.tight_layout()
    fig.savefig(output/'synthetic_counts_and_fit.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    times=saved['real_timestamps']
    baseline=result['real_diagnostic']['previous_B_K3']
    source=ROOT/'results/step10_count_selection/fits.npz'
    with np.load(source) as data:
        previous=data[f'30_33_B_valleys_w1e-06_run{baseline["index"]}_positions'].copy()
    curves=[('Original',saved['real_original'],'black'),('Original B K=3',previous,'tab:orange'),
            ('Saved K=4 reference',saved['real_reference4'],'tab:green')]
    best=result['real_diagnostic']['best_probe_index']
    if best is not None:
        curves.append(('Best K=3 diagnostic probe',saved[f'real_probe_{best}'],'tab:blue'))
    for name,p,color in curves:
        for c,axis in enumerate(axes):
            axis.plot(times,p[:,c],color=color,label=name,linewidth=1.1)
    for c,axis in enumerate(axes):
        axis.set_ylabel(f'{"XYZ"[c]} position (m)')
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Original clip time (s)')
    fig.suptitle('Same K=3, extra higher-K-informed starts: diagnostic only, not fair method score')
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/'real_search_diagnostic.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
