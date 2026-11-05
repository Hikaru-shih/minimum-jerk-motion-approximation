from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from .fixed_times import design_matrix, fit_fixed_endpoints
from .goal_count_selection import chord_knee, detection_speed, speed_valley_knots
from .initialization import make_initializations
from .joint_fit import fit_joint
from .stability import compare_aligned
from .step1_known_motion import generate_motion
from .step14_extend_counts import fit_task, SETTINGS
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv
from .opposition import opposition_residuals

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUTPUT = ROOT/'results/movement_unit_validation'
FIELDS = ('activation','completion','displacements','positions')


def cases():
    t = np.arange(361)/120
    d2 = np.array([[.30,.10,-.05],[-.12,.20,.08]])
    definitions = [
        ('single_clean',[.25],[2.75],[[.30,.10,.05]],0.,'Exact generating count; finite recovery experiment, not a uniqueness theorem'),
        ('two_separated',[.25,1.70],[1.15,2.60],d2,0.,'Two separated nonparallel components'),
        ('two_overlapping',[.25,.70],[1.15,1.60],d2,0.,'Same displacement vectors and durations, overlapping with opposing contributions'),
        ('single_perturbed',[.25],[2.75],[[.30,.10,.05]],.001,'Clean-base count only; perturbed observed signal has no asserted exact generating count')]
    result = []
    for name,a,b,d,amplitude,note in definitions:
        a,b,d = np.asarray(a),np.asarray(b),np.asarray(d)
        clean = generate_motion(t,np.zeros(3),d,a,b)
        observed = clean.copy()
        observed[:,0] += amplitude*np.sin(2*np.pi*6*t)*np.sin(np.pi*t/3)**2
        result.append({'case':name,'activation':a,'completion':b,'displacements':d,
                       'clean':clean,'observed':observed,'base_count':len(a),'perturbation_amplitude_m':amplitude,
                       'interpretation':note})
    return t,result


def analytical_checks(t,definitions):
    case = next(c for c in definitions if c['case']=='two_overlapping')
    a,b,d = case['activation'],case['completion'],case['displacements']
    def penalty(a,b,d):
        q = opposition_residuals(t,a,b,d,.3)
        return float(q@q/len(t))
    original = case['clean']
    base = penalty(a,b,d)
    assert base>0
    examples=[]
    for copies in (1,2,4):
        split_a,split_b = np.repeat(a,copies),np.repeat(b,copies)
        split_d = np.repeat(d/copies,copies,axis=0)
        curve = design_matrix(t,split_a,split_b)@split_d
        np.testing.assert_allclose(curve,original,atol=1e-14)
        value = penalty(split_a,split_b,split_d)
        np.testing.assert_allclose(value,base/copies**2,atol=1e-14)
        examples.append({'copies_per_component':copies,'K':len(split_a),
                         'max_trajectory_difference_m':float(np.max(np.abs(curve-original))),
                         'opposition_cost_m2':value,'objective_on_clean_target_m2':1e-6*value})
    a = np.array([.25,.25]); b = np.array([2.75,2.75])
    two = generate_motion(t,np.zeros(3),d,a,b)
    one = generate_motion(t,np.zeros(3),d.sum(axis=0,keepdims=True),a[:1],b[:1])
    np.testing.assert_allclose(two,one,atol=1e-14)
    return {'single_count_limit':'Current chord_knee selects interior counts only; K=1 cannot be selected from 1..12',
            'split_equivalence':examples,
            'coincident_two_vs_one_max_difference_m':float(np.max(np.abs(two-one))),
            'interpretation':'Algebraic invariance check, not fitted evidence; exact duplicate splitting reduces the current pairwise squared penalty while preserving the trajectory'}


def context_review():
    with np.load(PROJECT/'results/wrist_trajectory.npz') as z:
        t,p = z['timestamps'],z['positions'].astype(float)
    raw_v = np.gradient(p,t,axis=0,edge_order=2)
    speeds = {w:detection_speed(t,p,w) for w in (9,17,25)}
    smooth_v = savgol_filter(p,17,3,deriv=1,delta=1/120,axis=0,mode='interp')
    rows=[]
    for start,end in ((18,21),(30,33)):
        mask = (t>=start-3)&(t<=end+3)
        cropped = (t>=start)&(t<=end)
        cut_v = np.gradient(p[cropped],t[cropped],axis=0,edge_order=2)
        boundaries = [int(np.argmin(np.abs(t-boundary))) for boundary in (start,end)]
        detections = {w:serializable(speed_valley_knots(t[mask],speeds[w][mask],.1,.15)) for w in (9,17,25)}
        rows.append({'window':f'{start}_{end}','context':[start-3,end+3],
                     'raw_speed_at_boundaries_full_difference_m_s':[float(np.linalg.norm(raw_v[i])) for i in boundaries],
                     'raw_speed_at_boundaries_cropped_difference_m_s':np.linalg.norm(cut_v[[0,-1]],axis=1),
                     'smoothed_speed_at_boundaries_m_s':{w:[float(speeds[w][i]) for i in boundaries] for w in speeds},
                     'context_speed_candidates':detections,
                     'limit':'Speed minima are candidate cues, not confirmed movement boundaries; nonzero speed alone does not identify boundary error or noise'})
        fig,axes = plt.subplots(3,1,figsize=(11,8),sharex=True)
        axes[0].plot(t[mask],np.linalg.norm(raw_v[mask],axis=1),color='0.75',linewidth=.7,label='Raw full-clip difference')
        for w in speeds: axes[0].plot(t[mask],speeds[w][mask],label=f'Detection window {w}')
        for c in range(3):
            axes[1].plot(t[mask],smooth_v[mask,c],label='XYZ'[c])
            axes[2].plot(t[mask],p[mask,c]-p[mask][0,c],label='XYZ'[c])
        for axis in axes:
            axis.axvspan(start,end,color='tab:orange',alpha=.12)
            axis.axvline(start,color='black',linestyle=':'); axis.axvline(end,color='black',linestyle=':')
            axis.grid(alpha=.2); axis.legend(fontsize=8)
        axes[0].set_ylabel('Speed (m/s)'); axes[1].set_ylabel('Smoothed XYZ velocity (m/s)')
        axes[2].set_ylabel('XYZ relative position (m)'); axes[2].set_xlabel('Full clip time (s)')
        fig.suptitle('Raw motion context; orange band is analyzed crop, not an inferred movement unit')
        fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(OUTPUT/f'context_{start}_{end}.png',dpi=150); plt.close(fig)
    return rows


def main(workers):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    t,definitions = cases()
    inputs = [PROJECT/'results/wrist_trajectory.npz']+[Path(__file__).parent/name for name in
              ('validate_movement_structure.py','joint_fit.py','fixed_times.py','timing_fit.py','initialization.py',
               'opposition.py','goal_count_selection.py','step14_extend_counts.py','step1_known_motion.py')]
    hashes = {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    protocol = {'scope':'Fixed-method structural validation, no parameter tuning or new selector',
                'cases':[{key:value for key,value in c.items() if key not in ('clean','observed')} for c in definitions],
                'counts':list(range(1,13)),'weight':1e-6,'settings':SETTINGS,'planned_records':480,
                'workers':workers,'source_hashes':hashes,
                'selection':'Same current total-objective interior chord knee, upper K=6,8,10,12; no monotone envelope or endpoint fallback',
                'recovery':'Report continuous matched parameter errors at known base K; do not assign a success tolerance after seeing results',
                'oracle_limit':'True-timing diagnostic starts are separate and excluded from primary search pool',
                'perturbed_limit':'Base K=1 is not asserted as the exact count of the perturbed signal',
                'ambiguous_limit':'Identical timing components can merge algebraically; no count-recovery demand on that example'}
    protocol_path=OUTPUT/'protocol.json'
    if protocol_path.exists():
        previous=json.loads(protocol_path.read_text())
        assert all(previous[key]==serializable(protocol[key]) for key in ('cases','counts','settings','source_hashes'))
    else: protocol_path.write_text(json.dumps(serializable(protocol),indent=2)+'\n')
    algebra=analytical_checks(t,definitions)
    context=context_review()
    (OUTPUT/'preflight.json').write_text(json.dumps(serializable({'analytical_checks':algebra,'real_context':context}),indent=2)+'\n')
    print('PREFLIGHT',serializable(algebra),flush=True)
    checkpoint=OUTPUT/'checkpoints'; checkpoint.mkdir(exist_ok=True)
    rows=[]; saved={'timestamps':t}; tasks=[]
    for case in definitions:
        name,p=case['case'],case['observed']
        saved[name+'_observed'],saved[name+'_clean']=p,case['clean']
        for k in range(1,13):
            starts,_=make_initializations(t,p,k,2/120)
            indices=[0] if k==1 else range(10)
            for i in indices:
                prefix=f'{name}_k{k}_run{i}'
                path=checkpoint/(prefix+'.json')
                if path.exists() and path.with_suffix('.npz').exists():
                    row=json.loads(path.read_text()); rows.append(row)
                    with np.load(path.with_suffix('.npz')) as z: saved.update({prefix+'_'+field:z[field].copy() for field in FIELDS})
                else: tasks.append((name,k,1e-6,i,starts[i],t,p))
    print('Pending distinct fits',len(tasks),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(fit_task,task):task[:4] for task in tasks}
        for future in as_completed(futures):
            row,fit=future.result()
            name,k,weight,i=futures[future]
            row['origin']='structural_validation_solve'
            prefix=f'{name}_k{k}_run{i}'
            np.savez_compressed(checkpoint/(prefix+'.npz'),**fit)
            (checkpoint/(prefix+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
            rows.append(row); saved.update({prefix+'_'+field:value for field,value in fit.items()})
            print('DONE',len(rows),'/444',name,k,i,'eligible',row['eligible'],'cm',round(100*row['position_rmse_m'],7),flush=True)
            (OUTPUT/'progress.json').write_text(json.dumps({'completed_distinct_fits':len(rows),'planned_distinct_fits':444,'last':row},indent=2)+'\n')
    for case in definitions:
        base=next(r for r in rows if r['window']==case['case'] and r['component_count']==1)
        starts,_=make_initializations(t,case['observed'],1,2/120)
        for i in range(1,10):
            row={**base,'index':i,'origin':'identical_K1_reuse','initialization':starts[i]['name'],'initialization_details':serializable(starts[i])}
            rows.append(row)
            for field in FIELDS: saved[f'{case["case"]}_k1_run{i}_{field}']=saved[f'{case["case"]}_k1_run0_{field}']
    selections=[]; results=[]; oracle=[]
    for case in definitions:
        name=case['case']; p=case['observed']
        for k in range(1,13):
            group=[r for r in rows if r['window']==name and r['component_count']==k]
            eligible=[r for r in group if r['eligible']]
            best=min(eligible,key=lambda r:(r['objective_m2'],r['index'])) if eligible else None
            selections.append({'case':name,'K':k,'eligible_count':len(eligible),'selected':best})
        own=[r for r in selections if r['case']==name]
        chosen=[]
        for upper in (6,8,10,12):
            subset=own[:upper]
            knee=chord_knee(range(1,upper+1),[r['selected']['objective_m2'] for r in subset]) if all(r['selected'] for r in subset) else {'selected_count':None,'reason':'Missing eligible fit'}
            chosen.append({'upper_K':upper,**knee})
        true={'activation':case['activation'],'completion':case['completion'],'displacements':case['displacements'],'positions':case['clean']}
        at_base=own[case['base_count']-1]['selected']
        recovery=None
        if at_base:
            prefix=f'{name}_k{case["base_count"]}_run{at_base["index"]}'
            recovery=compare_aligned(t,true,{field:saved[prefix+'_'+field] for field in FIELDS})
        fixed=fit_fixed_endpoints(t,case['clean'],case['clean'][0],case['clean'][-1],case['activation'],case['completion'])
        np.testing.assert_allclose(fixed['positions'],case['clean'],atol=1e-12)
        diagnostic=fit_joint(t,p,p[0],p[-1],case['activation'],case['completion'],2/120,1e-6,.3,max_nfev=1000)
        oracle.append({'case':name,'fixed_true_time_clean_LS_rmse_m':fixed['position_rmse_m'],
                       'true_time_initialized_joint_fit':{key:value for key,value in diagnostic.items() if key!='positions'},
                       'recovery':compare_aligned(t,true,diagnostic)})
        saved[name+'_oracle_positions']=diagnostic['positions']
        results.append({'case':name,'base_count':case['base_count'],'count_interpretation':case['interpretation'],
                        'selections_by_upper':chosen,'best_at_base_count':at_base,'base_count_parameter_recovery':recovery})
    result={**protocol,'analytical_checks':algebra,'real_context':context,'case_results':results,
            'selections':selections,'details':sorted(rows,key=lambda r:(r['window'],r['component_count'],r['index'])),
            'oracle_diagnostics':oracle,'distinct_primary_solves':444,'identical_K1_reuses':36}
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
    columns=('window','component_count','index','origin','eligible','position_rmse_m','velocity_rmse_m_s','objective_m2','opposition_cost_m2','status','nfev')
    write_csv(OUTPUT/'runs.csv',[{key:r[key] for key in columns} for r in result['details']])
    np.savez_compressed(OUTPUT/'fits.npz',**saved)
    plot_results(result,saved)
    for name,digest in hashes.items(): assert hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()==digest
    print('COMPLETE',serializable(results),flush=True)


def plot_results(result,saved):
    fig,axes=plt.subplots(2,4,figsize=(15,7))
    t=saved['timestamps']
    for col,case in enumerate(result['case_results']):
        name=case['case']; group=[r for r in result['selections'] if r['case']==name]
        axes[0,col].plot(range(1,13),[r['selected']['objective_m2'] if r['selected'] else np.nan for r in group],'o-',markersize=3)
        axes[0,col].set_title(name.replace('_',' '),fontsize=10)
        axes[0,col].set_xlabel('K'); axes[0,col].set_ylabel('Total fitted objective (m2)')
        axes[0,col].set_xticks((1,2,4,8,12)); axes[0,col].grid(alpha=.2)
        observed=saved[name+'_observed']; clean=saved[name+'_clean']
        axes[1,col].plot(t,np.linalg.norm(np.gradient(observed,t,axis=0,edge_order=2),axis=1),color='0.65',label='Observed speed')
        selected=case['selections_by_upper'][-1]['selected_count']
        if selected is not None:
            r=group[selected-1]['selected']; curve=saved[f'{name}_k{selected}_run{r["index"]}_positions']
            axes[1,col].plot(t,np.linalg.norm(np.gradient(curve,t,axis=0,edge_order=2),axis=1),label=f'Selected K={selected}')
        else:
            axes[1,col].text(.03,.93,'No count selected',transform=axes[1,col].transAxes,va='top',fontsize=9)
        axes[1,col].set_ylabel('Speed (m/s)'); axes[1,col].set_xlabel('Time (s)'); axes[1,col].grid(alpha=.2); axes[1,col].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT/'synthetic_pipeline.png',dpi=150); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1: parser.error('workers must be positive')
    main(args.workers)
