import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .goal_count_selection import chord_knee,detection_speed,speed_valley_knots,valley_initializations
from .initialization import make_initializations,valid_timing_result
from .joint_fit import fit_joint
from .metrics import trajectory_errors
from .step4_diagnose_initialization import component_velocities,composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=(0,1e-6)
FILTER_WINDOWS=(9,17,25)
PROMINENCES=(.05,.1,.2)


def main():
    output=ROOT/'results/step10_count_selection'
    output.mkdir(parents=True,exist_ok=True)
    previous_path=ROOT/'results/step9_count_comparison/summary.json'
    arrays_path=previous_path.parent/'fits.npz'
    source=ROOT.parents[1]/'results/wrist_trajectory.npz'
    hashes={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (previous_path,arrays_path,source)}
    protocol={'A':'Maximum positive normalized perpendicular chord distance, primary total fitted objective, K=1..6; no monotonic envelope',
              'A_sensitivity':'Upper K=3,4,5,6; total objective and position MSE; both weights',
              'B':'Prominent speed lobes separated by lowest sample between adjacent retained peaks; endpoints cover crop',
              'B_nominal':{'savgol_window_samples':17,'polyorder':3,'relative_peak_prominence':.1,'minimum_peak_gap_s':.15},
              'B_sensitivity':{'filter_windows_samples':FILTER_WINDOWS,'relative_prominences':PROMINENCES,
                               'note':'Counts only for sensitivity grid; fit nominal candidate setting only'},
              'shared':{'weights':WEIGHTS,'length_scale_m':.3,'minimum_duration_s':2/120,'max_nfev':1000,'seed':23},
              'fairness':'Same raw targets, endpoints and solver. A reuses ten common starts per K; B-native uses five uniform plus five detected-valley starts. B-common uses same common starts as A at B-selected K to isolate initialization effect. Same per-fit cap, not equal total search cost.',
              'sources':hashes,'scope':'Two-window count-selector comparison, not biological truth or validation of universal thresholds'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    previous=json.loads(previous_path.read_text(encoding='utf-8'))
    assert previous['settings']['length_scale_m']==.3 and previous['settings']['max_nfev']==1000
    with np.load(source) as data:
        full_times,full_positions=data['timestamps'].astype(float),data['positions'].astype(float)
    filtered={w:detection_speed(full_times,full_positions,w) for w in FILTER_WINDOWS}
    raw_speed=np.linalg.norm(np.gradient(full_positions,full_times,axis=0,edge_order=2),axis=1)
    with np.load(arrays_path) as data:
        cache={key:data[key].copy() for key in data.files}
    prior_rows={(r['window'],r['component_count'],r['weight'],r['index']):r for r in previous['details']}
    comparisons,knees,sensitivity,runs,fit_details,windows=[],[],[],[],[],[]
    saved={}

    def unpack_prior(label,count,weight,index):
        row=prior_rows[(label,count,weight,index)]
        prefix=f'w{label}_k{count}_lambda{weight}_run{index}'
        result={key:cache[f'{prefix}_{key}'] for key in ('activation','completion','displacements','positions')}
        result.update(row)
        result['success']=row['status']>0
        assert valid_timing_result(result)==row['eligible']
        return result

    def record_fit(label,method,count,weight,index,start,fit,origin):
        times,original=cache[f'w{label}_timestamps'],cache[f'w{label}_original']
        metrics,_=trajectory_errors(original,fit['positions'],times)
        assert np.allclose(original[0]+design_matrix(times,fit['activation'],fit['completion'])@fit['displacements'],fit['positions'],atol=1e-9,rtol=0)
        assert fit['initial_error_m']<1e-12 and fit['final_error_m']<1e-12
        assert np.all(fit['activation']>=times[0]) and np.all(fit['completion']<=times[-1])
        assert np.all(fit['completion']-fit['activation']>=2/120-1e-12)
        if 'initial_objective_m2' in fit:
            assert fit['objective_m2']<=fit['initial_objective_m2']+1e-10
        composition=composition_metrics(times,component_velocities(times,fit['activation'],fit['completion'],fit['displacements']))
        row={'window':label,'method':method,'weight':weight,'component_count':count,'index':index,
             'initialization':start['name'],'origin':origin,'eligible':valid_timing_result(fit),**metrics,
             'objective_m2':fit['objective_m2'],'opposition_cost_m2':fit['opposition_cost_m2'],
             'sum_displacement_norms_m':fit['sum_displacement_norms_m'],
             'relative_component_length_excess':composition['relative_component_length_excess'],
             'status':fit['status'],'message':fit['message'],'nfev':fit['nfev'],
             'residual_evaluations':fit['residual_evaluations'],'optimality':fit['optimality']}
        runs.append(row)
        fit_details.append({**row,'initialization_details':start,'activation':fit['activation'],
                            'completion':fit['completion'],'displacements':fit['displacements']})
        saved[f'{label}_{method}_w{weight}_run{index}_positions']=fit['positions']
        print(label,method,'K',count,'lambda',weight,'run',index,origin,'RMSE cm',100*metrics['position_rmse_m'],flush=True)
        return row

    for start_time,end_time in ((18,21),(30,33)):
        label=f'{start_time}_{end_time}'
        times,original=cache[f'w{label}_timestamps'],cache[f'w{label}_original']
        mask=(full_times>=start_time)&(full_times<=end_time)
        np.testing.assert_array_equal(full_times[mask],times)
        np.testing.assert_array_equal(full_positions[mask],original)
        nominal=None
        for filter_window in FILTER_WINDOWS:
            for prominence in PROMINENCES:
                detection=speed_valley_knots(times,filtered[filter_window][mask],prominence,.15)
                sensitivity.append({'window':label,'filter_window_samples':filter_window,'relative_prominence':prominence,
                                    'component_count':detection['component_count'],'knots':detection['knots'],
                                    'peak_frames':detection['peak_frames'],'no_peak_fallback':detection['no_peak_fallback']})
                if filter_window==17 and prominence==.1:
                    nominal=detection
        count=nominal['component_count']
        windows.append({'window':label,'nominal_detection':nominal})
        saved[f'{label}_timestamps'],saved[f'{label}_original']=times,original
        saved[f'{label}_detection_speed']=filtered[17][mask]
        native_starts=valley_initializations(nominal['knots'])
        common_starts,_=make_initializations(times,original,count,2/120)
        for weight in WEIGHTS:
            for metric in ('objective_m2','position_mse_m2'):
                for upper in (3,4,5,6):
                    selected_rows=[next(s for s in previous['selections'] if s['window']==label and s['component_count']==k and s['weight']==weight)['selected'] for k in range(1,upper+1)]
                    if any(r is None for r in selected_rows):
                        knee={'selected_count':None,'reason':'Missing eligible K fit'}
                    else:
                        costs=[r['objective_m2'] if metric=='objective_m2' else r['position_rmse_m']**2 for r in selected_rows]
                        knee=chord_knee(range(1,upper+1),costs)
                    knees.append({'window':label,'weight':weight,'metric':metric,'upper_count':upper,**knee})
            knee=next(k for k in knees if k['window']==label and k['weight']==weight and k['metric']=='objective_m2' and k['upper_count']==6)
            chosen=knee['selected_count']
            if chosen is not None:
                selection=next(s for s in previous['selections'] if s['window']==label and s['component_count']==chosen and s['weight']==weight)
                fit=unpack_prior(label,chosen,weight,selection['best_index'])
                row=record_fit(label,'A_knee',chosen,weight,selection['best_index'],
                               {'name':selection['selected']['initialization']},fit,'prior_count_sweep')
                comparisons.append({**row,'eligible_count':selection['eligible_count']})
            else:
                comparisons.append({'window':label,'method':'A_knee','weight':weight,'component_count':None,
                                    'reason':knee['reason']})
            pool={}
            common_rows=[]
            for index,init in enumerate(common_starts):
                signature=(init['activation'].tobytes(),init['completion'].tobytes())
                if (label,count,weight,index) in prior_rows:
                    fit,origin=unpack_prior(label,count,weight,index),'prior_count_sweep'
                    for field in ('activation','completion'):
                        np.testing.assert_array_equal(init[field],prior_rows[(label,count,weight,index)]['initialization_details'][field])
                elif signature in pool:
                    fit,origin=pool[signature],'identical_initialization'
                else:
                    fit=fit_joint(times,original,original[0],original[-1],init['activation'],init['completion'],2/120,weight,.3)
                    origin='new_solve'
                pool[signature]=fit
                common_rows.append(record_fit(label,'B_common',count,weight,index,init,fit,origin))
            native_rows=[]
            for index,init in enumerate(native_starts):
                signature=(init['activation'].tobytes(),init['completion'].tobytes())
                if signature in pool:
                    fit,origin=pool[signature],'same_initialization_reuse'
                else:
                    fit=fit_joint(times,original,original[0],original[-1],init['activation'],init['completion'],2/120,weight,.3)
                    origin='new_solve'
                pool[signature]=fit
                native_rows.append(record_fit(label,'B_valleys',count,weight,index,init,fit,origin))
            for method,group in [('B_common',common_rows),('B_valleys',native_rows)]:
                eligible=[r for r in group if r['eligible']]
                if eligible:
                    best=min(eligible,key=lambda r:(r['objective_m2'],r['index']))
                    comparisons.append({**best,'eligible_count':len(eligible)})
                else:
                    comparisons.append({'window':label,'method':method,'weight':weight,'component_count':count,
                                        'eligible_count':0,'reason':'No eligible result; no fallback'})
    result={**protocol,'windows':windows,'knee_sensitivity':knees,'valley_sensitivity':sensitivity,
            'comparisons':comparisons,'runs':runs,'details':fit_details,
            'accounting':{origin:sum(r['origin']==origin for r in runs) for origin in set(r['origin'] for r in runs)},
            'limitations':['Custom operational speed-valley detector, not a verbatim reproduction of a published algorithm',
                           'B sensitivity grid reports count changes, not full fitting at every detector setting',
                           'K=1 fallback for no prominent peaks is not proof of one biological unit',
                           'A range dependence and B detector settings remain assumptions',
                           'No hard timing boundary after fitting; candidate valleys may move',
                           'One clip, two windows, no ground-truth goal count']}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(output/'runs.csv',serializable(runs))
    write_csv(output/'comparisons.csv',serializable([r for r in comparisons if 'position_rmse_m' in r]))
    np.savez_compressed(output/'fits.npz',**saved)
    plot_results(output,result,saved,raw_speed,full_times)
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT.parents[1]/name).read_bytes()).hexdigest()==digest
    for row in comparisons:
        print('COMPARISON',row,flush=True)


def plot_results(output,result,saved,raw_speed,full_times):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,window in enumerate(result['windows']):
        label=window['window']
        times=saved[f'{label}_timestamps']
        mask=(full_times>=times[0])&(full_times<=times[-1])
        speed=saved[f'{label}_detection_speed']
        det=window['nominal_detection']
        axes[row,0].plot(times,raw_speed[mask],color='gray',alpha=.6,label='Raw speed')
        axes[row,0].plot(times,speed,color='tab:orange',label='Detection speed only')
        peaks=np.asarray(det['peak_frames'],dtype=int)
        boundaries=np.asarray(det['boundary_frames'],dtype=int)
        axes[row,0].scatter(times[peaks],speed[peaks],color='tab:green',label='Retained peaks')
        axes[row,0].scatter(times[boundaries[1:-1]],speed[boundaries[1:-1]],color='red',label='Valley boundaries')
        axes[row,0].set_title(f'{label.replace("_","-")} s: B proposes K={det["component_count"]}')
        axes[row,0].set_ylabel('Speed (m/s)')
        axes[row,0].set_xlabel('Original clip time (s)')
        knee=next(k for k in result['knee_sensitivity'] if k['window']==label and k['weight']==1e-6 and k['metric']=='objective_m2' and k['upper_count']==6)
        if 'normalized_counts' in knee:
            axes[row,1].plot(knee['normalized_counts'],knee['normalized_costs'],'o-',label='Normalized total objective')
            axes[row,1].plot([0,1],[1,0],'--',color='gray',label='Endpoint chord')
            for k,x,y in zip(range(1,7),knee['normalized_counts'],knee['normalized_costs']):
                axes[row,1].annotate(f'K={k}',(x,y),xytext=(3,5),textcoords='offset points',fontsize=8)
        axes[row,1].set_title(f'A, lambda=1e-6: selected K={knee["selected_count"]}')
        axes[row,1].set_xlim(-.04,1.10)
        axes[row,1].set_ylim(-.04,1.08)
        axes[row,1].set_xlabel('Normalized K')
        axes[row,1].set_ylabel('Normalized cost')
    for axis in axes.flat:
        axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[0,1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'selection_explanation.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for axis,window in zip(axes,result['windows']):
        matrix=np.array([[next(r['component_count'] for r in result['valley_sensitivity'] if r['window']==window['window'] and r['filter_window_samples']==fw and r['relative_prominence']==pr) for pr in PROMINENCES] for fw in FILTER_WINDOWS])
        axis.imshow(matrix,origin='lower',vmin=1,vmax=max(6,int(matrix.max())),cmap='viridis')
        for i in range(3):
            for j in range(3):
                axis.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white')
        axis.set_xticks(range(3),[str(p) for p in PROMINENCES])
        axis.set_yticks(range(3),[str(w) for w in FILTER_WINDOWS])
        axis.set_xlabel('Relative peak prominence')
        axis.set_ylabel('Derivative filter window (samples)')
        axis.set_title(window['window'].replace('_','-')+' s: B proposed K')
    fig.tight_layout()
    fig.savefig(output/'valley_count_sensitivity.png',dpi=160)
    plt.close(fig)
    for window in result['windows']:
        label=window['window']
        times,original=saved[f'{label}_timestamps'],saved[f'{label}_original']
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves=[('Original',original,'black')]
        for method,color in [('A_knee','tab:blue'),('B_valleys','tab:orange'),('B_common','tab:green')]:
            record=next(r for r in result['comparisons'] if r['window']==label and r['weight']==1e-6 and r['method']==method)
            if 'index' in record:
                p=saved[f'{label}_{method}_w1e-06_run{record["index"]}_positions']
                curves.append((f'{method}, K={record["component_count"]}',p,color))
        for name,p,color in curves:
            v=np.gradient(p,times,axis=0,edge_order=2)
            for c in range(3):
                axes[c,0].plot(times,p[:,c],color=color,label=name,linewidth=1.1)
                axes[c,1].plot(times,v[:,c],color=color,label=name,linewidth=1)
        for c in range(3):
            axes[c,0].set_ylabel(f'{"XYZ"[c]} position (m)')
            axes[c,1].set_ylabel(f'{"XYZ"[c]} velocity (m/s)')
        for axis in axes.flat:
            axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        for axis in axes[-1]:
            axis.set_xlabel('Original clip time (s)')
        fig.tight_layout()
        fig.savefig(output/f'trajectories_{label}.png',dpi=160)
        plt.close(fig)


def plot_saved():
    output=ROOT/'results/step10_count_selection'
    result=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with np.load(output/'fits.npz') as data:
        saved={key:data[key].copy() for key in data.files}
    with np.load(ROOT.parents[1]/'results/wrist_trajectory.npz') as data:
        times,positions=data['timestamps'].astype(float),data['positions'].astype(float)
    raw_speed=np.linalg.norm(np.gradient(positions,times,axis=0,edge_order=2),axis=1)
    plot_results(output,result,saved,raw_speed,times)


if __name__=='__main__':
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.plots_only:
        plot_saved()
    else:
        main()
