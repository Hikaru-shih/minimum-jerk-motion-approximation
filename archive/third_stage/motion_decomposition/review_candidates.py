import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fixed_times import design_matrix
from .metrics import trajectory_errors
from .stability import compare_aligned
from .step4_diagnose_initialization import component_velocities, composition_metrics
from .step5_fit_real_window import serializable
from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('activation', 'completion', 'displacements', 'positions')


def main():
    source = ROOT/'results/step14_extended_counts'
    valley_source = ROOT/'results/step10_count_selection/summary.json'
    output = ROOT/'results/candidate_review'
    output.mkdir(parents=True, exist_ok=True)
    paths = [source/'summary.json', source/'fits.npz', valley_source]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    previous = json.loads(paths[0].read_text())
    valleys = json.loads(valley_source.read_text())
    cases, runs, components = [], [], []
    with np.load(source/'fits.npz') as z:
        for window, k in (('18_21',3), ('30_33',4)):
            selection = next(r for r in previous['selections'] if r['window']==window
                             and r['component_count']==k and r['weight']==1e-6)
            reference_index = selection['best_index']
            t, p = z[f'w{window}_timestamps'], z[f'w{window}_original']
            group = [r for r in previous['details'] if r['window']==window
                     and r['component_count']==k and r['weight']==1e-6]
            fits = {r['index']:{field:z[f'w{window}_k{k}_lambda1e-06_run{r["index"]}_{field}']
                               for field in FIELDS} for r in group}
            reference = fits[reference_index]
            basis = design_matrix(t,reference['activation'],reference['completion'])
            np.testing.assert_allclose(p[0]+basis@reference['displacements'],reference['positions'],atol=1e-12)
            errors, _ = trajectory_errors(p,reference['positions'],t)
            error_norm = np.linalg.norm(reference['positions']-p,axis=1)
            peak = int(np.argmax(error_norm))
            velocity = component_velocities(t,reference['activation'],reference['completion'],reference['displacements'])
            composition = composition_metrics(t,velocity)
            opposition = np.zeros(len(t))
            pairs = []
            for i in range(k):
                for j in range(i+1,k):
                    q = np.maximum(0,-np.sum(velocity[:,i]*velocity[:,j],axis=1))
                    opposition += q
                    pairs.append({'first':i+1,'second':j+1,'maximum_negative_dot_m2_s2':float(q.max())})
            goals = p[0]+np.cumsum(reference['displacements'],axis=0)
            for i in range(k):
                d = reference['displacements'][i]
                components.append({'window':window,'K':k,'component':i+1,
                                   'activation_s':reference['activation'][i], 'completion_s':reference['completion'][i],
                                   'dx_m':d[0],'dy_m':d[1],'dz_m':d[2], 'norm_m':np.linalg.norm(d),
                                   'goal_x_m':goals[i,0],'goal_y_m':goals[i,1],'goal_z_m':goals[i,2],
                                   'nearest_reconstructed_sample_to_goal_m':float(np.linalg.norm(reference['positions']-goals[i],axis=1).min())})
            for r in group:
                compared = compare_aligned(t,reference,fits[r['index']])
                runs.append({'window':window,'K':k,'index':r['index'],'eligible':r['eligible'],
                             'initialization':r['initialization'],'position_rmse_m':r['position_rmse_m'],
                             'velocity_rmse_m_s':r['velocity_rmse_m_s'], 'objective_m2':r['objective_m2'],
                             'objective_gap_m2':r['objective_m2']-selection['selected']['objective_m2'],
                             **compared})
            own = [r for r in runs if r['window']==window]
            case = {'window':window,'K':k,'selected_run':reference_index,'metrics':errors,
                    'peak_position_error_time_s':float(t[peak]),'peak_position_error_m':float(error_norm[peak]),
                    'composition':composition,'opposing_pairs':pairs,
                    'simultaneous_opposition_frame_count':int(np.count_nonzero(opposition>0)),
                    'initial_endpoint_error_m':float(np.linalg.norm(reference['positions'][0]-p[0])),
                    'final_endpoint_error_m':float(np.linalg.norm(reference['positions'][-1]-p[-1])),
                    'eligible_count':sum(r['eligible'] for r in own),
                    'ranges':{key:[min(r[key] for r in own),max(r[key] for r in own)] for key in
                              ('position_rmse_m','velocity_rmse_m_s','maximum_activation_difference_s',
                               'maximum_completion_difference_s','maximum_displacement_vector_difference_m',
                               'trajectory_difference_rmse_m','velocity_signature_rmse_m_s')},
                    'sum_displacement_norms_m':float(np.linalg.norm(reference['displacements'],axis=1).sum())}
            cases.append(case)
            knots = next(r['knots'] for r in valleys['valley_sensitivity'] if r['window']==window
                         and r['filter_window_samples']==17 and r['relative_prominence']==.1)
            plot_case(output,window,k,t,p,reference,goals,velocity,knots)
    result = {'scope':'Consolidated read-only review of existing candidate K=3/4; no fitting or new selector',
              'weight':1e-6,'source_hashes':hashes,'cases':cases,'runs':runs,'components':components,
              'alignment':'Match component analytical XYZ velocity signatures over all 3! or 4! permutations',
              'limits':['Ten deterministic initializations, not independent motion observations',
                        'Nonzero simultaneous opposition is allowed; penalty does not require zero',
                        'Cumulative goals need not lie on the wrist path during overlap',
                        'No threshold-based clustering or biological-unit labels assigned']}
    (output/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
    write_csv(output/'runs_vs_selected.csv',serializable([{key:value for key,value in r.items() if key!='alignment'} for r in runs]))
    write_csv(output/'components.csv',serializable(components))
    for name,digest in hashes.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    print(json.dumps(serializable(cases),indent=2))


def plot_case(output,window,k,t,p,fit,goals,velocity,knots):
    fig = plt.figure(figsize=(13,13))
    grid = fig.add_gridspec(4,2,height_ratios=(1,1,1,1.5))
    original_velocity = np.gradient(p,t,axis=0,edge_order=2)
    colors = plt.get_cmap('tab10').colors
    for c in range(3):
        axis = fig.add_subplot(grid[c,0])
        axis.plot(t,p[:,c],color='0.5',label='Original')
        axis.plot(t,fit['positions'][:,c],color='black',linestyle='--',label='Selected fit')
        axis.set_ylabel('XYZ'[c]+' position (m)'); axis.grid(alpha=.2)
        if c==0: axis.legend(fontsize=8)
        axis = fig.add_subplot(grid[c,1])
        axis.plot(t,original_velocity[:,c],color='0.7',linewidth=.8,label='Original finite difference')
        for i in range(k): axis.plot(t,velocity[:,i,c],color=colors[i],label=f'Component {i+1}')
        axis.plot(t,velocity.sum(axis=1)[:,c],color='black',linestyle='--',label='Analytical sum')
        axis.set_ylabel('XYZ'[c]+' velocity (m/s)'); axis.grid(alpha=.2)
        if c==0: axis.legend(fontsize=7,ncol=2)
    axis = fig.add_subplot(grid[3,0])
    for i in range(k):
        axis.plot([fit['activation'][i],fit['completion'][i]],[i+1,i+1],linewidth=7,color=colors[i])
    for knot in knots[1:-1]: axis.axvline(knot,color='0.5',linestyle=':',alpha=.8)
    axis.set_yticks(range(1,k+1)); axis.set_xlim(t[0],t[-1]); axis.set_xlabel('Clip time (s)')
    axis.set_ylabel('Component interval'); axis.set_title('Dotted: saved nominal speed valleys (not constraints)',fontsize=9)
    axis.grid(alpha=.2)
    axis = fig.add_subplot(grid[3,1],projection='3d')
    axis.plot(*p.T,color='0.6',label='Original wrist')
    axis.plot(*fit['positions'].T,color='black',linestyle='--',label='Selected fit')
    goal_chain = np.vstack([p[0],goals])
    axis.plot(*goal_chain.T,color='tab:pink',marker='o',linestyle=':',label='Cumulative goals, not wrist path')
    axis.text(*p[0],'Start',fontsize=8)
    for i,goal in enumerate(goals): axis.text(*goal,f'G{i+1}',fontsize=8)
    axis.set_xlabel('X (m)'); axis.set_ylabel('Y (m)'); axis.set_zlabel('Z (m)')
    axis.legend(fontsize=7,loc='upper left')
    fig.suptitle(window.replace('_','-')+f' s: selected K={k}, lambda=1e-6; existing fitted parameters',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97))
    fig.savefig(output/f'review_{window}.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
