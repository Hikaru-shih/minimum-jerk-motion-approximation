"""Recompute the main report evidence without invoking the fitting routines."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def reconstruct(t,p0,fit):
    a,b,d=(np.asarray(fit[key]) for key in ('activation','completion','displacements'))
    u=np.clip((t[:,None]-a)/(b-a),0,1)
    pred=p0+(10*u**3-15*u**4+6*u**5)@d
    np.testing.assert_allclose(pred,fit['positions'],atol=2e-10,rtol=0)
    return pred


def verify_fit(t,p,fit,metrics=None):
    pred=reconstruct(t,p[0],fit)
    np.testing.assert_allclose(pred[[0,-1]],p[[0,-1]],atol=1e-12,rtol=0)
    distance=np.linalg.norm(pred-p,axis=1)
    v=np.gradient(pred,t,axis=0,edge_order=2)-np.gradient(p,t,axis=0,edge_order=2)
    expected={'position_rmse_m':float(np.sqrt(np.mean(distance**2))),
              'velocity_rmse_m_s':float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),
              'max_position_error_m':float(distance.max()),
              'sum_displacement_norms_m':float(np.linalg.norm(fit['displacements'],axis=1).sum())}
    for key,value in expected.items():
        if key in (metrics or fit):
            np.testing.assert_allclose((metrics or fit)[key],value,atol=2e-10,rtol=1e-8)


def main():
    main_dir=ROOT/'results/mainline';sens_dir=ROOT/'results/timing_bound_sensitivity'
    main=json.loads((main_dir/'summary.json').read_text())
    sensitivity=json.loads((sens_dir/'summary.json').read_text())
    z=dict(np.load(main_dir/'fits.npz'));sz=dict(np.load(sens_dir/'fits.npz'))
    for name,digest in main['protocol']['code_sha256'].items():
        assert hashlib.sha256((ROOT/'motion_decomposition'/name).read_bytes()).hexdigest()==digest
    for name,digest in sensitivity['protocol']['code_sha256'].items():
        assert hashlib.sha256((ROOT/'motion_decomposition'/name).read_bytes()).hexdigest()==digest
    for name,digest in sensitivity['protocol']['source_sha256'].items():
        assert hashlib.sha256((main_dir/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'results/final_pipeline/summary.json').read_bytes()).hexdigest()==main['protocol']['source_sha256']
    control=json.loads((ROOT/'results/final_pipeline/summary.json').read_text())
    for name,digest in control['protocol']['code_sha256'].items():
        assert hashlib.sha256((ROOT/'motion_decomposition'/name).read_bytes()).hexdigest()==digest
    source=ROOT.parents[1]/'results/wrist_trajectory.npz'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==control['protocol']['source_sha256']
    with np.load(source) as original:
        for label in ('18_21','30_33'):
            left,right=map(int,label.split('_'));mask=(original['timestamps']>=left)&(original['timestamps']<=right)
            np.testing.assert_array_equal(original['timestamps'][mask],z[f'{label}_timestamps'])
            np.testing.assert_array_equal(original['positions'][mask],z[f'{label}_original'])
    for row in main['rows']:
        label,k,method=row['window'],row['K'],row['method']
        fit={key:z[f'{label}_{method}_k{k}_{key}'] for key in ('activation','completion','displacements','positions')}
        verify_fit(z[f'{label}_timestamps'],z[f'{label}_original'],fit,row)
    count=0
    for name,item in main['details'].items():
        if 'runs' not in item:continue
        label=name.split('_partition')[0];t,p=z[f'{label}_timestamps'],z[f'{label}_original']
        for fit in item['runs']:
            verify_fit(t,p,fit);count+=1
            timing=np.r_[fit['activation'],fit['completion']]
            assert np.all(timing>=np.asarray(item['lower_bounds'])-1e-12)
            assert np.all(timing<=np.asarray(item['upper_bounds'])+1e-12)
        best=min([fit for fit in item['runs'] if fit['success']],key=lambda f:(f['position_rmse_m'],f['index']))
        assert item['best']['index']==best['index']
    assert len(main['sweep'])==14 and count==56 and len(main['rows'])==30
    for row in main['sweep']:
        item=main['details'][f'{row["window"]}_partition{row["partition"]}']
        fit=item['best'];label=row['window']
        assert row['K']==len(item['detection']['knots'])-1
        verify_fit(z[f'{label}_timestamps'],z[f'{label}_original'],fit,row)
    for row in sensitivity['rows']:
        label,name=row['window'],row['setting'];item=sensitivity['details'][f'{label}_{name}']
        t,p=sz[f'{label}_timestamps'],sz[f'{label}_original']
        verify_fit(t,p,item['best'],row)
        for key in ('activation','completion','displacements','positions'):
            np.testing.assert_allclose(sz[f'{label}_{name}_{key}'],item['best'][key],atol=1e-14)
        for fit in item['runs']:
            verify_fit(t,p,fit)
            timing=np.r_[fit['activation'],fit['completion']]
            assert np.all(timing>=np.asarray(item['lower_bounds'])-1e-12)
            assert np.all(timing<=np.asarray(item['upper_bounds'])+1e-12)
        best=min([r for r in item['runs'] if r['eligible']],key=lambda f:(f['position_rmse_m'],f['index']))
        assert best['index']==row['best_start']
        if name=='nominal':
            np.testing.assert_allclose(best['positions'],main['details'][label]['fit']['positions'],atol=1e-10,rtol=0)
    # The sensitivity changes feasible time boxes, not the detected count or knots.
    for label in ('18_21','30_33'):
        np.testing.assert_array_equal(sz[f'{label}_knots'],main['details'][label]['detection']['knots'])
        for a,b in [('narrow','nominal'),('nominal','wide')]:
            small=sensitivity['details'][f'{label}_{a}'];large=sensitivity['details'][f'{label}_{b}']
            assert np.all(np.asarray(large['lower_bounds'])<=small['lower_bounds'])
            assert np.all(np.asarray(large['upper_bounds'])>=small['upper_bounds'])
    for directory,summary in [(main_dir,main),(sens_dir,sensitivity)]:
        csv_rows=list(csv.DictReader((directory/'metrics.csv').open(newline='')))
        assert len(csv_rows)==len(summary['rows'])
        for actual,expected in zip(csv_rows,summary['rows']):
            for key in ('position_rmse_m','velocity_rmse_m_s','max_position_error_m'):
                np.testing.assert_allclose(float(actual[key]),expected[key],atol=1e-14)
    goals=list(csv.DictReader((main_dir/'goals.csv').open(newline='')))
    assert len(goals)==7
    for row in goals:
        label=row['window'];i=int(row['component'])-1;fit=main['details'][label]['fit']
        goal=z[f'{label}_original'][0]+np.cumsum(fit['displacements'],axis=0)[i]
        for j,c in enumerate('xyz'):np.testing.assert_allclose(float(row[f'goal_{c}_m']),goal[j],atol=1e-14)
        for j,c in enumerate('xyz'):np.testing.assert_allclose(float(row[f'd{c}_m']),fit['displacements'][i][j],atol=1e-14)
        np.testing.assert_allclose(float(row['activation_s']),fit['activation'][i],atol=1e-14)
        np.testing.assert_allclose(float(row['completion_s']),fit['completion'][i],atol=1e-14)
    inventory={'verified':True,'mainline_comparisons':len(main['rows']),'mainline_runs':count,
               'sensitivity_comparisons':len(sensitivity['rows']),'sensitivity_runs':len(sensitivity['runs']),
               'goals':len(goals),'nominal_reproduced':True,
               'scope':'Source and code hashes, original targets, independent polynomial reconstruction, metrics, endpoints, time bounds, count partitions, CSVs and goals checked. This is numerical consistency, not physiological validation.',
               'evidence_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for directory in (main_dir,sens_dir) for p in sorted(directory.iterdir()) if p.is_file() and p.name!='audit.json'}}
    (sens_dir/'audit.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in inventory.items() if k not in ('evidence_sha256','scope')},indent=2))


if __name__=='__main__':main()
