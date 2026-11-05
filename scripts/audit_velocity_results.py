"""Independently check saved joint fits without rerunning optimization."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/velocity_objective'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    data=np.load(OUT/'fits.npz')
    mainline=json.loads((ROOT/'results/mainline/summary.json').read_text())
    with (OUT/'metrics.csv').open(newline='') as file:
        csv_rows=list(csv.DictReader(file))
    assert len(csv_rows)==len(summary['rows'])==8
    count=0
    for row,csv_row in zip(summary['rows'],csv_rows):
        label,weight=row['window'],row['weight']
        key=f'{label}_lambda{weight:g}'
        details=summary['details'][key]
        t,p=data[f'{label}_timestamps'],data[f'{label}_original']
        length=max(float(np.sqrt(np.mean(np.sum((p-p.mean(axis=0))**2,axis=1)))),1e-12)
        speed=max(float(np.sqrt(np.mean(np.sum(np.gradient(p,t,axis=0,edge_order=2)**2,axis=1)))),1e-12)
        np.testing.assert_allclose([length,speed],[details['length_scale_m'],details['speed_scale_m_s']],atol=1e-12)
        for run in details['runs']:
            a,b=np.asarray(run['activation']),np.asarray(run['completion'])
            assert np.all(b>a)
            x=np.r_[a,b]
            assert np.all(x>=np.asarray(details['lower_bounds'])-1e-12)
            assert np.all(x<=np.asarray(details['upper_bounds'])+1e-12)
            u=np.clip((t[:,None]-a)/(b-a),0,1)
            basis=u**3*(10+u*(-15+6*u))
            q=p[0]+basis@np.asarray(run['displacements'])
            np.testing.assert_allclose(q,run['positions'],atol=1e-12)
            np.testing.assert_allclose(q[[0,-1]],p[[0,-1]],atol=1e-12)
            pr=float(np.sqrt(np.mean(np.sum((q-p)**2,axis=1))))
            vr=float(np.sqrt(np.mean(np.sum((np.gradient(q,t,axis=0,edge_order=2)-np.gradient(p,t,axis=0,edge_order=2))**2,axis=1))))
            maximum=float(np.max(np.linalg.norm(q-p,axis=1)))
            objective=pr**2/length**2+weight*vr**2/speed**2
            np.testing.assert_allclose([pr,vr,maximum,objective],[run['position_rmse_m'],run['velocity_rmse_m_s'],run['max_position_error_m'],run['objective']],atol=1e-12)
            count+=1
        best=min([r for r in details['runs'] if r['eligible']],key=lambda r:(r['objective'],r['index']))
        assert best['index']==details['best']['index']
        for name in ('positions','activation','completion','displacements'):
            np.testing.assert_allclose(data[f'{key}_{name}'],best[name],atol=1e-12)
        for name in ('position_rmse_m','velocity_rmse_m_s','max_position_error_m','objective'):
            np.testing.assert_allclose([row[name],float(csv_row[name])],best[name],atol=1e-12)
        if weight==0:
            np.testing.assert_allclose(best['positions'],mainline['details'][label]['fit']['positions'],atol=1e-8,rtol=0)
    for name,digest in summary['protocol']['source_sha256'].items():
        assert hashlib.sha256((ROOT/'results/mainline'/name).read_bytes()).hexdigest()==digest
    result={'verified':True,'conditions':8,'runs':count,'source_mainline_unchanged':True,
            'checks':['reconstruction','metrics','objective','bounds','endpoints','best selection','CSV/NPZ','zero controls']}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
