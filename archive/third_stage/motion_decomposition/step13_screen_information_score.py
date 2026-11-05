import hashlib
import json
from pathlib import Path

import numpy as np

from .step6_explain_real_components import write_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'results/step9_count_comparison'
    output = ROOT/'results/step13_information_score_screen'
    output.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
              for name in ('summary.json', 'fits.npz')}
    s = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    rows, selections = [], []
    with np.load(source/'fits.npz') as z:
        for window in ('18_21', '30_33'):
            for weight in (0, 1e-6):
                for k in range(1, 7):
                    saved = next(r for r in s['selections'] if r['window'] == window
                                 and r['weight'] == weight and r['component_count'] == k)
                    index = saved['best_index']
                    prefix = f'w{window}_k{k}_lambda{weight}_run{index}'
                    residual = (z[prefix+'_positions']-z[f'w{window}_original'])[1:-1]
                    n = residual.size
                    rss = float(np.sum(residual**2))
                    p = 5*k-3
                    assert saved['selected']['parameter_count'] == p
                    correlations = [float(np.corrcoef(residual[:-1,c], residual[1:,c])[0,1])
                                    if np.std(residual[:-1,c])>1e-14 and np.std(residual[1:,c])>1e-14 else None
                                    for c in range(3)]
                    score = float(n*np.log(rss/n)+(p+1)*np.log(n))
                    rows.append({'window': window, 'weight': weight, 'K': k, 'index': index,
                                 'interior_frames': len(residual), 'assumed_independent_scalar_samples': n,
                                 'trajectory_parameters': p, 'variance_parameters': 1,
                                 'rss_m2': rss, 'iid_gaussian_BIC_style_score': score,
                                 'lag1_x': correlations[0], 'lag1_y': correlations[1], 'lag1_z': correlations[2],
                                 'fit_is_penalized': weight != 0,
                                 'sum_displacement_norms_m': saved['selected']['sum_displacement_norms_m']})
                candidates = [r for r in rows if r['window']==window and r['weight']==weight]
                best = min(candidates, key=lambda r: (r['iid_gaussian_BIC_style_score'], r['K']))
                selections.append({'window': window, 'weight': weight, 'screening_minimum_K': best['K'],
                                   'at_tested_upper_boundary': best['K']==6})
    result = {'scope': 'Assumption screen only; not a validated BIC selector or new fitted method',
              'formula': 'n*log(RSS/n)+(5*K-3+1)*log(n); constants common across K omitted',
              'conditioning': 'Observed initial/final positions fixed; endpoint residuals excluded',
              'assumptions_not_established': ['Independent scalar Gaussian residuals with one common variance',
                                             'Regular identifiable model and maximum-likelihood fitted parameters'],
              'source_hashes': hashes, 'rows': rows, 'selections': selections}
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_csv(output/'scores.csv', rows)
    for name, digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for r in selections: print(r)
    for window in ('18_21','30_33'):
        values = [r[c] for r in rows if r['window']==window and r['weight']==1e-6
                  for c in ('lag1_x','lag1_y','lag1_z') if r[c] is not None]
        print(window, 'lag1 correlation range', min(values), max(values))


if __name__ == '__main__':
    main()
