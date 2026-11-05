import json
from pathlib import Path
import numpy as np

from .io_utils import write_csv


def main():
    output=Path(__file__).resolve().parents[1]/'results/mainline'
    summary=json.loads((output/'summary.json').read_text())
    rows=[]
    for label in ('18_21','30_33'):
        fit=summary['details'][label]['fit']
        with np.load(output/'fits.npz') as z:initial=z[f'{label}_original'][0]
        goals=initial+np.cumsum(fit['displacements'],axis=0)
        for i,(a,b,d,g) in enumerate(zip(fit['activation'],fit['completion'],fit['displacements'],goals),1):
            rows.append({'window':label,'component':i,'activation_s':a,'completion_s':b,
                         **{f'd{c}_m':float(d[j]) for j,c in enumerate('xyz')},
                         **{f'goal_{c}_m':float(g[j]) for j,c in enumerate('xyz')}})
    write_csv(output/'goals.csv',rows)


if __name__=='__main__':main()
