import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/'results/uniform_baseline_comparison'
    output = source/'report_figures'
    output.mkdir(exist_ok=True)
    s = json.loads((source/'summary.json').read_text())
    methods = ('uniform_sampled_goals','uniform_fitted_displacements','adaptive_timing')
    names = ('Uniform sampled','Uniform fitted','Fitted timing')
    colors = ('0.5','tab:blue','tab:orange')
    plt.rcParams.update({'font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5})
    fig,axes = plt.subplots(2,2,figsize=(7.1,5.2))
    for r,window in enumerate(s['windows']):
        for method,name,color in zip(methods,names,colors):
            rows = [x for x in s['rows'] if x['window']==window and x['method']==method]
            for c,key,scale in ((0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)):
                axes[r,c].plot(range(1,13),[scale*x[key] for x in rows],'-o',markersize=3,color=color,label=name,linewidth=1)
        for c in range(2):
            axes[r,c].axvline(s['selected_counts'][window],color='black',linestyle=':',linewidth=.8)
            axes[r,c].set_title(window.replace('_','-')+' s'); axes[r,c].set_xticks((1,3,6,9,12)); axes[r,c].set_xlabel('K'); axes[r,c].grid(alpha=.2)
        axes[r,0].set_ylabel('Position RMSE (cm)'); axes[r,1].set_ylabel('Velocity RMSE (m/s)')
    handles,labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(output/'errors_vs_count.png',dpi=240); plt.close(fig)
    with np.load(source/'fits.npz') as z:
        for window,k in s['selected_counts'].items():
            t,p = z[window+'_timestamps'],z[window+'_original']
            fig,axes = plt.subplots(3,2,figsize=(7.1,6.6),sharex=True)
            curves = [('Original',p,'black')]+[(name,z[f'{window}_{method}_k{k}_positions'],color) for method,name,color in zip(methods,names,colors)]
            for name,fit,color in curves:
                v = np.gradient(fit,t,axis=0,edge_order=2)
                for c in range(3):
                    axes[c,0].plot(t,fit[:,c],color=color,label=name,linewidth=.9)
                    axes[c,1].plot(t,v[:,c],color=color,label=name,linewidth=.8)
            for c in range(3):
                axes[c,0].set_ylabel('XYZ'[c]+' (m)'); axes[c,1].set_ylabel('XYZ'[c]+' (m/s)')
            for axis in axes.flat: axis.grid(alpha=.2)
            axes[0,0].set_title('Position'); axes[0,1].set_title('Velocity')
            for axis in axes[-1]: axis.set_xlabel('Clip time (s)')
            handles,labels = axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=4,fontsize=7.5)
            fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(output/f'trajectories_{window}.png',dpi=240); plt.close(fig)


if __name__=='__main__':
    main()
