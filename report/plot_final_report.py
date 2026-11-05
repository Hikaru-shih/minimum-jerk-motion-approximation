"""Render saved evidence with labels sized for the Word report; no fitting."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'final_report_assets'


def main():
    OUT.mkdir(exist_ok=True)
    s=json.loads((ROOT/'results/mainline/summary.json').read_text())
    sensitivity=json.loads((ROOT/'results/timing_bound_sensitivity/summary.json').read_text())
    z=dict(np.load(ROOT/'results/mainline/fits.npz'))
    plt.rcParams.update({'font.size':14,'axes.titlesize':15,'legend.fontsize':12,'axes.labelsize':14})
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=False)
    for ax,label in zip(axes,('18_21','30_33')):
        t,v=z[f'{label}_timestamps'],z[f'{label}_speed'];d=s['details'][label]['detection']
        ax.plot(t,v,label='Detection speed',color='tab:blue');peaks=d['peak_frames']
        ax.scatter(t[peaks],v[peaks],color='green',s=25,label='Retained peaks',zorder=3)
        for q in d['knots'][1:-1]:ax.axvline(q,color='red',linestyle=':',linewidth=1.4)
        ax.set(title=label.replace('_','–')+f' s   K={d["component_count"]}',ylabel='Speed (m/s)',xlabel='Clip time (s)');ax.grid(alpha=.2)
    axes[0].legend(loc='upper left',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'count_selection.png',dpi=220);plt.close(fig)
    for label in ('18_21','30_33'):
        t,p=z[f'{label}_timestamps'],z[f'{label}_original'];detail=s['details'][label];k=detail['detection']['component_count']
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        curves=[('Original',p,'black')]+[(name,z[f'{label}_{key}_k{k}_positions'],c) for key,name,c in [('uniform_sampled','Uniform sampled','tab:gray'),('uniform_fitted','Uniform fitted','tab:blue'),('valley_fitted','Valley fitted','tab:green')]]
        for name,curve,color in curves:
            v=np.gradient(curve,t,axis=0,edge_order=2)
            for j in range(3):axes[j,0].plot(t,curve[:,j],label=name,color=color,linewidth=1.2);axes[j,1].plot(t,v[:,j],color=color,linewidth=1.2)
        for j in range(3):axes[j,0].set_ylabel('XYZ'[j]+' position (m)');axes[j,1].set_ylabel('XYZ'[j]+' velocity (m/s)')
        for ax in axes.flat:ax.grid(alpha=.2);ax.tick_params(labelsize=12)
        for ax in axes[-1]:ax.set_xlabel('Clip time (s)')
        axes[0,0].legend(fontsize=11);fig.suptitle(label.replace('_','–')+f' s   Nominal K={k}',fontsize=16)
        fig.tight_layout();fig.savefig(OUT/f'trajectories_{label}.png',dpi=220);plt.close(fig)
        f=detail['fit'];a,b,d=(np.asarray(f[n]) for n in ('activation','completion','displacements'))
        u=np.clip((t[:,None]-a)/(b-a),0,1)
        vv=30*u[:,:,None]**2*(1-u[:,:,None])**2/(b-a)[None,:,None]*d[None,:,:]
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        axes[0].plot(t,z[f'{label}_speed'],color='black',label='Detection speed')
        for q in detail['detection']['knots'][1:-1]:axes[0].axvline(q,color='red',linestyle=':')
        for j in range(k):axes[1].plot(t,np.linalg.norm(vv[:,j,:],axis=1),label=f'Component {j+1}')
        axes[0].set_ylabel('Speed (m/s)');axes[1].set_ylabel('Component speed (m/s)');axes[1].set_xlabel('Clip time (s)')
        for ax in axes:ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=11)
        fig.tight_layout();fig.savefig(OUT/f'components_{label}.png',dpi=220);plt.close(fig)
    plt.rcParams.update({'font.size':15,'axes.labelsize':15,'axes.titlesize':16})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,label in enumerate(('18_21','30_33')):
        for method,name,c in [('uniform_sampled','Uniform sampled','tab:gray'),('uniform_fitted','Uniform fitted','tab:blue')]:
            rr=sorted([r for r in s['rows'] if r['window']==label and r['method']==method],key=lambda r:r['K'])
            for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:axes[i,j].plot([r['K'] for r in rr],[r[key]*scale for r in rr],'o-',label=name,color=c)
        rr=[r for r in s['sweep'] if r['window']==label]
        for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
            axes[i,j].scatter([r['K'] for r in rr],[r[key]*scale for r in rr],color='tab:green',label='Valley partitions',zorder=3)
            for r in rr:
                if r['nominal']:axes[i,j].scatter(r['K'],r[key]*scale,marker='*',s=170,color='black',label='Nominal proposal',zorder=4)
            axes[i,j].set(title=label.replace('_','–')+' s',xlabel='Component count K');axes[i,j].grid(alpha=.2);axes[i,j].tick_params(labelsize=13)
        axes[i,0].set_ylabel('Position RMSE (cm)');axes[i,1].set_ylabel('Velocity RMSE (m/s)')
    axes[0,0].legend(fontsize=10,loc='upper right');fig.tight_layout();fig.savefig(OUT/'errors_vs_count.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,label in enumerate(('18_21','30_33')):
        rr=[r for r in sensitivity['rows'] if r['window']==label]
        for j,key,scale in [(0,'position_rmse_m',100),(1,'velocity_rmse_m_s',1)]:
            axes[i,j].plot([r['setting'].capitalize() for r in rr],[r[key]*scale for r in rr],'o-')
            axes[i,j].set_title(label.replace('_','–')+f' s   K={rr[0]["K"]}');axes[i,j].grid(alpha=.2);axes[i,j].tick_params(labelsize=12)
        axes[i,0].set_ylabel('Position RMSE (cm)');axes[i,1].set_ylabel('Velocity RMSE (m/s)')
    fig.tight_layout();fig.savefig(OUT/'sensitivity.png',dpi=220);plt.close(fig)
    print('Rendered seven report figures from saved mainline and sensitivity evidence')


if __name__=='__main__':main()
