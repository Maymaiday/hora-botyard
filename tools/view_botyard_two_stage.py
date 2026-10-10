#!/usr/bin/env python3
"""Offline geometry and simulation diagnostics; does not initialize Isaac Gym."""
import argparse,json
from pathlib import Path
import numpy as np
from botyard_two_stage_core import transform,expand

def plot_geometry(geo,c,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=geo.points(expand(c['close16']),transform(c['rootStart7'][:3],quat=c['rootStart7'][3:]))
    fig=plt.figure(figsize=(9,6));ax=fig.add_subplot(111,projection='3d')
    colors={'ff':'tab:blue','lf':'tab:orange','mf':'tab:green','rf':'tab:red','th':'tab:purple'}
    for name,v in points.items():
        stride=max(1,len(v)//200);v=v[::stride]
        ax.scatter(v[:,0],v[:,1],v[:,2],s=2,alpha=.35,color=colors.get(name[:2],'gray'))
    theta=np.linspace(0,2*np.pi,50);z=np.array([0,.08*c['scale']])+.001;x=.04*c['scale']*np.cos(theta);y=.04*c['scale']*np.sin(theta)
    ax.plot_surface(np.tile(x,(2,1)),np.tile(y,(2,1)),np.tile(z[:,None],(1,50)),color='gold',alpha=.4)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='Closed target geometry ONLY; not physics-validated')
    ax.set_box_aspect((1,1,1));fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def plot_report(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads(Path(path).read_text());rows=d.get('frames',[])
    if not rows:return
    t=np.array([r['simTime'] for r in rows]);relative=np.array([r['relativeObject7'][:3] for r in rows])
    fig,ax=plt.subplots(4,1,figsize=(10,10),sharex=True)
    ax[0].plot(t,[r['bottomZ']*1000 for r in rows]);ax[0].set_ylabel('Cylinder bottom (mm)')
    ax[1].plot(t,np.linalg.norm(relative-relative[0],axis=1)*1000);ax[1].set_ylabel('Relative change (mm)')
    ax[2].plot(t,[len(r['contacts']['fingers']) for r in rows],label='side-contact fingers')
    ax[2].plot(t,['th' in r['contacts']['fingers'] for r in rows],label='thumb contact');ax[2].legend();ax[2].set(xlabel='Simulated time (s)',ylabel='Contact count',ylim=(-.2,5.3))
    if all(r.get('target20') and r.get('q20') for r in rows):
        errors=np.abs(np.array([r['target20'] for r in rows])-np.array([r['q20'] for r in rows]))
        ax[3].plot(t,errors.max(axis=1),label='max joint target error')
        ax[3].plot(t,errors[:,16:].max(axis=1),label='max thumb target error')
        ax[3].axhline(.08,color='red',linestyle='--',label='empty-hand gate .08 rad')
        ax[3].legend()
    ax[3].set(xlabel='Simulated time (s)',ylabel='Target error (rad)')
    last=None
    for row in rows:
        if row['stage']!=last:
            for a in ax:a.axvline(row['simTime'],color='gray',alpha=.35)
            ax[0].text(row['simTime'],ax[0].get_ylim()[1],row['stage'],rotation=75,fontsize=7,va='top');last=row['stage']
    fig.suptitle('accepted='+str(d.get('accepted'))+' | '+Path(path).name);fig.tight_layout();fig.savefig(Path(path).with_suffix('.png'),dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('report',type=Path);a=ap.parse_args();plot_report(a.report);print(a.report.with_suffix('.png'))
if __name__=='__main__':main()
