"""K-Dense scientific-visualization workflow: real aligned images, fixed cases, raw patient points."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from common import *
COLORS=['#f4f4f4','#0072B2','#E69F00','#CC79A7']
SEG=ListedColormap(COLORS);SEGN=BoundaryNorm(np.arange(-.5,4.5,1),4)
def readcsv(name):
    with (W/'results'/name).open() as f:return list(csv.DictReader(f))
def numeric(r,key):return float(r[key])
def panel(ax,data,title,cmap='gray',vmin=None,vmax=None,norm=None,bar=None):
    im=ax.imshow(data,cmap=cmap,norm=norm,**({} if norm is not None else dict(vmin=vmin,vmax=vmax)),interpolation='nearest',origin='upper')
    ax.set_title(title,loc='left',fontweight='bold',pad=3);ax.set_xticks([]);ax.set_yticks([])
    if bar:
        cax=ax.inset_axes([0,-.11,1,.035]);cb=ax.figure.colorbar(im,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=5,pad=1);cb.set_label(bar,fontsize=5.5,labelpad=1)
    return im
def export(fig,slug,sources,description):
    out=W/'figures';out.mkdir(exist_ok=True)
    for fmt in ['svg','pdf','png']:fig.savefig(out/f'{slug}.{fmt}',dpi=500 if fmt!='png' else 220,facecolor='white',metadata={'Creator':'Matplotlib; actual frozen model outputs'} if fmt=='pdf' else None)
    js(out/f'{slug}_provenance.json',dict(source_files=sources,source_sha256={p:sha(p) for p in sources},case_selection='fixed before outputs; not selected by Dice, error, or visual appearance',transforms='nearest-neighbor display; no smoothing; full image; common image contrast; no data edits',width_mm=190,height_mm=fig.get_figheight()*25.4,vector_text=True,raster_native_grid=[256,256],png_dpi=220,embedded_raster_export_dpi=500,description=description,manuscript_phase='Neurocomputing draft; journal-specific guide returned403, final production choices pending verification'))
    plt.close(fig)
def style():return plt.rc_context({'font.family':'DejaVu Sans','font.size':7,'axes.titlesize':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def joint():
    paths=[W/'arrays/joint'/f'dynamic_{mode}_b0_p0.npz' for mode in ['uniform','CTW']]
    u,c=[dict(np.load(p)) for p in paths];fig=plt.figure(figsize=(190/25.4,150/25.4));gs=fig.add_gridspec(3,4,height_ratios=[1,1,1.08],left=.07,right=.98,bottom=.12,top=.96,wspace=.28,hspace=.55)
    lo=min(c['image1'].min(),c['image2'].min());hi=max(c['image1'].max(),c['image2'].max())
    ax=fig.add_subplot(gs[0,0]);panel(ax,c['image1'],'a  Mixed weak view',vmin=lo,vmax=hi);ax.plot([12,44],[240,240],color='white',lw=2);ax.text(12,230,'32 grid px',color='white',fontsize=6)
    panel(fig.add_subplot(gs[0,1]),c['image2'],'b  Aligned strong view',vmin=lo,vmax=hi)
    panel(fig.add_subplot(gs[0,2]),c['gt'],'c  Mixed anatomical GT',cmap=SEG,norm=SEGN)
    panel(fig.add_subplot(gs[0,3]),c['disagree'],'d  Disagreement',vmin=0,vmax=1,cmap='Greys')
    panel(fig.add_subplot(gs[1,0]),c['w'],'e  CTW weight',vmin=.25,vmax=1.25,cmap='cividis',bar='w')
    panel(fig.add_subplot(gs[1,1]),c['e'],'f  Consistency error',vmin=0,vmax=.5,cmap='magma',bar='e')
    mx=max(float(u['grad'].max()),float(c['grad'].max()))*1e9
    panel(fig.add_subplot(gs[1,2]),u['grad']*1e9,'g  Uniform gradient',vmin=0,vmax=mx,cmap='magma',bar='L1 gradient × 10⁹')
    panel(fig.add_subplot(gs[1,3]),c['grad']*1e9,'h  CTW gradient',vmin=0,vmax=mx,cmap='magma',bar='L1 gradient × 10⁹')
    rr=readcsv('joint_patient_region.csv');regions=['anatomy','seam','cutout','other_valid'];names=['Anatomy','Seam','Cutout','Other']
    ax=fig.add_subplot(gs[2,:2])
    for i,mode in enumerate(['uniform','CTW']):
        means=[]
        for j,reg in enumerate(regions):
            vals=[100*numeric(r,'gradient_fraction') for r in rr if r['condition']=='dynamic_'+mode and r['region']==reg];means.append(np.mean(vals));xx=j+(-.16 if i==0 else .16)
            ax.scatter(np.full(len(vals),xx),vals,s=8,color='#555555',marker='o' if i==0 else '^',alpha=.55,zorder=3)
        ax.bar(np.arange(4)+(-.16 if i==0 else .16),means,width=.29,color=['#999999','#0072B2'][i],label=mode,hatch='//' if i==0 else None)
    ax.set(title='i  Consistency gradient allocation',ylabel='Patient mean gradient share (%)',xticks=range(4),xticklabels=names,ylim=(0,100));ax.legend(frameon=False)
    ax=fig.add_subplot(gs[2,2:]);regions2=['anatomy','seam','pseudo_correct','pseudo_error'];names2=['Anatomy','Seam','Pseudo\ncorrect','Pseudo\nerror']
    for i,mode in enumerate(['uniform','CTW']):
        means=[]
        for j,reg in enumerate(regions2):
            vals=[numeric(r,'mean_w') for r in rr if r['condition']=='dynamic_'+mode and r['region']==reg and r['mean_w']];means.append(np.mean(vals));ax.scatter(np.full(len(vals),j+(-.16 if i==0 else .16)),vals,s=8,color='#555555',marker='o' if i==0 else '^',alpha=.55,zorder=3)
        ax.bar(np.arange(4)+(-.16 if i==0 else .16),means,width=.29,color=['#999999','#0072B2'][i],label=mode,hatch='//' if i==0 else None)
    ax.set(title='j  Spatial weights by region',ylabel='Patient mean weight',xticks=range(4),xticklabels=names2,ylim=(0,1.25))
    fig.supxlabel('Main Full, frozen validation; dynamic and fixed masks both 97 × 97 in all 9 batches.\nPoints: 10 patients. Gradient is from the 0.05-scaled CTW item only. GT classes: RV blue, MYO orange, LV purple.',fontsize=6)
    export(fig,'Fig_joint_TCSMix_CTW',[str(p) for p in paths]+[str(W/'results/joint_patient_region.csv')],'Aligned mixed images, actual weights/errors/logit gradients and patient-level spatial allocations. Dynamic and fixed geometry coincide on these inputs; the figure does not demonstrate a mixing interaction.')
def trc():
    frame=CFG['val_frames'][0]
    import h5py
    with h5py.File(DATA/'data'/f'{frame}.h5') as f:z=len(f['image'])//2
    path=W/'arrays/trc'/f'{frame}_z{z}.npz';a=dict(np.load(path));fig=plt.figure(figsize=(190/25.4,145/25.4));gs=fig.add_gridspec(3,4,height_ratios=[1,1,1.06],left=.07,right=.98,bottom=.12,top=.96,wspace=.28,hspace=.55)
    panel(fig.add_subplot(gs[0,0]),a['image'],'a  Validation image',vmin=a['image'].min(),vmax=a['image'].max())
    panel(fig.add_subplot(gs[0,1]),a['gt'],'b  Anatomical GT',cmap=SEG,norm=SEGN)
    panel(fig.add_subplot(gs[0,2]),a['pred_normal'],'c  Normal TRC',cmap=SEG,norm=SEGN)
    panel(fig.add_subplot(gs[0,3]),a['pred_gamma0'],'d  γ = 0',cmap=SEG,norm=SEGN)
    panel(fig.add_subplot(gs[1,0]),a['feature_norm'],'e  Pre-TRC feature',vmin=0,vmax=float(a['feature_norm'].max()),cmap='cividis',bar='Feature L2 norm')
    panel(fig.add_subplot(gs[1,1]),a['residual_norm'],'f  Applied residual',vmin=0,vmax=float(a['residual_norm'].max()),cmap='cividis',bar='||γr||₂')
    panel(fig.add_subplot(gs[1,2]),a['logit_change'],'g  Logit response',vmin=0,vmax=float(a['logit_change'].max()),cmap='magma',bar='||z − zγ=0||₂')
    change=np.zeros(a['gt'].shape,np.uint8);change[a['corrected']]=1;change[a['introduced']]=2;change[a['bothwrong_changed']]=3
    ax=fig.add_subplot(gs[1,3]);panel(ax,change,'h  Error changes',cmap=ListedColormap(['#f4f4f4','#009E73','#D55E00','#CC79A7']),norm=BoundaryNorm(np.arange(-.5,4.5,1),4));ax.text(8,242,f"{int(a['changed'].sum())} changed px",fontsize=6,color='black')
    pd=readcsv('trc_patient_dice.csv');ax=fig.add_subplot(gs[2,:2])
    for j,cls in enumerate(['RV','MYO','LV']):
        vals=[numeric(r,'delta_dice')*100 for r in pd if r['class_name']==cls];ax.scatter(np.full(len(vals),j),vals,color=['#0072B2','#E69F00','#CC79A7'][j],s=13,alpha=.75);ax.plot([j-.2,j+.2],[np.mean(vals)]*2,'k-',lw=2)
    ax.axhline(0,color='#555555',lw=.6);ax.set(title='i  Patient Dice response',ylabel='Normal − γ=0 Dice (pp)',xticks=range(3),xticklabels=['RV','MYO','LV'])
    rr=readcsv('trc_patient_region.csv');ax=fig.add_subplot(gs[2,2:]);regs=['anatomy','organ_interior','background']
    for i,key in enumerate(['corrected_rate','introduced_rate']):
        means=[]
        for j,reg in enumerate(regs):
            vals=[100*numeric(r,key) for r in rr if r['region']==reg];means.append(np.mean(vals));ax.scatter(np.full(len(vals),j+(-.16 if i==0 else .16)),vals,s=8,color='#555555',marker='o' if i==0 else '^',alpha=.6,zorder=3)
        ax.bar(np.arange(3)+(-.16 if i==0 else .16),means,width=.29,color=['#009E73','#D55E00'][i],hatch=None if i==0 else '//',label=['Corrected','Introduced'][i])
    ax.set(title='j  Residual-dependent error changes',ylabel='Patient mean pixels (%)',xticks=range(3),xticklabels=['Boundary','Interior','Background']);ax.legend(frameon=False)
    fig.supxlabel(f'{frame}, slice {z}; fixed first validation frame / middle slice.\nGreen: corrected, vermilion: introduced, purple: changed but wrong in both. Points: 10 patients; bars/lines: means.',fontsize=6)
    export(fig,'Fig_TRC_residual_intervention',[str(path),str(W/'results/trc_patient_dice.csv'),str(W/'results/trc_patient_region.csv')],'Same-checkpoint TRC intervention with real feature/residual responses, predictions and errors; all restored outputs were bitwise identical.')
def main():
    with style():joint();trc()
if __name__=='__main__':main()
