"""CPU-only distribution-tail visualization from the completed, unchanged CSV."""
import numpy as np
from common import *
def main():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from finish import savefig
 plt.rcParams['svg.fonttype']='none'
 path=W/'results/A3_weight_distribution.csv';rr=rows(path);methods=['Full_main','R1_v2_B','R2_fixed92'];fig,axs=plt.subplots(2,3,figsize=(12,6.5));facts=[]
 for col,method in enumerate(methods):
  ss=sorted([r for r in rr if r['method_id']==method],key=lambda r:float(r['bin_low']));x=np.array([(float(r['bin_low'])+float(r['bin_high']))/2 for r in ss]);h=np.array([float(r['spatial_fraction']) for r in ss]);u=np.array([float(r['same_mean_uniform_fraction']) for r in ss]);assert abs(h.sum()-1)<1e-12 and abs(u.sum()-1)<1e-12
  axs[0,col].step(x,h,where='mid',label='Spatial CTW');axs[0,col].step(x,u,where='mid',label='Same-mean uniform',ls='--');axs[0,col].set(title=method,ylabel='Patient-equal pixel fraction',ylim=(0,1.02),xlim=(.25,1.25));axs[0,col].legend(fontsize=7)
  axs[1,col].step(x,np.where(h>0,h,np.nan),where='mid',label='Spatial CTW');axs[1,col].plot(x[u>0],u[u>0],ls='none',marker='o',markersize=3,label='Uniform nonzero bins');axs[1,col].set(yscale='log',ylim=(1e-6,1.02),xlim=(.25,1.25),xlabel='Weight',ylabel='Pixel fraction (log scale)');axs[1,col].legend(fontsize=7)
  assert h[h>0].min()>1e-6
  facts.append(dict(method_id=method,weight_bin='[0.25,0.27)',patient_equal_pixel_fraction=float(h[0]),checkpoint_tag='best',strategy='matched92',scope='all image pixels; not exact equality to0.25; no threshold selection'))
 savefig(fig,'A3_weight_distribution_with_tail','Same unchanged fixed50-bin weight-distribution CSV as the main distribution figure. Upper row linear y; lower row logarithmic y shows low-support transition weights. Zero-support bins are omitted only on the log plot and remain exact zero in CSV; no pseudocounts. Uniform nonzero bins are marked explicitly. Same fixed [0.25,1.25] x range and same y range across methods. Best eval matched92, source patients equal, all image pixels including padding/occlusion as in the original weight normalization. This presentation change adds no model forward, training, GT-dependent threshold or parameter selection.')
 csvout(W/'results/A3_floor_interval_support.csv',facts)
 section='\n## 权重分布补充\n\n'+''.join(f"{r['method_id']}：患者等权像素中{100*r['patient_equal_pixel_fraction']:.3f}%权重落在[0.25,0.27)，不是指精确等于0.25。同均值uniform分布与空间分配均保留。新增linear/log双视图只改善尾部显示，不改变任何统计、阈值或方法。\n" for r in facts)
 p=W/'reports/A3_CTW_SPATIAL_REPORT_CN.md';text=p.read_text();marker='\n## 权重分布补充';text=text.split(marker)[0]+section;atomic(p,text)
 js(W/'reports/PRESENTATION_ADDENDUM_RECEIPT.json',dict(status='PASS',script_sha256=sha(__file__),source_csv_sha256=sha(path),new_figure='A3_weight_distribution_with_tail',GPU_forwards=0,training_updates=0,original_distribution_retained=True))
if __name__=='__main__':main()
