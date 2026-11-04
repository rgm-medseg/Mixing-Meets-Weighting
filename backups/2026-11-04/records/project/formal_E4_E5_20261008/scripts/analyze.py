"""Own selected-model comparison to R1; complete paired descriptive evidence."""
import argparse,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
KEYS=['dice','iou','hd95','asd']
def pairs(cur,ref,ids,task):
    lookup={tuple(r[k] for k in ids):r for r in ref};assert len(lookup)==len(ref)
    assert {tuple(r[k] for k in ids) for r in cur}==set(lookup)
    out=[]
    for c in cur:
        r=lookup[tuple(c[k] for k in ids)];row=dict(task=task,scope='v2_B+spatial CTW TRC vs plain' if task=='E4' else 'common protocol official AdaMix CT vs R1',**{k:c[k] for k in ids})
        for k in KEYS:
            a,b=float(c[k]),float(r[k]);row.update({f'current_{k}':a,f'R1_{k}':b,f'R1_minus_current_{k}':b-a,f'current_minus_R1_{k}':a-b})
            if k in ['dice','iou']:row.update({f'R1_minus_current_{k}_pp':100*(b-a),f'current_minus_R1_{k}_pp':100*(a-b)})
        row['unit']='Dice/IoU fractions; deltas pp; distances grid, lower better';out.append(row)
    return out
def savefig(fig,name):
    fig.canvas.draw();rend=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    texts=[t for a in fig.axes for t in [a.title,a.xaxis.label,a.yaxis.label]]
    if fig._suptitle:texts.append(fig._suptitle)
    for t in texts:
        if t.get_text():
            b=t.get_window_extent(rend);assert b.x0>=-1 and b.y0>=-1 and b.x1<=width+1 and b.y1<=height+1,(name,t.get_text())
    for ext in ['png','svg']:fig.savefig(WORK/'analysis'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
def independent_metrics(p,g):
    from scipy.ndimage import binary_erosion,distance_transform_edt,generate_binary_structure
    p=p.astype(bool);g=g.astype(bool)
    if not p.any() and not g.any():return dict.fromkeys(KEYS,0.)
    if not p.any() or not g.any():return dict(dice=0.,iou=0.,hd95=float(np.linalg.norm(p.shape)),asd=float(np.linalg.norm(p.shape)))
    n=int((p&g).sum());conn=generate_binary_structure(3,1);ps=p^binary_erosion(p,conn);gs=g^binary_erosion(g,conn)
    dp=distance_transform_edt(~gs)[ps];dg=distance_transform_edt(~ps)[gs]
    return dict(dice=2*n/(int(p.sum())+int(g.sum())),iou=n/int((p|g).sum()),hd95=float(np.percentile(np.concatenate([dp,dg]),95)),asd=float(dp.mean()))
def qualitative(task,patientpairs,framepairs):
    import h5py
    median=float(np.median([r['current_minus_R1_dice'] for r in patientpairs]))
    ordered=sorted(patientpairs,key=lambda r:(r['current_minus_R1_dice'],r['patient']))
    choices=[('maximum_current_minus_R1',ordered[-1]),('nearest_median_current_minus_R1',min(patientpairs,key=lambda r:(abs(r['current_minus_R1_dice']-median),r['patient']))),('minimum_current_minus_R1',ordered[0])]
    idxC={r['case']:r for r in read_table(WORK/'results/prediction_index.csv')};idxR={r['case']:r for r in read_table(PARENT/'results/prediction_index.csv')};out=[];checks=[]
    for rule,pair in choices:
        candidates=[r for r in framepairs if r['patient']==pair['patient']]
        selected=sorted(candidates,key=lambda r:(-abs(r['current_minus_R1_dice']),r['case'],int(r['class_id'])))[0]
        case=selected['case'];cl=int(selected['class_id'])
        with h5py.File(DATA/'data'/(case+'.h5'),'r') as f:im=f['image'][:];gt=f['label'][:]
        with np.load(idxC[case]['prediction']) as f:cur=f['prediction']
        with np.load(idxR[case]['prediction']) as f:ref=f['prediction']
        g=gt==cl;a=cur==cl;b=ref==cl;visible=(g|a|b).sum((1,2));errors=((a^g).astype(int)+(b^g).astype(int)).sum((1,2));valid=np.where(visible>0)[0];assert len(valid)>0
        z=int(valid[np.argmax(errors[valid])]) if errors.max()>0 else int(np.argmax(g.sum((1,2))))
        assert visible[z]>0
        fig,axes=plt.subplots(1,4,figsize=(12,3.7),layout='constrained');base=im[z];lo,hi=np.percentile(base,[1,99]);base=np.clip((base-lo)/max(hi-lo,1e-6),0,1)
        for ax,title,m in zip(axes,['Image','GT','R1 (TRC)',task+' (plain residual)' if task=='E4' else 'E5 (official AdaMix CT)'],[None,gt[z],ref[z],cur[z]]):
            ax.imshow(base,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest')
            if m is not None:
                for c,color in [(1,'#0072B2'),(2,'#E69F00'),(3,'#D55E00')]:
                    if (m==c).any():ax.contour(m==c,levels=[.5],colors=[color],linewidths=.9)
            ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{rule} | {case} | {selected["class_name"]} z={z}\nRV blue / MYO orange / LV red; native axes; common validation-selected student1',fontsize=9);savefig(fig,'A6_'+rule)
        out.append(dict(task=task,rule=rule,patient=pair['patient'],case=case,class_id=cl,class_name=selected['class_name'],z=z,current_minus_R1_Dice_pp=pair['current_minus_R1_dice_pp'],GT_target_pixels=int(g[z].sum()),current_target_pixels=int(a[z].sum()),R1_target_pixels=int(b[z].sum()),union_errors=int(errors[z]),slice_rule='visible GT/pred target; maximum combined class error; first z tie; zero error maxGT',axes='z,y,x; origin upper; equal coordinate/scale',colors='RV blue MYO orange LV red',checkpoint_sha256=read_json(WORK/'reports/SELECTED_CHECKPOINT.json')['sha256'],figure='analysis/A6_'+rule+'.png'))
        for label,pred in [(task,cur),('R1',ref)]:
            source=read_table((WORK if label==task else PARENT)/'results/case_class_metrics.csv')
            for c in [1,2,3]:
                m=independent_metrics(pred==c,gt==c);old=next(r for r in source if r['case']==case and int(r['class_id'])==c)
                assert all(abs(m[k]-float(old[k]))<1e-10 for k in KEYS)
                checks.append(dict(task=label,case=case,class_id=c,prediction_sha256=idxC[case]['prediction_sha256'] if label==task else idxR[case]['prediction_sha256'],GT_sha256=sha(DATA/'data'/(case+'.h5')),**m))
    write_csv(WORK/'results/qualitative_selection.csv',out);write_csv(WORK/'reports/INDEPENDENT_CASE_RECALC.csv',checks);return out
def main():
    task=os.environ['BETA_TASK'];receipt=WORK/'reports/ANALYSIS_RECEIPT.json'
    if receipt.exists():
        v=read_json(receipt);assert all(sha(WORK/p)==h for p,h in v['outputs'].items());print('Analysis reused');return
    cur={k:read_table(WORK/'results'/n) for k,n in [('summary',f'{task}_results.csv'),('patients',f'{task}_patient_metrics.csv'),('frames',f'{task}_case_metrics.csv'),('classes',f'{task}_class_metrics.csv'),('frame_classes','case_class_metrics.csv')]}
    ref={k:read_table(PARENT/'results'/n) for k,n in [('summary','R1_results.csv'),('patients','R1_patient_metrics.csv'),('frames','R1_case_metrics.csv'),('classes','R1_class_metrics.csv'),('frame_classes','case_class_metrics.csv')]}
    pp={}
    for name,ids in [('summary',[]),('patients',['patient']),('frames',['case','patient']),('classes',['class_name']),('frame_classes',['case','patient','class_id','class_name'])]:
        pp[name]=pairs(cur[name],ref[name],ids,task);write_csv(WORK/'results'/f'paired_{name}.csv',pp[name])
    assert len(cur['frame_classes'])==120 and len(cur['patients'])==20
    for k in KEYS:
        assert abs(np.mean([float(r[k]) for r in cur['frame_classes']])-float(cur['summary'][0][k]))<1e-12
        for patient in cur['patients']:assert abs(np.mean([float(r[k]) for r in cur['frame_classes'] if r['patient']==patient['patient']])-float(patient[k]))<1e-12
    selection=qualitative(task,pp['patients'],pp['frame_classes'])
    ordered=sorted(pp['patients'],key=lambda r:r['R1_minus_current_dice_pp']);fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    ax.bar([r['patient'] for r in ordered],[r['R1_minus_current_dice_pp'] for r in ordered]);ax.axhline(0,color='k',lw=.8);ax.tick_params(axis='x',rotation=60);ax.set(ylabel='R1 minus '+task+' Dice (pp)',title='All 20 paired patients, ED/ES means');savefig(fig,'A4_patient_Dice')
    vals=read_table(WORK/'results/validation_trace.csv');rv=read_table(PARENT/'results/validation_trace.csv');trace=read_table(WORK/'results'/f'{task}_control_trace.csv')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for rows,label in [(rv,'R1 (TRC + v2_B + spatial CTW)'),(vals,task)]:
        ax.plot([int(r['step']) for r in rows],[100*float(r['student1_macro_dice']) for r in rows],label=label)
    ax.legend();ax.set(xlabel='Joint iteration',ylabel='Validation foreground macro Dice (%)',title='Each run uses its own strict validation selection');savefig(fig,'A4_validation')
    fields=['q_B','s','actual_area','ctw_loss','weighted_ctw_loss','loss','step_seconds'] if task=='E4' else ['loss','supervised_loss','unsupervised_loss','included_ratio','step_seconds']
    windows=[]
    for lo,hi in [(i,i+5000) for i in range(0,30000,5000)]+[(0,30000)]:
        rr=[r for r in trace if lo<int(r['step'])<=hi];row=dict(task=task,lower_exclusive=lo,upper_inclusive=hi,physical_rows=len(rr),unique_steps=len({r['step'] for r in rr}))
        for k in fields:
            a=np.array([float(r[k]) for r in rr]);row.update({k+'_mean':float(a.mean()),k+'_min':float(a.min()),k+'_max':float(a.max())})
        if task=='E4':row['rectangle_sides']=';'.join(map(str,sorted({int(r['rectangle_h']) for r in rr})))
        windows.append(row)
    write_csv(WORK/'results/fixed_window_control_summary.csv',windows)
    fig,axes=plt.subplots(2,1,figsize=(9,6),layout='constrained');steps=[int(r['step']) for r in trace]
    axes[0].plot(steps,[float(r['loss']) for r in trace],lw=.4);axes[0].set(ylabel='Official total loss' if task=='E5' else 'Inherited total loss',title=task+' training trajectory')
    if task=='E4':
        axes[1].plot(steps,[float(r['actual_area']) for r in trace],lw=.4);axes[1].set(ylabel='Actual exchanged area fraction')
        write_csv(WORK/'results/scale_distribution.csv',[dict(side=d,count=sum(int(r['rectangle_h'])==d for r in trace),fraction=sum(int(r['rectangle_h'])==d for r in trace)/len(trace),actual_area=d*d/65536) for d in sorted({int(r['rectangle_h']) for r in trace})])
    else:
        axes[1].plot(steps,[float(r['included_ratio']) for r in trace],lw=.4);axes[1].set(ylabel='Official confident pixel ratio')
    axes[1].set(xlabel='Joint iteration');savefig(fig,'A1_control_and_loss')
    history=read_json(WORK/'state/stage_history.json');segments=read_json(WORK/'state/train_segments.json')
    write_csv(WORK/'results/runtime_cost.csv',[dict(task=task,stage=h['stage'],start=h['start'],end=h['end'],seconds=h['elapsed_seconds'],exit_code=h['exit_code'],scope='Popen wall, includes imports/validation/IO and <=5s polling lag') for h in history])
    resource=read_table(MAIN_WORK/'results/resource_samples.csv');nv=[int(r['own_NVML_process_mib']) for r in resource if r['task']==task and r['stage']=='TRAIN' and r['own_NVML_process_mib']!='NA']
    atomic_json(WORK/'results/runtime_cost.json',dict(task=task,training_Popen_seconds=sum(h['elapsed_seconds'] for h in segments),failed_training_seconds=sum(h['elapsed_seconds'] for h in segments if h['exit_code'] not in [0,75]),peak_cuda_allocated_bytes=max(int(r['peak_cuda_bytes']) for r in trace),peak_cuda_reserved_bytes=max(int(r['peak_cuda_reserved_bytes']) for r in trace),sampled_peak_NVML_own_process_mib=max(nv) if nv else 'NA',NVML_sampling_interval_seconds=30,NVML='observed sampled process memory; separately whole-card memory; not CUDA allocator values',short=read_json(WORK/'reports/SHORT_CHECK.json'),validation_seconds='included in train wall; E5 independent val times in traces; E4 not separately instrumented'))
    delta=np.array([r['current_minus_R1_dice'] for r in pp['patients']]);counts=dict(improved=int((delta>1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),declined=int((delta<-1e-12).sum()))
    atomic_json(WORK/'reports/ANALYSIS_RECEIPT.json',dict(status='PASS',time=now(),task=task,patient_counts=counts,selected_examples=selection,independent_metrics=True,outputs={str(p.relative_to(WORK)):sha(p) for d in ['results','analysis'] for p in (WORK/d).glob('*') if p.is_file() and p.suffix in ['.csv','.json','.png','.svg']},canvas_bounds_checked=True))
    print(task,'analysis PASS',counts)
if __name__=='__main__':main()
