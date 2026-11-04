"""Predefined full-trajectory, non-test quality and honest historical comparisons."""
import collections
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def write_area_proposal(rows):
    latest={int(r['step']):r for r in rows}
    complete=set(latest)==set(range(1,30001))
    if not complete:
        (WORK/'R2_FIXED_AREA_PROPOSAL.md').write_text('# R2固定面积建议\n\nNA：R1完整30000步真实面积轨迹尚未完成；不以部分轨迹锁定固定尺度。R2未授权、未启动。\n');return
    mean=sum(float(latest[i]['actual_area']) for i in range(1,30001))/30000
    pixels=mean*65536;d=min(range(1,256),key=lambda x:(abs(x*x-pixels),x));area=d*d/65536
    proposal=dict(status='PROPOSAL_ONLY_NOT_RUN',time=now(),trace_sha256=sha(WORK/'results/R1_control_trace.csv'),n_final_logical_steps=30000,
        raw_rows=len(rows),duplicates=len(rows)-30000,mean_actual_area=mean,mean_actual_pixels=pixels,h=d,w=d,fixed_side=d/256,fixed_area=area,
        area_absolute_error=area-mean,relative_error=(area-mean)/mean,error_percentage_points=100*(area-mean),
        rule='nearest integer square pixel area to full mean actual area; lower side wins exact tie',test_metrics_read=False,R2_started=False)
    atomic_json(WORK/'analysis/R2_area_proposal.json',proposal)
    (WORK/'R2_FIXED_AREA_PROPOSAL.md').write_text(f'''# R2固定面积建议（不启动）

仅使用R1完整真实逻辑步1–30000的实际交换面积，保留所有原始记录；恢复重复step采用最后一次对应最终轨迹的记录。平均面积{mean:.10f}（{100*mean:.7f}%），平均像素面积{pixels:.4f}。
原方形比例下选择最接近平均像素面积的整数平方：**{d}×{d}**，fixed_side={d}/256={d/256:.10f}，实际面积{area:.10f}。有符号面积差{area-mean:.10f}，绝对匹配误差{abs(area-mean):.10f}，相对误差{(area-mean)/mean:.6%}，百分点差{100*(area-mean):.7f}。
定义只读取trace，不读取测试指标；建议在最终测试前由train_postcheck生成。当前不写正式R2锁定配置，不启动R2。以后是否执行需用户另行授权；R1相对旧Full差值不能单独归因于动态调度。
轨迹SHA256：{proposal['trace_sha256']}。面积计算原始统计见analysis/R2_area_proposal.json。
''')

def trajectory():
    path=WORK/'results/R1_control_trace.csv';rows=read_table(path) if path.exists() else [];latest={int(r['step']):r for r in rows}
    final=[latest[k] for k in sorted(latest)];summary=[]
    for lo,hi in [(1,5000),(5001,10000),(10001,15000),(15001,20000),(20001,25000),(25001,30000),(1,30000)]:
        rr=[r for r in final if lo<=int(r['step'])<=hi]
        for version,key in [('v2_B','q'),('old_same_teacher_replay','old_q')]:
            ds=[int(r['rectangle_h']) if version=='v2_B' else round(256*(.18+.20*float(r[key]))) for r in rr]
            counts=collections.Counter(ds)
            summary.append(dict(segment_start=lo,segment_end=hi,controller=version,n_recorded=len(rr),requested_n=hi-lo+1,
                coverage='COMPLETE' if len(rr)==hi-lo+1 else 'PARTIAL',sizes=json.dumps(dict(sorted(counts.items()))),n_sizes=len(counts),
                modal_fraction=max(counts.values())/len(rr) if rr else 'NA',q_min=min(float(r[key]) for r in rr) if rr else 'NA',q_max=max(float(r[key]) for r in rr) if rr else 'NA',
                q_mean=np.mean([float(r[key]) for r in rr]) if rr else 'NA',area_mean=np.mean([d*d/65536 for d in ds]) if ds else 'NA',
                area_min=min(d*d/65536 for d in ds) if ds else 'NA',area_max=max(d*d/65536 for d in ds) if ds else 'NA',
                min_endpoint_hits=ds.count(46),max_endpoint_hits=ds.count(97),
                fallback=sum(r['fallback']=='True' for r in rr) if version=='v2_B' else 0,
                size_changes=sum(a!=b for a,b in zip(ds,ds[1:])),scope='recorded logical iterations; old replay uses identical v2 teacher, not independently trained old model'))
    write_csv(WORK/'analysis/scale_distribution.csv',summary);write_area_proposal(rows)
    if final:
        steps=[int(r['step']) for r in final];fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        axes[0,0].plot(steps,[float(r['old_q']) for r in final],label='old: same v2 teacher',linewidth=.5);axes[0,0].plot(steps,[float(r['q']) for r in final],label='v2_B',linewidth=.5);axes[0,0].legend();axes[0,0].set_ylabel('Teacher control proxy')
        axes[0,1].plot(steps,[int(r['rectangle_h']) for r in final],linewidth=.5);axes[0,1].set_ylabel('Actual rectangle side (pixels)')
        axes[1,0].plot(steps,[100*float(r['actual_area']) for r in final],linewidth=.5);axes[1,0].set_ylabel('Actual exchanged area (%)')
        for c in [1,2,3]:axes[1,1].plot(steps,[int(r[f'n_{c}']) for r in final],label=f'class{c}',linewidth=.5)
        axes[1,1].set_ylabel('Original LCC support pixels');axes[1,1].legend()
        axes[2,0].plot(steps,[float(r['ctw_loss']) for r in final],linewidth=.5);axes[2,0].set_ylabel('Original spatial CTW loss')
        for r in summary[:-2]:
            if r['controller']=='v2_B' and r['n_recorded']:
                cs=json.loads(r['sizes']);axes[2,1].plot([int(k) for k in cs],[v/r['n_recorded'] for v in cs.values()],label=f"{r['segment_start']}-{r['segment_end']}",marker='o')
        axes[2,1].legend(fontsize=7);axes[2,1].set(xlabel='Rectangle side (pixels)',ylabel='Within-stage fraction')
        for a in axes.ravel()[:-1]:a.set_xlabel('Joint iteration')
        fig.suptitle('R1 v2_B: real training trajectory | predefined 5000-step segments')
        save(fig,'control_trajectory')
    stable=[float(r['step_seconds']) for r in final if int(r['step'])>50 and int(r['step'])%200 not in range(0,11)]
    segments=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else []
    cost=dict(recorded_logical_steps=len(final),raw_trace_rows=len(rows),training_active_segment_seconds=sum(r['elapsed_seconds'] for r in segments),
        stable_timer_median_seconds=float(np.median(stable)) if stable else 'NA',stable_recorded_steps=len(stable),
        timer_scope='start after loader yields to control trace after optimizer; excludes validation/EMA/checkpoints following record and batch waiting',
        peak_allocated_bytes=max(int(r['peak_cuda_bytes']) for r in final) if final else 'NA',
        median_controller_wall_ms=float(np.median([float(r['controller_ms']) for r in final])) if final else 'NA',
        host='n1',gpu='GPU0 A10040GB',precision='FP32',batch24=24,original_teacher_forward_count=4,
        log_overhead='append optimizer events/step trace and inherited revision telemetry; no duplicate old-teacher forward; no claim of paired full training overhead delta')
    atomic_json(WORK/'analysis/runtime_cost.json',cost)
    return final,summary

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(WORK/f'analysis/{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def confidence_quality():
    import torch
    import revision_runtime as r
    from controller import signals
    from accepted_diagnostic_inputs import samples,pseudolabel,scores
    cfg=read_json(WORK/'configs/R1_full_v2B.yaml');r.CFG=cfg
    items=[s for s in samples() if s['gt'] is not None];assert len(items)==34
    original=read_table(WORK/'handoff_sources/diagnostic_samples.csv');current=read_table(WORK/'analysis/diagnostic_samples.csv')
    assert {s['sample']:s['input_sha256'] for s in original}=={s['sample']:s['input_sha256'] for s in current}
    rows=[]
    for which in ['best','last']:
        cp=Path(cfg['run_root'])/f'checkpoints/{which}.ckpt'
        if not cp.exists():continue
        model=r.BCP_net(ema=True);ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['ema_model'],strict=True)
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
        for mode in ['eval','train_snapshot']:
            model.train(mode=='train_snapshot')
            for cohort in ['labeled_train','validation']:
                subset=[s for s in items if s['cohort']==cohort]
                for start in range(0,len(subset),12):
                    batch=subset[start:start+12];n=len(batch)//2;assert n*2==len(batch)
                    model.load_state_dict(frozen);torch.manual_seed(4261100+start//12);torch.cuda.manual_seed_all(4261100+start//12)
                    x=torch.stack([s['x'] for s in batch]).cuda()
                    with torch.no_grad():
                        la=model(x[:n]);lb=model(x[n:]);y=torch.cat([pseudolabel(la,nms=1),pseudolabel(lb,nms=1)]);logits=torch.cat([la,lb])
                        for i,sample in enumerate(batch):
                            logit=logits[i:i+1];yy=y[i:i+1];st=signals(logit,logit,yy,yy,cfg['foreground_ids'],cfg['background_id'])
                            pred=yy[0].cpu().numpy();gt=sample['gt'];quality=scores(pred,gt)
                            row=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),checkpoint_step=ck['global_step'],branch='ema_model',mode=mode,
                                sample=sample['id'],patient=sample['patient'],cohort=cohort,old_q=st['old_q'],q_B=st['q_margin'],fallback=st['fallback'],
                                background_fraction=st['predicted_background_fraction'],background_confidence=st['background_confidence'],
                                rectangle_side=round(256*(.18+.20*st['q_margin'])),GT_foreground_fraction=float((gt>0).mean()),
                                GT_present_classes=';'.join(str(c) for c in [1,2,3] if (gt==c).any()),**quality)
                            for c in cfg['foreground_ids']:row.update({f'n_{c}':st['foreground_counts'][str(c)]//2,f'prob_class{c}':st['q_by_class'][str(c)],f'margin_class{c}':st['margin_by_class'][str(c)]})
                            rows.append(row)
                    model.load_state_dict(frozen)
        del model,ck,frozen;torch.cuda.empty_cache()
    write_csv(WORK/'analysis/confidence_quality.csv',rows)
    if rows:
        from scipy.stats import spearmanr
        corr=[]
        for cp in sorted({row['checkpoint'] for row in rows}):
            for mode in ['eval','train_snapshot']:
                for cohort in ['labeled_train','validation']:
                    rr=[row for row in rows if row['checkpoint']==cp and row['mode']==mode and row['cohort']==cohort]
                    rho=spearmanr([row['q_B'] for row in rr],[row['macro_dice'] for row in rr]).statistic
                    corr.append(dict(checkpoint=cp,mode=mode,cohort=cohort,slices=len(rr),patients=len({row['patient'] for row in rr}),
                        spearman=float(rho) if np.isfinite(rho) else 'NA',interpretation='descriptive repeated snapshots/slices; no parameter or checkpoint selection by quality'))
        write_csv(WORK/'analysis/confidence_quality_correlations.csv',corr)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for mode in ['eval','train_snapshot']:
            rr=[row for row in rows if row['mode']==mode and row['cohort']=='validation'];axes[0].scatter([row['q_B'] for row in rr],[row['macro_dice'] for row in rr],s=13,alpha=.6,label=mode)
            axes[1].scatter([row['background_fraction'] for row in rr],[row['q_B'] for row in rr],s=13,alpha=.6,label=mode)
        axes[0].set(xlabel='Foreground class-balanced probability margin',ylabel='Three-foreground macro Dice');axes[1].set(xlabel='Processed predicted background fraction',ylabel='Control proxy')
        for a in axes:a.legend()
        fig.suptitle('Offline final teachers on fixed validation slices | not test / not full trajectory');save(fig,'confidence_quality')
    return dict(status='DONE' if rows else 'PARTIAL',rows=len(rows),labeled_patients=7,validation_patients=10,test_inputs=0,
        GT_use='diagnostic only; never in unlabeled controller',middle_checkpoints='NA: only retained selected best and last teachers',
        state='independent process, reset cloned buffers per batch, no optimizer updates; post-step teachers not historical per-step logits')

def historical_comparison():
    current=WORK/'results/case_class_metrics.csv'
    if not current.exists():return dict(status='PARTIAL',reason='R1 final evaluation missing')
    nowrows=read_table(current);differences=[];summaries=[]
    for ref in ['Historical_Full','Historical_baseline']:
        path=WORK/f'reports/{ref}_case_class_metrics.csv';receipt=WORK/f'reports/{ref}_EVALUATION_RECEIPT.json'
        if not path.exists() or not receipt.exists():continue
        old=read_json(receipt)
        if old['protocol']!='ACDC_v22_grid_directed_ASD' or old['manifest_sha256']!=sha(DATA/'test.list'):continue
        rows=read_table(path);lookup={(r['case'],r['class_name']):r for r in rows}
        for r in nowrows:
            prior=lookup.get((r['case'],r['class_name']))
            if prior is None:continue
            differences.append(dict(reference=ref,case=r['case'],patient=r['patient'],class_name=r['class_name'],unit='grid',
                **{k+'_delta':float(r[k])-float(prior[k]) for k in ['dice','iou','hd95','asd']},
                **{k+'_R1':float(r[k]) for k in ['dice','iou','hd95','asd']},**{k+'_reference':float(prior[k]) for k in ['dice','iou','hd95','asd']}))
        paired=[d for d in differences if d['reference']==ref]
        for level,field in [('class','class_name'),('case','case'),('patient','patient')]:
            for value in sorted({r[field] for r in paired}):
                rr=[r for r in paired if r[field]==value]
                summaries.append(dict(reference=ref,level=level,id=value,n_rows=len(rr),**{k+'_delta':np.mean([r[k+'_delta'] for r in rr]) for k in ['dice','iou','hd95','asd']}))
    write_csv(WORK/'analysis/paired_case_class_differences.csv',differences);write_csv(WORK/'analysis/paired_aggregated_differences.csv',summaries)
    pp=[r for r in summaries if r['level']=='patient' and r['reference']=='Historical_Full']
    if pp:
        pp.sort(key=lambda r:r['dice_delta']);fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.bar(range(len(pp)),[100*r['dice_delta'] for r in pp]);ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(len(pp)),[r['id'] for r in pp],rotation=65);ax.set_ylabel('R1 − old Full Dice (percentage points)');save(fig,'patient_differences')
    qualitative(summaries)
    return dict(status='DONE' if len(differences)==240 else 'PARTIAL',matched_case_class_rows=len(differences),
        comparison='same reconstruction-grid evaluation/test list; older imported-source/pretrain-generating details incomplete',
        interpretation='R1-old Full delta alone does not isolate dynamic scheduling benefit; R2 not run')

def qualitative(summaries):
    import h5py
    rr=[r for r in summaries if r['level']=='case' and r['reference']=='Historical_Full']
    if not rr:return
    median=float(np.median([r['dice_delta'] for r in rr]));ordered=[('largest_gain',max(rr,key=lambda r:(r['dice_delta'],r['id']))),
        ('typical_median_delta',min(rr,key=lambda r:(abs(r['dice_delta']-median),r['id']))),('largest_decline_or_smallest_gain',min(rr,key=lambda r:(r['dice_delta'],r['id'])))]
    selected=[]
    for rule,row in ordered:
        case=row['id'];oldpath=PROJECT/f'revision_runs_20261006/runs/Historical_Full/evaluation/predictions/{case}.npz';newpath=WORK/f'results/predictions/{case}.npz'
        if not oldpath.exists() or not newpath.exists():continue
        source=DATA/'data'/f'{case}.h5'
        with h5py.File(source,'r') as f:image=f['image'][:];gt=f['label'][:]
        z=int(np.argmax((gt>0).sum((1,2))))
        with np.load(oldpath) as f:oldpred=f['prediction']
        with np.load(newpath) as f:newpred=f['prediction']
        fig,axs=plt.subplots(1,4,figsize=(10,3),layout='constrained');base=image[z];lo,hi=np.percentile(base,[1,99]);base=np.clip((base-lo)/max(hi-lo,1e-6),0,1)
        for ax,title,pred in zip(axs,['Image','GT','Old Full','R1 v2_B'],[None,gt[z],oldpred[z],newpred[z]]):
            ax.imshow(base,cmap='gray',vmin=0,vmax=1)
            if pred is not None:
                for c,color in [(1,'#0072B2'),(2,'#E69F00'),(3,'#D55E00')]:
                    if (pred==c).any():ax.contour(pred==c,levels=[.5],colors=[color],linewidths=.8)
            ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{rule} | anonymous ACDC ID {case} | z={z} | Dice delta={100*row["dice_delta"]:.3f}pp',fontsize=9)
        name='qualitative_'+rule;save(fig,name)
        selected.append(dict(rule=rule,case=case,z=z,dice_delta=row['dice_delta'],slice_selection='max GT foreground area, deterministic first tie',
            deidentification='existing ACDC anonymous patient/frame IDs, image-only normalized render, no DICOM headers/PHI or raw arrays',
            prediction=str(newpath),reference_prediction=str(oldpath),GT=str(source),figure=f'analysis/{name}.png'))
    write_csv(WORK/'analysis/qualitative_selection.csv',selected)

def main():
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    final,summary=trajectory();failures=[];details={}
    if (WORK/'results/EVALUATION_RECEIPT.json').exists():
        for name,fn in [('confidence_quality',confidence_quality),('historical_comparison',historical_comparison)]:
            try:details[name]=fn()
            except Exception as exc:
                import traceback
                text=traceback.format_exc();(WORK/f'reports/analysis_{name}_failure.txt').write_text(text);failures.append(dict(task=name,error=str(exc)));details[name]=dict(status='PARTIAL',reason=str(exc))
    else:details.update(confidence_quality=dict(status='PARTIAL',reason='final evaluation not completed'),historical_comparison=dict(status='PARTIAL',reason='R1 test metrics unavailable'))
    atomic_json(WORK/'analysis/ANALYSIS_RECEIPT.json',dict(status='ANALYZED' if not failures and len(final)==30000 and details['historical_comparison']['status']=='DONE' else 'PARTIAL',
        time=now(),recorded_final_logical_steps=len(final),predefined_segments='six5000-step segments plus whole',details=details,failures=failures,optimizer_updates=0,R2_started=False))
    print('Analysis finished; all recorded favorable and unfavorable cases retained.')

if __name__=='__main__':main()
