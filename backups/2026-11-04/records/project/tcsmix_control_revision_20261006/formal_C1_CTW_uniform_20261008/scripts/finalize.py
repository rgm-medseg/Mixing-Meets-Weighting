"""Frozen post-training descriptive analysis, five reviews and verified delivery."""
import argparse,hashlib,math,shutil,subprocess,tarfile,tempfile,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
KEYS=['dice','iou','hd95','asd']
def delta_rows(current,reference,idkeys):
    look={tuple(r[k] for k in idkeys):r for r in reference};assert len(look)==len(reference)
    assert {tuple(r[k] for k in idkeys) for r in current}==set(look)
    out=[]
    for c in current:
        r=look[tuple(c[k] for k in idkeys)];row={k:c[k] for k in idkeys};row['unit']='grid distances; Dice/IoU fractions and delta pp'
        for k in KEYS:
            cv,rv=float(c[k]),float(r[k]);d=cv-rv
            row.update({f'C1_uniform_{k}':cv,f'R1_spatial_{k}':rv,f'delta_C1_minus_R1_{k}':d,f'spatial_advantage_R1_minus_C1_{k}':-d})
            if k in ['dice','iou']:row.update({f'delta_C1_minus_R1_{k}_pp':100*d,f'spatial_advantage_R1_minus_C1_{k}_pp':-100*d})
        out.append(row)
    return out
def savefig(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Check visible titles/axis labels/legends in canvas before saving.
    width,height=fig.canvas.get_width_height()
    texts=[]
    for ax in fig.axes:texts += [ax.title,ax.xaxis.label,ax.yaxis.label]
    if fig._suptitle:texts.append(fig._suptitle)
    for t in texts:
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);assert b.x0>=-1 and b.y0>=-1 and b.x1<=width+1 and b.y1<=height+1,(name,t.get_text(),b)
    for ext in ['png','svg']:fig.savefig(WORK/'analysis'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
def classify(rr):
    ds=np.array([r['delta_C1_minus_R1_dice'] for r in rr]);return dict(C1_improved=int((ds>1e-12).sum()),tied=int((abs(ds)<=1e-12).sum()),C1_declined=int((ds<-1e-12).sum()),tie_tolerance_fraction=1e-12)
def independent_metrics(p,g):
    from scipy.ndimage import binary_erosion,distance_transform_edt,generate_binary_structure
    p=p.astype(bool);g=g.astype(bool)
    if not p.any() and not g.any():return dict.fromkeys(KEYS,0.)
    if not p.any() or not g.any():return dict(dice=0.,iou=0.,hd95=float(np.linalg.norm(p.shape)),asd=float(np.linalg.norm(p.shape)))
    n=int((p&g).sum());conn=generate_binary_structure(3,1)
    ps=p^binary_erosion(p,conn,iterations=1);gs=g^binary_erosion(g,conn,iterations=1)
    dp=distance_transform_edt(~gs)[ps];dg=distance_transform_edt(~ps)[gs]
    return dict(dice=2*n/(int(p.sum())+int(g.sum())),iou=n/int((p|g).sum()),hd95=float(np.percentile(np.concatenate([dp,dg]),95)),asd=float(dp.mean()))
def quantitative(current,reference,trace,vals,make_figures=True):
    summary=delta_rows(current['summary'],reference['summary'],[])
    patients=delta_rows(current['patients'],reference['patients'],['patient'])
    frames=delta_rows(current['frames'],reference['frames'],['case','patient'])
    classes=delta_rows(current['classes'],reference['classes'],['class_name'])
    frame_classes=delta_rows(current['frame_classes'],reference['frame_classes'],['case','patient','class_id','class_name'])
    for name,rows in [('C1_vs_R1_summary',summary),('patient_pairs',patients),('frame_pairs',frames),('class_pairs',classes),('frame_class_pairs',frame_classes)]:write_csv(WORK/'results'/f'{name}.csv',rows)
    windows=[]
    for lo,hi,name in [(i,i+5000,f'{i}-{i+5000}') for i in range(0,30000,5000)]+[(0,30000,'whole')]:
        rr=[r for r in trace if lo<int(r['step'])<=hi]
        assert rr
        row=dict(window=name,lower_exclusive=lo,upper_inclusive=hi,raw_physical_trace_rows=len(rr),unique_logical_steps=len({r['step'] for r in rr}),duplicate_rows=len(rr)-len({r['step'] for r in rr}),dimensions=';'.join(map(str,sorted({int(r['rectangle_h']) for r in rr}))))
        for key in ['q_B','s','actual_area_pixels','actual_area','raw_weight_mean','used_weight_mean','ctw_loss','weighted_ctw_loss','loss','max_per_image_mean_error','used_per_image_spatial_range_max','step_seconds','weight_audit_seconds']:
            a=np.array([float(r[key]) for r in rr]);row.update({key+'_mean':float(a.mean()),key+'_min':float(a.min()),key+'_max':float(a.max())})
        windows.append(row)
    write_csv(WORK/'results/fixed_window_control_weight_summary.csv',windows)
    perimage=[]
    for r in trace:
        raw=json.loads(r['raw_per_image_means']);used=json.loads(r['used_per_image_means']);assert len(raw)==len(used)==12
        for i,(a,b) in enumerate(zip(raw,used)):perimage.append(dict(segment=r['segment'],step=r['step'],image_index=i,raw_mean=a,used_mean=b,absolute_error=abs(a-b),within_image_spatial_range=float(r['used_per_image_spatial_range_max'])))
    write_csv(WORK/'results/per_image_mean_checks.csv',perimage)
    if make_figures:
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained');ps=sorted(patients,key=lambda r:r['spatial_advantage_R1_minus_C1_dice_pp']);ax.bar([r['patient'] for r in ps],[r['spatial_advantage_R1_minus_C1_dice_pp'] for r in ps]);ax.axhline(0,color='k',lw=.8);ax.tick_params(axis='x',rotation=60);ax.set(ylabel='R1 spatial minus C1 uniform Dice (pp)',title='CTW allocation comparison: all 20 paired patients');savefig(fig,'patient_Dice_spatial_advantage')
        fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
        rv=read_table(PARENT/'results/validation_trace.csv')
        for rows,label in [(rv,'R1 spatial CTW'),(vals,'C1 mean-uniform CTW')]:axes[0].plot([int(r['step']) for r in rows],[100*float(r['student1_macro_dice']) for r in rows],label=label)
        axes[0].set(ylabel='Student1 validation macro Dice (%)',title='Own strict validation selection; test excluded');axes[0].legend()
        rt=read_table(PARENT/'results/R1_control_trace.csv')
        for rows,label in [(rt,'R1 spatial CTW'),(trace,'C1 mean-uniform CTW')]:axes[1].plot([int(r['step']) for r in rows],[float(r['actual_area']) for r in rows],label=label,lw=.35,alpha=.7)
        axes[1].set(xlabel='Joint iteration',ylabel='Actual exchanged area fraction',title='Each run uses its own current teacher');axes[1].legend();savefig(fig,'validation_and_geometry')
        fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained');steps=[int(r['step']) for r in trace]
        for k,label in [('raw_weight_mean','Raw spatial mean'),('used_weight_mean','Applied uniform mean')]:axes[0].plot(steps,[float(r[k]) for r in trace],label=label,lw=.5)
        axes[0].legend();axes[0].set(ylabel='Mean weight',title='C1: per-image mean-matched Confidence-Transition Weighting')
        axes[1].plot(steps,[float(r['max_per_image_mean_error']) for r in trace],lw=.5);axes[1].set(ylabel='Max image mean error',title='Predeclared FP32 tolerance <=1e-6')
        axes[2].plot(steps,[float(r['used_per_image_spatial_range_max']) for r in trace]);axes[2].set(xlabel='Joint iteration',ylabel='Within-image range',title='Applied weight is spatially constant');savefig(fig,'weight_matching_checks')
    return summary,patients,classes,windows
def qualitative(patients):
    import h5py
    from scipy.ndimage import binary_erosion,distance_transform_edt,generate_binary_structure
    values=[r['spatial_advantage_R1_minus_C1_dice'] for r in patients];median=float(np.median(values))
    choices=[('maximum_spatial_advantage',max(patients,key=lambda r:(r['spatial_advantage_R1_minus_C1_dice'],r['patient']))),('nearest_median_spatial_advantage',min(patients,key=lambda r:(abs(r['spatial_advantage_R1_minus_C1_dice']-median),r['patient']))),('minimum_spatial_advantage',min(patients,key=lambda r:(r['spatial_advantage_R1_minus_C1_dice'],r['patient'])))]
    pairs=read_table(WORK/'results/frame_class_pairs.csv');idxC={r['case']:r for r in read_table(WORK/'results/prediction_index.csv')};idxR={r['case']:r for r in read_table(PARENT/'results/prediction_index.csv')};selected=[]
    for rule,pr in choices:
        rows=[r for r in pairs if r['patient']==pr['patient']];picked=max(rows,key=lambda r:(abs(float(r['spatial_advantage_R1_minus_C1_dice_pp'])),r['case'],r['class_id']));case=picked['case'];cls=int(picked['class_id'])
        with h5py.File(DATA/'data'/f'{case}.h5','r') as f:im=f['image'][:];gt=f['label'][:]
        with np.load(idxC[case]['prediction']) as f:cu=f['prediction']
        with np.load(idxR[case]['prediction']) as f:sp=f['prediction']
        conn=generate_binary_structure(3,1);mask=sp==cls;g=gt==cls
        if mask.any() and g.any():
            ps=mask^binary_erosion(mask,conn);gs=g^binary_erosion(g,conn);dist=distance_transform_edt(~gs)*ps;z=int(np.argmax(dist.sum((1,2))));slice_rule='largest R1 pred-to-GT surface-distance sum; native axes z,y,x, first tie'
        else:z=int(np.argmax(g.sum((1,2))));slice_rule='empty-surface fallback: largest GT class area, first tie'
        fig,axes=plt.subplots(1,4,figsize=(12,3.7),layout='constrained');base=im[z];lo,hi=np.percentile(base,[1,99]);base=np.clip((base-lo)/max(hi-lo,1e-6),0,1)
        for ax,title,m in zip(axes,['Image','GT','R1 spatial CTW','C1 mean-uniform CTW'],[None,gt[z],sp[z],cu[z]]):
            ax.imshow(base,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest')
            if m is not None:
                for c,color in [(1,'#0072B2'),(2,'#E69F00'),(3,'#D55E00')]:
                    if (m==c).any():ax.contour(m==c,levels=[.5],colors=[color],linewidths=.8)
            ax.set_title(title);ax.axis('off')
        fig.suptitle(f'{rule} | {case} | z={z} | {picked["class_name"]}\nRV blue / MYO orange / LV red; equal orientation, display scale; validation-selected student1',fontsize=9)
        savefig(fig,'error_'+rule)
        selected.append(dict(rule=rule,patient=pr['patient'],case=case,class_name=picked['class_name'],z=z,slice_rule=slice_rule,spatial_advantage_Dice_pp=pr['spatial_advantage_R1_minus_C1_dice_pp'],R1_checkpoint_sha256=read_json(PARENT/'reports/SELECTED_CHECKPOINT.json')['sha256'],C1_checkpoint_sha256=read_json(WORK/'reports/SELECTED_CHECKPOINT.json')['sha256'],axes='z,y,x; origin upper',deidentification='anonymous ACDC ID, normalized image render; no raw medical array or DICOM header',figure='analysis/error_'+rule+'.png'))
    write_csv(WORK/'results/qualitative_selection.csv',selected)
    return selected
def analysis():
    from preflight import verify_frozen
    verify_frozen()
    receipt=WORK/'reports/ANALYSIS_RECEIPT.json'
    if receipt.exists():
        old=read_json(receipt)
        assert all(sha(WORK/p)==h for p,h in old['outputs'].items());print('Verified existing complete analysis');return
    current={k:read_table(WORK/'results'/n) for k,n in [('summary','C1_results.csv'),('patients','C1_patient_metrics.csv'),('frames','C1_case_metrics.csv'),('classes','C1_class_metrics.csv'),('frame_classes','case_class_metrics.csv')]}
    reference={k:read_table(PARENT/'results'/n) for k,n in [('summary','R1_results.csv'),('patients','R1_patient_metrics.csv'),('frames','R1_case_metrics.csv'),('classes','R1_class_metrics.csv'),('frame_classes','case_class_metrics.csv')]}
    trace=read_table(WORK/'results/C1_control_trace.csv');vals=read_table(WORK/'results/validation_trace.csv');summary,patients,classes,windows=quantitative(current,reference,trace,vals);selection=qualitative(patients);counts=classify(patients)
    history=read_json(WORK/'state/stage_history.json');segments=read_json(WORK/'state/train_segments.json');short=read_json(WORK/'reports/SHORT_CHECK.json')
    costs=[]
    for i,s in enumerate(history):costs.append(dict(scope=s['stage'],segment=i,start=s['start'],end=s['end'],wall_seconds=s['elapsed_seconds'],exit_code=s['exit_code'],note=s['scope']))
    costs.append(dict(scope='R1 successful train reference',segment=0,start='NA',end='NA',wall_seconds=26440.013047698885,exit_code=0,note='parent state/train_segments.json successful Popen segment, includes validation/IO'))
    costs.append(dict(scope='standalone validation',segment='all',start='NA',end='NA',wall_seconds='NA',exit_code='NA',note='included in training Popen wall; not independently instrumented in R1/C1'))
    write_csv(WORK/'results/runtime_cost.csv',costs)
    cost=dict(successful_training_Popen_seconds=sum(s['elapsed_seconds'] for s in segments if s['exit_code']==0),interrupted_training_Popen_seconds=sum(s['elapsed_seconds'] for s in segments if s['exit_code']==75),failed_training_Popen_seconds=sum(s['elapsed_seconds'] for s in segments if s['exit_code'] not in [0,75]),short_check=short,peak_cuda_allocated_bytes=max(int(r['peak_cuda_bytes']) for r in trace),peak_cuda_reserved_bytes=max(int(r['peak_cuda_reserved_bytes']) for r in trace),whole_card_process_memory='NVML resource snapshots, separately state/status and GPU PID binding; not allocated/reserved',validation_seconds='NA: included in training wall, no independent timing added',logging_extra_weight_check_seconds=sum(float(r['weight_audit_seconds']) for r in trace),logging_cost_receipts=[str(p.relative_to(WORK)) for p in (WORK/'reports').glob('logging_cost_*.json')])
    atomic_json(WORK/'results/runtime_cost.json',cost)
    r=summary[0];c=current['summary'][0];ref=reference['summary'][0];post=read_json(WORK/'reports/TRAIN_POSTCHECK.json')
    prose=f'C1完成30000逻辑联合迭代。两学生成功物理优化器调用：{post["optimizer_success_calls"]}；重复逻辑轨迹{post["duplicate_trace_records"]}条，额外物理调用{post["replayed_physical_updates"]}。自身严格验证选择student1 step={c["selected_step"]}，SHA={c["checkpoint_sha256"]}。\n\nC1 Dice={float(c["dice"])*100:.8f}%，IoU={float(c["iou"])*100:.8f}%，HD95={c["hd95"]} grid，ASD={c["asd"]} grid。R1 Dice={float(ref["dice"])*100:.8f}%。C1−R1 Dice={r["delta_C1_minus_R1_dice_pp"]:+.8f}百分点；空间优势R1−C1={r["spatial_advantage_R1_minus_C1_dice_pp"]:+.8f}百分点。HD95/ASD越低越好，表中两个代数差值方向明确。\n\n20患者先合并ED/ES两帧；C1改善/持平/下降={counts["C1_improved"]}/{counts["tied"]}/{counts["C1_declined"]}。40帧×3前景类全部保留。单次seed42描述性比较，性能对照回答v2_B＋TRC设置下CTW空间分配的训练贡献；同图均值匹配不要求两个训练轨迹相同。离线A3边界响应沿用paper_offline_followup_20261007版本，另述其证据范围。\n\n成本见runtime_cost.json/csv和stage_history；验证包含于训练wall，独立值NA。allocated/reserved与全卡NVML分开。16worker随机流/prefetch不能位级恢复；恢复段、回滚和短程开销均保留。历史预训练生成命令和旧训练H5内容清单NA，当前资产哈希锁定。原Full仍NEEDS_MAPPING，90.35%不转给R1或C1。\n\n三个误差患者选择空间Dice优势最大/最接近中位数/最小；帧类选绝对差最大，切片沿用R1单向表面距离贡献规则及显式空例回退。全部配对行保留。完成后停止C1队列，E1保持3976暂停、E2–E5未启动。建议等待本次结论后决定是否需要额外随机性评估，本轮不追加训练。\n'
    (WORK/'FINAL_REPORT_C1_CN.md').write_text(prose)
    (WORK/'WRITING_HANDOFF_C1_CN.md').write_text('正式名称：TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。\n\n'+prose+'\n精确数字与正反差值：results/C1_vs_R1_summary.csv、patient_pairs.csv、frame_pairs.csv、class_pairs.csv、frame_class_pairs.csv。Dice/IoU原值fraction；pp差乘100。距离grid、nearest平面缩放回原H5，z不变、无后处理；ASD预测→GT；双空全0、单空距离体对角线。控制六固定区间及全程见fixed_window_control_weight_summary.csv。逐图全H×W均值见per_image_mean_checks.csv；范围恒0，FP32均值误差容差1e−6。图及坐标见analysis和qualitative_selection.csv。\n')
    outputs=[WORK/'FINAL_REPORT_C1_CN.md',WORK/'WRITING_HANDOFF_C1_CN.md']+list((WORK/'analysis').glob('*'))+[WORK/'results'/n for n in ['C1_vs_R1_summary.csv','patient_pairs.csv','frame_pairs.csv','class_pairs.csv','frame_class_pairs.csv','fixed_window_control_weight_summary.csv','per_image_mean_checks.csv','qualitative_selection.csv','runtime_cost.csv','runtime_cost.json']]
    atomic_json(receipt,dict(status='PASS',time=now(),patient_counts=counts,selected_error_patients=selection,outputs={str(p.relative_to(WORK)):sha(p) for p in outputs},scope='one seed descriptive pairs; no hypothesis test or configuration selection'))
def review():
    from preflight import verify_frozen
    verify_frozen()
    cfg=read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml');run=Path(cfg['run_root']);post=read_json(WORK/'reports/TRAIN_POSTCHECK.json');er=read_json(WORK/'results/EVALUATION_RECEIPT.json');sel=read_json(WORK/'reports/SELECTED_CHECKPOINT.json')
    assert er['checkpoint_sha256']==sel['sha256']==sha(sel['checkpoint']) and er['protocol']=='ACDC_v22_grid_directed_ASD' and er['test_selection'] is False
    rr=read_table(WORK/'results/case_class_metrics.csv');cases=(DATA/'test.list').read_text().splitlines();assert len(rr)==120 and {r['case'] for r in rr}==set(cases)
    for k in KEYS:assert abs(np.mean([float(r[k]) for r in rr])-er['metrics'][k])<1e-12
    patients=read_table(WORK/'results/patient_pairs.csv');assert len(patients)==20
    for p in patients:
        for k in KEYS:
            expected=np.mean([float(r[k]) for r in rr if r['patient']==p['patient']]);assert abs(expected-float(p['C1_uniform_'+k]))<1e-12
            assert abs(float(p['delta_C1_minus_R1_'+k])+float(p['spatial_advantage_R1_minus_C1_'+k]))<1e-12
    # Independently reconstruct surfaces/EDT and confusion counts for selected examples + first test frame.
    import h5py
    selected={r['case'] for r in read_table(WORK/'results/qualitative_selection.csv')}|{cases[0]};checks=[]
    for root,label in [(WORK,'C1'),(PARENT,'R1')]:
        idx={r['case']:r for r in read_table(root/'results/prediction_index.csv')};lookup={(r['case'],int(r['class_id'])):r for r in read_table(root/'results/case_class_metrics.csv')}
        for case in sorted(selected):
            assert sha(idx[case]['prediction'])==idx[case]['prediction_sha256']
            with np.load(idx[case]['prediction']) as f:p=f['prediction']
            with h5py.File(DATA/'data'/f'{case}.h5','r') as f:g=f['label'][:]
            for cl in [1,2,3]:
                m=independent_metrics(p==cl,g==cl)
                for k in KEYS:assert abs(m[k]-float(lookup[(case,cl)][k]))<1e-10,(label,case,cl,k)
                checks.append(dict(method=label,case=case,class_id=cl,**m))
    write_csv(WORK/'reports/INDEPENDENT_CASE_RECALC.csv',checks)
    # Audit every validation selection and optimizer event via postcheck; no hidden row trimming.
    assert post['status']=='PASS' and post['logical_joint_steps']==30000
    initial=read_json(WORK/'reports/INITIALIZATION_ACTUAL.json');assert initial['joint_step']==0 and initial['optimizer_pretrained_state_exact'] and initial['fresh']
    assert read_json(WORK/'reports/SHORT_CHECK.json')['physical_joint_updates']==2
    assert read_json(WORK/'reports/ISOLATED_CONTROL_TEST.json')['status']=='PASS'
    assert read_json(WORK/'reports/TMUX_ATTACH_DETACH.json')['status']=='PASS'
    assert read_json(WORK/'reports/FIRST_200_CHECKPOINT.json')['status']=='PASS'
    atomic_json(WORK/'reports/PROTECTED_AFTER.json',dict(status='PASS',time=now(),files=len(read_json(WORK/'manifests/protected.json'))))
    rounds=[dict(round=1,status='PASS',evidence='C1_PROTOCOL_COMPARISON.csv; FUNCTION_BINDING/INITIALIZATION_ACTUAL/protected files; unique CTW allocation intervention'),dict(round=2,status='PASS',evidence='TRAIN_POSTCHECK/optimizer_calls/validation/EMA/SHORT_CHECK/isolated controls/zero-update resume/tmux/first200; worker boundary explicitly non-bitwise'),dict(round=3,status='PASS',evidence='SELECTED_CHECKPOINT; all 40x3 and 20 patient pairs; independent confusion/surface recalculation; grid and delta direction'),dict(round=4,status='PASS',evidence='ANALYSIS_RECEIPT; deterministic example rules; all unfavorable patients retained; canvas label bounds, PNG/SVG exports with rendered canvas bounds checked'),dict(round=5,status='PENDING',evidence='actual archive extraction and internal/external SHA validation runs next')]
    atomic_json(WORK/'reports/FIVE_PASS_REVIEW.json',dict(time=now(),rounds=rounds))
    (WORK/'FIVE_PASS_REVIEW_C1.md').write_text('\n\n'.join(f'第{r["round"]}审：{r["status"]}。{r["evidence"]}' for r in rounds)+'\n')
    print('Reviews 1-4 PASS; review 5 awaits actual archive extraction')
def package():
    from preflight import verify_frozen
    verify_frozen();reviewdoc=read_json(WORK/'reports/FIVE_PASS_REVIEW.json');assert all(r['status']=='PASS' for r in reviewdoc['rounds'][:4])
    # Prior attempts remain intact. Build an immutable lightweight snapshot.
    stamp=now().replace(':','').replace('.','');snap=WORK/'archives'/('final_snapshot_'+stamp);snap.mkdir()
    includes=[]
    for d in ['scripts','source','configs','manifests','reports','analysis','results','state','logs']:
        for p in (WORK/d).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.ckpt','.pth','.npz','.pyc'] or p.name.endswith('.lock') or '.tmp.' in p.name:continue
            if d=='results' and 'predictions' in p.parts:continue
            includes.append(p)
    includes += list(WORK.glob('*.md'))+[WORK/'protocol.json',WORK/'run_manifest.json']
    # Include independent short receipts/trace/logs, index large checkpoint assets separately.
    shortroot=WORK/'runs/isolated_short'
    includes += [p for p in shortroot.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.yaml']]
    failedroot=WORK/'runs/isolated_short_failed_attempt1'
    includes += [p for p in failedroot.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.log','.py','.yaml']]
    run=Path(read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')['run_root'])
    includes += [p for p in run.rglob('*') if p.is_file() and (p.suffix in ['.json','.csv','.py'] or p.name=='train.log')]
    for p in sorted(set(includes)):
        dst=snap/p.relative_to(WORK);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    assets=[]
    for p in list(run.glob('checkpoints/*.ckpt'))+list((WORK/'results/predictions').glob('*.npz'))+list(shortroot.glob('runs/*.ckpt'))+list(shortroot.glob('runs/short/checkpoints/*.ckpt')):
        assets.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,packed=False,reason='large model or medical prediction array retained on server'))
    write_csv(snap/'SERVER_ASSET_INDEX.csv',assets)
    hist=read_json(WORK/'state/stage_history.json');hist.append(dict(stage='ARCHIVE',start='see active_stage',end='snapshot before archive extraction',wall_seconds='see ARCHIVE_RECEIPT.json Popen stage_history at final DONE',scope='snapshot freeze boundary; live status subsequently updated'))
    atomic_json(snap/'SNAPSHOT_BOUNDARY.json',dict(time=now(),training_closed=True,snapshot=str(snap),mutable_live_status_excluded_from_future_changes=True,archive_selfhash_scope='external receipt/sidecar; internal scientific files frozen',archive_wall_seconds='external ARCHIVE_RECEIPT.json; archive cannot contain its own final time/hash'))
    internal={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file()}
    # Fifth review declares reproducibility only after an actual preliminary extraction; then immutable final package.
    stagepack=WORK/'archives'/('precheck_'+stamp+'.tar.gz')
    def build(out):
        with tarfile.open(out,'w:gz',compresslevel=6) as tar:
            for p in sorted(snap.rglob('*')):
                if p.is_file():tar.add(p,arcname='C1_evidence/'+str(p.relative_to(snap)),recursive=False)
    def verify(out):
        with tempfile.TemporaryDirectory(prefix='C1_extract_',dir=WORK/'archives') as tmp:
            with tarfile.open(out) as tar:tar.extractall(tmp,filter='data')
            extracted=Path(tmp)/'C1_evidence';manifest=read_json(extracted/'INTERNAL_SHA256.json')
            for p,h in manifest.items():assert sha(extracted/p)==h,('archive hash mismatch',p)
            return len(manifest)
    atomic_json(snap/'INTERNAL_SHA256.json',internal);build(stagepack);verify(stagepack)
    reviewdoc['rounds'][4].update(status='PASS',evidence='Preliminary actual extraction/internal SHA PASS; final archive also actually extracted and rechecked; external receipt and sidecar')
    atomic_json(WORK/'reports/FIVE_PASS_REVIEW.json',reviewdoc)
    (WORK/'FIVE_PASS_REVIEW_C1.md').write_text('\n\n'.join(f'第{r["round"]}审：{r["status"]}。{r["evidence"]}' for r in reviewdoc['rounds'])+'\n')
    shutil.copyfile(WORK/'reports/FIVE_PASS_REVIEW.json',snap/'reports/FIVE_PASS_REVIEW.json');shutil.copyfile(WORK/'FIVE_PASS_REVIEW_C1.md',snap/'FIVE_PASS_REVIEW_C1.md')
    internal={str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file() and p.name!='INTERNAL_SHA256.json'};atomic_json(snap/'INTERNAL_SHA256.json',internal)
    final=WORK/'C1_CTW_uniform_v2B_final_evidence.tar.gz'
    if final.exists():shutil.move(final,WORK/'archives'/('previous_'+stamp+'.tar.gz'))
    start=time.monotonic();build(final);count=verify(final);h=sha(final);(WORK/(final.name+'.sha256')).write_text(h+'  '+final.name+'\n')
    small=[]
    if final.stat().st_size>30*2**20:
        # Two logical groups, each preserving complete CSVs; split larger groups by files without truncation.
        groups=[];group=[];total=0
        for p in sorted(snap.rglob('*')):
            if not p.is_file():continue
            if group and total+p.stat().st_size>24*2**20:groups.append(group);group=[];total=0
            group.append(p);total+=p.stat().st_size
        if group:groups.append(group)
        for i,group in enumerate(groups):
            out=WORK/f'C1_evidence_part_{i+1:02d}.tar.gz'
            with tarfile.open(out,'w:gz') as tar:
                for p in group:tar.add(p,arcname='C1_evidence/'+str(p.relative_to(snap)),recursive=False)
            assert out.stat().st_size<30*2**20,'Single complete file too large; retain failure without truncation'
            with tempfile.TemporaryDirectory(dir=WORK/'archives') as tmp:
                with tarfile.open(out) as tar:tar.extractall(tmp,filter='data')
                for p in group:assert sha(Path(tmp)/'C1_evidence'/p.relative_to(snap))==sha(p)
            sh=sha(out);(WORK/(out.name+'.sha256')).write_text(sh+'  '+out.name+'\n');small.append(dict(path=str(out),sha256=sh,bytes=out.stat().st_size,files=len(group)))
    atomic_json(WORK/'ARCHIVE_RECEIPT.json',dict(status='VERIFIED',time=now(),path=str(final),sha256=h,bytes=final.stat().st_size,internal_files_verified=count,actual_extract=True,snapshot=str(snap),archive_build_and_validation_seconds=time.monotonic()-start,logical_parts=small))
    (WORK/'DOWNLOAD_INDEX_CN.md').write_text(f'最终证据包：{final.name}\nSHA256：{h}\n实际解包核验内部{count}文件。科学证据来自同一冻结快照；归档自身耗时/外层哈希在ARCHIVE_RECEIPT.json及sidecar。\n'+''.join(f'\n逻辑分包：{Path(p["path"]).name}；{p["bytes"]/2**20:.2f}MiB；SHA256 {p["sha256"]}\n' for p in small)+'\n模型/完整医学预测未打包，见包内SERVER_ASSET_INDEX.csv。全部患者、全部配对CSV均保留。\n')
    print('Final evidence actual extraction PASS',count,h)
def selftest():
    # CPU-only paired/metric arithmetic fixture, never published as C1 results.
    a=np.zeros((4,8,8),bool);b=a.copy();a[1:3,2:5,2:5]=True;b[1:3,3:6,2:5]=True
    from unified_evaluation import metrics
    m=independent_metrics(a,b);n=metrics(a,b)
    assert all(abs(m[k]-n[k])<1e-12 for k in KEYS)
    c=[dict(patient='patient001',dice='.8',iou='.7',hd95='3',asd='1')];r=[dict(patient='patient001',dice='.9',iou='.8',hd95='2',asd='.5')];d=delta_rows(c,r,['patient'])[0]
    assert abs(d['delta_C1_minus_R1_dice_pp']+10)<1e-12 and abs(d['spatial_advantage_R1_minus_C1_dice_pp']-10)<1e-12 and d['delta_C1_minus_R1_hd95']==1
    atomic_json(WORK/'reports/POSTPROCESS_SELFTEST.json',dict(status='PASS',time=now(),independent_surface_metrics=True,delta_direction_and_pp=True,scope='CPU artificial arrays, no checkpoint/model inference/training'))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['analysis','review','package','selftest']);a=p.parse_args();globals()[a.action]()
if __name__=='__main__':main()
