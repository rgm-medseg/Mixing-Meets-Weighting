"""Evidence-based five reviews and immutable actually extracted archives."""
import argparse,shutil,tarfile,tempfile,time,traceback
import numpy as np
from common import *
KEYS=['dice','iou','hd95','asd']
def tasks():return read_json(MAIN_WORK/'state/tasks.json')
def summary():
    rows=[];patients=[];classes=[];caseclasses=[];paired=[];cost=[]
    for t in ['E4','E5']:
        root=MAIN_WORK/'tasks'/t;status=tasks()[t]['status']
        if status=='DONE':
            rows+=read_table(root/'results'/f'{t}_results.csv')
            patients+=read_table(root/'results'/f'{t}_patient_metrics.csv');classes+=read_table(root/'results'/f'{t}_class_metrics.csv')
            caseclasses+=read_table(root/'results/case_class_metrics.csv');paired+=read_table(root/'results/paired_patients.csv')
            cost+=read_table(root/'results/runtime_cost.csv')
        else:rows.append(dict(run=t,status=status,checkpoint='NA',checkpoint_sha256='NA',selected_step='NA',cases='NA',patients='NA',unit='grid',**dict.fromkeys(KEYS,'NA')))
    for folder,prefix in [('formal_R1_20261007','R1'),('formal_R2_fixed92_20261007','R2'),('formal_C1_CTW_uniform_20261008','C1')]:
        root=PROJECT/'tcsmix_control_revision_20261006'/folder
        rows+=read_table(root/'results'/f'{prefix}_results.csv')
        dst=MAIN_WORK/'historical_supplements'/prefix;dst.mkdir(exist_ok=True,parents=True)
        for p in [root/'results'/f'{prefix}_results.csv',root/'results/EVALUATION_RECEIPT.json',root/'reports/SELECTED_CHECKPOINT.json',root/'protocol.json']:
            shutil.copyfile(p,dst/p.name)
    # Original Full retains its own version and provenance; copy actual stored evaluation.
    old=PROJECT/'revision_runs_20261006'
    oldroot=old/'runs/Historical_Full/evaluation';receipt=read_json(oldroot/'EVALUATION_RECEIPT.json')
    assert receipt['protocol']=='ACDC_v22_grid_directed_ASD' and sha(receipt['checkpoint'])==receipt['checkpoint_sha256']
    oldrows=read_table(oldroot/'case_class_metrics.csv');assert len(oldrows)==120
    for k in KEYS:assert abs(np.mean([float(r[k]) for r in oldrows])-receipt['metrics'][k])<1e-12
    rows.append(dict(run='Historical_Full_original90.35',status='EVALUATED; mechanism provenance NEEDS_MAPPING',checkpoint=receipt['checkpoint'],checkpoint_sha256=receipt['checkpoint_sha256'],selected_step=receipt['checkpoint_step'],cases=40,patients=20,unit='grid',**receipt['metrics']))
    for p in oldroot.glob('*'):
        if p.is_file() and p.suffix in ['.json','.csv']:
            dst=MAIN_WORK/'historical_supplements/old_Full';dst.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst/p.name)
    cost=[dict(task=h.get('task') or 'queue',stage=h['stage'],start=h['start'],end=h['end'],seconds=h['elapsed_seconds'],exit_code=h['exit_code'],scope='Completed server Popen stages before summary snapshot; imports/validation/IO included') for h in read_json(MAIN_WORK/'state/stage_history.json')]
    for t in ['E4','E5']:
        cp=MAIN_WORK/'tasks'/t/'reports/SHORT_CHECK.json'
        if cp.exists():
            for h in read_json(cp).get('history',[]):cost.append(dict(task=t,stage='ISOLATED_SHORT_UPDATE_'+str(h.get('target','NA')),start='NA: exact cold launch wall recorded as duration',end='NA',seconds=h['elapsed_seconds'],exit_code=h['exit_code'],scope='original 1+1 smoke process wall; successful intentional safe stop75; no formal updates; zero-probe/reload ancillary time not separately instrumented'))
    write_csv(MAIN_WORK/'results/experiment_results.csv',rows)
    for name,data in [('patient_metrics',patients),('class_metrics',classes),('case_class_metrics',caseclasses),('paired_differences',paired),('runtime_cost',cost)]:write_csv(MAIN_WORK/'results'/f'{name}.csv',data)
    text='本轮任务状态：'+json.dumps(tasks(),ensure_ascii=False)+'。\n\n'
    for t in ['E4','E5']:
        rr=next(r for r in rows if r['run']==t)
        if rr['status']!='EVALUATED':text+=f'{t} {rr["status"]}，指标NA；原因及日志见state/tasks.json和tasks/{t}/reports。\n\n';continue
        root=MAIN_WORK/'tasks'/t;d=read_table(root/'results/paired_summary.csv')[0];a=read_json(root/'reports/ANALYSIS_RECEIPT.json')
        text+=f'{t}完成30000联合迭代，自身student1严格验证选step{rr["selected_step"]}。Dice {float(rr["dice"])*100:.8f}%，IoU {float(rr["iou"])*100:.8f}%，HD95 {rr["hd95"]} grid，ASD {rr["asd"]} grid。R1−{t} Dice {float(d["R1_minus_current_dice_pp"]):+.8f} pp，IoU {float(d["R1_minus_current_iou_pp"]):+.8f} pp；距离差HD95 {float(d["R1_minus_current_hd95"]):+.8f}、ASD {float(d["R1_minus_current_asd"]):+.8f} grid（距离越低越好，负差表示R1更低）。20患者current−R1改善/持平/下降 {a["patient_counts"]["improved"]}/{a["patient_counts"]["tied"]}/{a["patient_counts"]["declined"]}；全部40帧×3类及全部患者保留。\n\n'
        if t=='E4':text+='该单次对照回答v2_B＋空间CTW设置中token条件化与已有参数接近普通残差适配层的表现。E4使用自身教师演化，实际结构与成本见MODEL_STRUCTURE_COMPARISON.csv。\n\n'
        else:text+='该单次对照回答共同数据、预算、选择和评价协议下官方AdaMix CT与R1的方法比较。官方随机初始化、AdamW、课程与增强保留；采样池/四路batch、清单和选择/评价适配见E5_PROTOCOL_COMPARISON.csv，发表原协议成绩另属其原实验。\n\n'
    text+='R1/R2和R1/C1分别保留两项条件对照结果，不构成完整混合×加权交互估计。旧Full90.35%继续原运行归属，来源映射NEEDS_MAPPING；ACDCtest未参与原候选方法选择。历史预训练命令和逐训练H5历史哈希NA。A2/A3/A5引用paper_offline_followup_20261007已核验版本；本轮只补新轨迹A1、病例类别A4、定性A6和成本。\n\n恢复主RNG与采样计划，worker随机流/预取不位级恢复；所有物理更新/失败/分段保留。E1继续3976暂停，原E2–E5未启动，E3未运行；本轮完成后停止，不衍生实验。\n'
    (MAIN_WORK/'FINAL_REPORT_E4_E5_CN.md').write_text(text)
    (MAIN_WORK/'WRITING_HANDOFF_E4_E5_CN.md').write_text('正式名称：TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。\n\n'+text+'\n主表精确原值results/experiment_results.csv；全病例/类别见results/*metrics.csv；配对正反差和pp见results/paired_differences.csv及tasks/E*/results/paired_*；距离grid，Dice/IoU fraction。图建议结构/方法表MODEL_STRUCTURE_COMPARISON与协议CSV、验证图A4_validation、全部患者A4_patient_Dice、新轨迹A1_control_and_loss、三例A6_*。定性病例/帧/类/z及选择规则见qualitative_selection.csv；图例RV蓝/MYO橙/LV红。五审各自状态与缺项见FIVE_PASS_REVIEW_E4_E5.md。源码/patch/commit/依赖/命令、初始化/训练/选择/评价回执实际入包，原数组与大权重见SERVER_ASSET_INDEX.csv。\n')
def reviews():
    summary();rounds=[]
    def check(n,fn,evidence):
        try:fn();rounds.append(dict(round=n,status='PASS',evidence=evidence))
        except BaseException:rounds.append(dict(round=n,status='FAIL',evidence=evidence,error=traceback.format_exc()))
    done=[t for t,s in tasks().items() if s['status']=='DONE']
    def r1():
        from preflight import verify_frozen
        verify_frozen()
        assert read_json(MAIN_WORK/'reports/STATIC_PREFLIGHT.json')['E3'] is False
        counts=read_table(MAIN_WORK/'MODEL_STRUCTURE_COMPARISON.csv');assert next(r for r in counts if r['task']=='E4')['parameters']=='1827475'
        assert read_json(MAIN_WORK/'reports/E4_FUNCTION_BINDING.json')['ctw_ast_identical']
        assert read_json(MAIN_WORK/'reports/E5_OFFICIAL_SOURCE.json')['GT_images']==12
        for t in done:
            root=MAIN_WORK/'tasks'/t;initial=read_json(root/'reports/INITIALIZATION_ACTUAL.json');assert initial['fresh'] and initial['joint_step']==0 and initial['code_identity']==task_identity(t)
            assert read_json(root/'reports/SHORT_CHECK.json')['status']=='PASS'
        assert sha(MAIN_WORK/'historical_supplements/C1/C1_PROTOCOL_COMPARISON.csv')==sha(PROJECT/'tcsmix_control_revision_20261006/formal_C1_CTW_uniform_20261008/C1_PROTOCOL_COMPARISON.csv')
    def r2():
        assert read_json(MAIN_WORK/'reports/ISOLATED_CONTROL_TEST.json')['status']=='PASS'
        assert read_json(MAIN_WORK/'reports/TMUX_ATTACH_DETACH.json')['status']=='PASS'
        for t in done:
            root=MAIN_WORK/'tasks'/t
            post=read_json(root/'reports/TRAIN_POSTCHECK.json');assert post['status']=='PASS' and post['logical_joint_steps']==30000
            assert read_json(root/'reports/FIRST_200_CHECKPOINT.json')['status']=='PASS'
            assert read_json(root/'reports/SHORT_CHECK.json')['physical_joint_updates']<=4
            assert read_json(root/'reports/SELECTED_CHECKPOINT.json')['test_metrics_consulted'] is False
            events=read_table(root/'results/optimizer_calls.csv');assert all(sum(r['optimizer']==n and r['event']=='success' for r in events)==post['optimizer_success_calls'][n] for n in ['optimizer1','optimizer2'])
    def r3():
        for t in done:
            root=MAIN_WORK/'tasks'/t;rows=read_table(root/'results/case_class_metrics.csv');patients=read_table(root/'results'/f'{t}_patient_metrics.csv');classes=read_table(root/'results'/f'{t}_class_metrics.csv');result=read_table(root/'results'/f'{t}_results.csv')[0]
            assert len(rows)==len({(r['case'],r['class_id']) for r in rows})==120 and len(patients)==20 and len(classes)==3
            assert {r['case'] for r in rows}==set((DATA/'test.list').read_text().splitlines())
            for k in KEYS:
                assert abs(np.mean([float(r[k]) for r in rows])-float(result[k]))<1e-12
                for p in patients:assert abs(np.mean([float(r[k]) for r in rows if r['patient']==p['patient']])-float(p[k]))<1e-12
                for c in classes:assert abs(np.mean([float(r[k]) for r in rows if r['class_name']==c['class_name']])-float(c[k]))<1e-12
            for pair in read_table(root/'results/paired_patients.csv'):
                for k in KEYS:
                    diff=float(pair['R1_'+k])-float(pair['current_'+k]);assert abs(diff-float(pair['R1_minus_current_'+k]))<1e-12 and abs(diff+float(pair['current_minus_R1_'+k]))<1e-12
                    if k in ['dice','iou']:assert abs(100*diff-float(pair['R1_minus_current_'+k+'_pp']))<1e-10
            assert len(read_table(root/'reports/INDEPENDENT_CASE_RECALC.csv'))==18
            er=read_json(root/'results/EVALUATION_RECEIPT.json');assert er['protocol']=='ACDC_v22_grid_directed_ASD' and er['postprocessing']=='none'
            assert sha(result['checkpoint'])==result['checkpoint_sha256']
    def r4():
        import matplotlib.image as mpimg
        for t in done:
            root=MAIN_WORK/'tasks'/t;a=read_json(root/'reports/ANALYSIS_RECEIPT.json');assert a['canvas_bounds_checked']
            assert all(sha(root/p)==h for p,h in a['outputs'].items())
            for r in read_table(root/'results/qualitative_selection.csv'):
                assert sum(int(r[k]) for k in ['GT_target_pixels','current_target_pixels','R1_target_pixels'])>0
                im=mpimg.imread(root/r['figure']);assert im.ndim==3 and np.isfinite(im).all() and im.std()>.01
        assert '90.35%' in (MAIN_WORK/'WRITING_HANDOFF_E4_E5_CN.md').read_text()
    check(1,r1,'protocol CSV; frozen scientific AST/binding; actual model counts; official commit/GT scope; C1 supplement')
    check(2,r2,'physical ledgers/full30k; all segments/failed preparation retained; initialization, real short resume, tmux detached progress, first200')
    check(3,r3,'complete40x3/20patients/3classes; independent aggregation/pp/directions; separate surface/EDT recomputation and hashes')
    check(4,r4,'predeclared examples incl decline; same coordinates/colors and visible targets; PNG rendering/readback and bounds; machine visual scope, human review NA')
    if len(done)<2:
        for r in rounds:
            if r['status']=='PASS':r['status']='PARTIAL';r['missing']='Only completed tasks reviewed; failed/blocked tasks retained with NA, no completion claimed'
    rounds.append(dict(round=5,status='PENDING',evidence='actual immutable archive extraction/internal SHA check executes next'))
    atomic_json(MAIN_WORK/'reports/FIVE_PASS_REVIEW.json',dict(time=now(),rounds=rounds,completed_tasks=done,failed_tasks={t:s for t,s in tasks().items() if t not in done}))
    reviewdoc(rounds);print('Reviews recorded',[(r['round'],r['status']) for r in rounds])
def reviewdoc(rounds):
    (MAIN_WORK/'FIVE_PASS_REVIEW_E4_E5.md').write_text('\n\n'.join(f'第{r["round"]}审：{r["status"]}。{r["evidence"]}。'+r.get('missing','')+('\n'+r['error'] if 'error' in r else '') for r in rounds)+'\n')
def snapshot(kind):
    stamp=now().replace(':','').replace('.','');snap=MAIN_WORK/'archives'/f'{kind}_snapshot_{stamp}';snap.mkdir()
    includes=[]
    for d in ['scripts','source','configs','manifests','reports','state','logs','analysis','results','tasks','historical_supplements','external','runs']:
        for p in (MAIN_WORK/d).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'predictions' in p.parts or p.suffix in ['.npz','.ckpt','.pth','.pyc'] or p.name.endswith('.lock') or '.tmp' in p.name:continue
            # Repository illustration/media are not necessary source dependencies.
            if d=='external' and p.suffix.lower() in ['.gif','.mp4','.pdf','.png','.jpg','.jpeg']:continue
            if p.name.startswith('ARCHIVE_') and p.suffix=='.log':continue
            includes.append(p)
    includes+=list(MAIN_WORK.glob('*.md'))+list(MAIN_WORK.glob('*.csv'))+[MAIN_WORK/'protocol.json']
    for p in sorted(set(includes)):
        dst=snap/p.relative_to(MAIN_WORK);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    assets=[]
    for d in ['tasks','runs']:
        for p in (MAIN_WORK/d).rglob('*'):
            if p.is_file() and p.suffix in ['.npz','.ckpt','.pth']:assets.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,packed=False,reason='large weights / complete medical predictions retained on server'))
    write_csv(snap/'SERVER_ASSET_INDEX.csv',assets)
    atomic_json(snap/'SNAPSHOT_BOUNDARY.json',dict(time=now(),kind=kind,scientific_training_closed=kind=='final',snapshot=str(snap),archive_self_hash='external receipt/sidecar; scientific files immutable',live_status_writes_after_snapshot_not_included=True))
    return snap
def build(snap,out,files=None):
    with tarfile.open(out,'w:gz',compresslevel=6) as tar:
        for p in files or sorted(snap.rglob('*')):
            if p.is_file():tar.add(p,arcname='E4_E5_evidence/'+str(p.relative_to(snap)),recursive=False)
def hashmanifest(snap):
    atomic_json(snap/'INTERNAL_SHA256.json',{str(p.relative_to(snap)):sha(p) for p in sorted(snap.rglob('*')) if p.is_file() and p.name!='INTERNAL_SHA256.json'})
def verify(out):
    with tempfile.TemporaryDirectory(prefix='E4_E5_extract_',dir=MAIN_WORK/'archives') as tmp:
        with tarfile.open(out) as tar:tar.extractall(tmp,filter='data')
        root=Path(tmp)/'E4_E5_evidence';m=read_json(root/'INTERNAL_SHA256.json')
        for p,h in m.items():assert sha(root/p)==h,('archive checksum mismatch',p)
        for name in ['E4_PROTOCOL_COMPARISON.csv','E5_PROTOCOL_COMPARISON.csv','MODEL_STRUCTURE_COMPARISON.csv','budget_and_differences.csv','historical_supplements/C1/C1_PROTOCOL_COMPARISON.csv','scripts/train_E4_entry.py','scripts/official_adapter.py','external/official_original/exp_acdc/train_adamix_ct.py']:assert (root/name).is_file()
        return len(m)
def package(kind='final'):
    snap=snapshot(kind);hashmanifest(snap);temp=MAIN_WORK/'archives'/f'{kind}_precheck_{time.time_ns()}.tar.gz';build(snap,temp);verify(temp)
    if kind=='final':
        r=read_json(MAIN_WORK/'reports/FIVE_PASS_REVIEW.json');r['rounds'][4].update(status='PASS',evidence='actual preliminary extraction SHA and required-files PASS; final archive extracted and SHA verified again; external receipt')
        atomic_json(MAIN_WORK/'reports/FIVE_PASS_REVIEW.json',r);reviewdoc(r['rounds'])
        for p in [MAIN_WORK/'reports/FIVE_PASS_REVIEW.json',MAIN_WORK/'FIVE_PASS_REVIEW_E4_E5.md']:shutil.copyfile(p,snap/p.relative_to(MAIN_WORK))
        hashmanifest(snap)
    out=MAIN_WORK/('E4_E5_final_evidence_20261008.tar.gz' if kind=='final' else 'E4_E5_start_evidence_20261008.tar.gz')
    if out.exists():shutil.move(out,MAIN_WORK/'archives'/f'previous_{out.name}_{time.time_ns()}')
    start=time.monotonic();build(snap,out);count=verify(out);h=sha(out);(MAIN_WORK/(out.name+'.sha256')).write_text(h+'  '+out.name+'\n')
    parts=[]
    if out.stat().st_size>30*2**20:
        groups=[];group=[];total=0
        for p in sorted(snap.rglob('*')):
            if not p.is_file():continue
            if group and total+p.stat().st_size>24*2**20:groups.append(group);group=[];total=0
            group.append(p);total+=p.stat().st_size
        if group:groups.append(group)
        for i,group in enumerate(groups):
            part=MAIN_WORK/f'E4_E5_{kind}_part_{i+1:02d}.tar.gz';build(snap,part,group);assert part.stat().st_size<30*2**20
            with tempfile.TemporaryDirectory(dir=MAIN_WORK/'archives') as tmpdir:
                with tarfile.open(part) as tar:tar.extractall(tmpdir,filter='data')
                for p in group:assert sha(Path(tmpdir)/'E4_E5_evidence'/p.relative_to(snap))==sha(p)
            ph=sha(part);(MAIN_WORK/(part.name+'.sha256')).write_text(ph+'  '+part.name+'\n');parts.append(dict(path=str(part),sha256=ph,bytes=part.stat().st_size))
    receipt=dict(status='VERIFIED',time=now(),path=str(out),sha256=h,bytes=out.stat().st_size,internal_files_verified=count,actual_extract=True,snapshot=str(snap),build_and_check_seconds=time.monotonic()-start,logical_parts=parts)
    atomic_json(MAIN_WORK/('ARCHIVE_RECEIPT.json' if kind=='final' else 'START_ARCHIVE_RECEIPT.json'),receipt)
    (MAIN_WORK/('DOWNLOAD_INDEX_CN.md' if kind=='final' else 'START_DOWNLOAD_INDEX_CN.md')).write_text(f'{out.name}；{out.stat().st_size/2**20:.2f}MiB；SHA256 {h}。\n已实际解包并核验内部{count}文件；数组/权重索引见SERVER_ASSET_INDEX.csv。\n'+''.join(f'\n{Path(p["path"]).name}；{p["bytes"]/2**20:.2f}MiB；{p["sha256"]}\n' for p in parts))
    print(kind,'archive VERIFIED',count,h)
def selftest():
    from analyze import pairs,independent_metrics
    from unified_evaluation import metrics
    a=np.zeros((4,8,8),bool);b=a.copy();a[1:3,2:5,2:5]=True;b[1:3,3:6,2:5]=True
    m=independent_metrics(a,b);n=metrics(a,b);assert all(abs(m[k]-n[k])<1e-12 for k in KEYS)
    c=[dict(patient='p',dice='.8',iou='.7',hd95='3',asd='1')];r=[dict(patient='p',dice='.9',iou='.8',hd95='2',asd='.5')];d=pairs(c,r,['patient'],'E4')[0]
    assert abs(d['R1_minus_current_dice_pp']-10)<1e-12 and d['R1_minus_current_hd95']==-1
    atomic_json(MAIN_WORK/'reports/POSTPROCESS_SELFTEST.json',dict(status='PASS',time=now(),surface_EDT_metrics=True,paired_direction_pp=True,training_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['review','package','start','selftest']);a=p.parse_args()
    {'review':reviews,'package':package,'start':lambda:package('start'),'selftest':selftest}[a.action]()
