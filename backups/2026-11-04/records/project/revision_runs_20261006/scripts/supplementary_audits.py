"""Read-only scientific audits; CPU-only follow-up and final evidence delivery.

Existing training code/configuration/controller are never edited.  --watch is
an ordinary tmux process which refreshes the new audit and final handoff after
the existing queue finishes its own analyses and packaging.
"""
import argparse
import ast
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from common import *

METRICS = ('dice', 'iou', 'hd95_mm', 'asd_mm')
AUDIT_STATE = WORK/'state/supplementary_audits.json'
MARKER = '\n## 本轮补充核查与论文更新\n'


def put(path, text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    temporary.write_text(text); os.replace(temporary,path)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def location(path, needle):
    path=Path(path)
    lines=path.read_text().splitlines()
    matches=[i+1 for i,line in enumerate(lines) if needle in line]
    return str(path)+':'+(str(matches[0]) if matches else 'NA')


def union_csv(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path,[{k:row.get(k,'NA') for k in fields} for row in rows],fields)


def lock_check():
    gate=read_json(WORK/'state/preflight.json')
    checked={}
    for kind in ('script_sha256','shell_sha256','source_sha256','config_sha256'):
        for rel,value in gate.get(kind,{}).items():
            assert sha(WORK/rel)==value, 'Existing locked artifact changed: '+rel
            checked[rel]=value
    assert gate['status']=='PASS' and gate['smoke_pass']
    return checked


def summary_stats(rows):
    out=dict(records=len(rows),distinct_steps=len({r['step'] for r in rows}),
             first_step=min(r['step'] for r in rows),last_step=max(r['step'] for r in rows))
    for key in ('q','s','area'):
        values=[r[key] for r in rows]
        out[key+'_min']=min(values); out[key+'_max']=max(values)
        out[key+'_mean']=sum(values)/len(values)
    out['s_rms']=math.sqrt(sum(r['s']**2 for r in rows)/len(rows))
    out['rectangle_97_fraction']=sum(r['rectangle_h']==97 and r['rectangle_w']==97 for r in rows)/len(rows)
    return out


def scale_audit():
    fixed=read_json(WORK/'configs/fixed_scale.json')
    path=FULL/'train.log'; raw=path.read_bytes()
    assert digest_bytes(raw)==fixed['source_sha256']
    historical=[]; excerpt=[]
    for number,line in enumerate(raw.decode().splitlines(),1):
        match=re.search(r'iteration (\d+) : adamix_endpoint=(\{.*\})',line)
        if not match: continue
        data=json.loads(match[2]); side=float(data['side_fraction'])
        h=max(1,min(255,round(256*side)))
        historical.append(dict(step=int(match[1]),q=float(data['teacher_conf']),s=side,
                    area=float(data['patch_area_ratio']),rectangle_h=h,rectangle_w=h,
                    log_line=number,batch_slice_ids='NA: not logged',teacher_checkpoint_at_forward='NA: not retained per step'))
        excerpt.append(f'original_line={number} {line}')
    assert len(historical)==150
    assert [r['step'] for r in historical]==list(range(200,30001,200))
    assert all(abs(r['s']-(.18+(.38-.18)*r['q']))<1e-10 for r in historical)
    assert all(r['area']==r['rectangle_h']*r['rectangle_w']/65536 for r in historical)
    stats=summary_stats(historical)
    assert fixed['observed_mean_actual_area']==stats['area_mean']
    error=abs(fixed['fixed_area']-stats['area_mean'])
    assert fixed['rectangle']==[97,97] and fixed['fixed_side']==97/256
    assert error==fixed['absolute_area_error']==0
    for task in ('E2','E3'):
        _,cfg=config_for(task)
        assert cfg['fixed_side']==fixed['fixed_side'] and cfg['mix_mode']=='fixed'
    union_csv(WORK/'reports/fixed_rectangle_historical_records.csv',historical)
    put(WORK/'reports/fixed_rectangle_historical_excerpt.log',
        'source='+str(path)+'\nsha256='+digest_bytes(raw)+'\n'+'\n'.join(excerpt)+'\n')
    new=[]; partial=0
    trace=WORK/'runs/E1/audit/revision_trace.csv'
    if trace.exists():
        captured=trace.read_bytes()
        put(WORK/'analysis/A1/E1_schedule_trace_capture.csv',captured.decode())
        assert sha(WORK/'analysis/A1/E1_schedule_trace_capture.csv')==digest_bytes(captured)
        for row in csv.DictReader(io.StringIO(captured.decode())):
            try:
                side=float(row['s']); h=max(1,min(255,round(side*256)))
                item=dict(step=int(row['step']),q=float(row['q']),s=side,
                          area=float(row['actual_area']),rectangle_h=h,rectangle_w=h)
                assert abs(item['area']-h*h/65536)<1e-12
                new.append(item)
            except (TypeError,ValueError,KeyError): partial+=1
        union_csv(WORK/'analysis/A1/E1_schedule_audit_snapshot.csv',new)
    e1=summary_stats(new) if new else None
    reference=[]
    for name,step in [('best.ckpt',29600),('last.ckpt',30000)]:
        cp=FULL/'checkpoints'/name
        reference.append(dict(path=str(cp),sha256=sha(cp),step=step,used_to_estimate_fixed_scale=False))
    result=dict(status='VERIFIED_LOCK_UNCHANGED',time=now(),source_type='real historical online training endpoint records',
                source=str(path),source_sha256=digest_bytes(raw),run=str(FULL),historical=stats,
                checkpoint_references=reference,offline_teacher_estimation_used=False,
                batch='original B24=12 labeled+12 unlabeled; q on two weak unlabeled groups of6, all256x256 pixels',
                batch_ids='NA: history does not record sampled slice IDs or augmentation/worker states',
                teacher_timing='q computed before the current student update and teacher EMA update; post-step checkpoints are not that exact forward teacher',
                fixed_side=fixed['fixed_side'],fixed_rectangle=[97,97],fixed_area=fixed['fixed_area'],
                absolute_area_error=error,relative_area_error=error/stats['area_mean'],
                error_percentage_points=100*error,coverage='150 points step200..30000 every200; mean at recorded points only, not all30000 steps',
                E1=e1,E1_source=str(trace),E1_capture_sha256=digest_bytes(captured) if trace.exists() else 'NA',
                E1_incomplete_tail_records_ignored=partial,configuration_changed=False)
    atomic_json(WORK/'reports/fixed_rectangle_audit.json',result)
    table='|来源|n|step范围|s最小|s最大|s均值|平均实际面积|97×97比例|\n|---|---:|---|---:|---:|---:|---:|---:|\n'
    for title,d in [('旧Full每200步真实记录',stats),('新增E1逐step记录',e1)]:
        if d: table+=f"|{title}|{d['records']}|{d['first_step']}–{d['last_step']}|{d['s_min']:.12f}|{d['s_max']:.12f}|{d['s_mean']:.12f}|{100*d['area_mean']:.9f}%|{100*d['rectangle_97_fraction']:.6f}%|\n"
    text=f"""# 固定矩形来源审计

更新时间{now()}。结论：97×97匹配旧Full已记录150个时点的平均实际交换面积，绝对误差0、相对误差0、面积百分点误差0。配置保持锁定，未发现需要暂停E2/E3的来源或计算错误。不是旧Full全部30000步的均值估计。

{table}

真实来源：{path}；SHA256 {digest_bytes(raw)}；run {FULL}。采样点step200、400、…、30000（150条），不是通过教师检查点离线重推。全部原行及行号在reports/fixed_rectangle_historical_excerpt.log，数值在reports/fixed_rectangle_historical_records.csv。

历史q范围{stats['q_min']:.12f}–{stats['q_max']:.12f}、均值{stats['q_mean']:.12f}。s的RMS={stats['s_rms']:.12f}。s=.18+.20q，Python round(256s)并clamp[1,255]。映射端点.18/.38对应可取整边长46/97；97确为这一定义的最大边长，但q高不证明前景可靠，q还包含大量背景。97取整所需q约≥0.984765625。

固定比例97/256=.37890625，固定面积9409/65536=.1435699462890625。选择按实际取整矩形面积的最近可表示值，不按mean(s)直接定边长。忽略取整时RMS边长只是辅助；未读取测试成绩或重训Full。

原batch24含12标注、12无标注；两个6例无标注weak视图的teacher最大类别概率分别平均后取均值，单个矩形共享batch/两视图，原随机位置np.random.randint(0,max(1,256-patch_size))。位置、真实切片ID和增强随机状态没有逐batch日志，因此采样批次ID为NA；seed42不能补回失落的worker随机流。

checkpoint仅作历史run身份核验：best step29600和last step30000，路径/hash见fixed_rectangle_audit.json。它们未用于估计固定尺度；中间teacher快照未逐时点保留。q是在该step EMA更新之前计算，不能把post-step checkpoint冒充同一前向teacher。

E1的真实逐步日志来自{trace}，覆盖范围见表；它是新的均匀CTW运行，teacher轨迹也可能不同，不能冒充旧Full历史，也不据E1面积重新选E2/E3尺度。重复恢复造成的重复step将按真实执行记录保留，并报告distinct_steps。刷新时忽略的未完整尾行={partial}。图analysis/A1/supplement_schedule.png及SVG按两列分开展示，旧Full仅画散点，不插值补出失落历史。
"""
    put(WORK/'FIXED_RECTANGLE_AUDIT.md',text)
    # Real data plots only; no generated illustrative assets.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none','font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(3,2,figsize=(10,7),layout='constrained')
    for column,(title,data,color) in enumerate([('Historical Full: every200 updates',historical,'#0072B2'),('New E1: logged updates',new,'#D55E00')]):
        for i,key in enumerate(('q','s','area')):
            ax=axes[i,column]
            if data:
                x=[r['step'] for r in data]; y=[r[key] for r in data]
                if column==0: ax.scatter(x,y,s=9,color=color)
                else: ax.plot(x,y,lw=.8,color=color)
            ax.set_ylabel({'q':'Teacher confidence q','s':'Side fraction s','area':'Actual exchanged area'}[key])
            ax.set_xlabel('Recorded update'); ax.grid(alpha=.15)
        axes[0,column].set_title(title,fontsize=10)
    for extension in ('png','svg'): fig.savefig(WORK/f'analysis/A1/supplement_schedule.{extension}',dpi=180)
    plt.close(fig)
    atomic_json(WORK/'analysis/A1/supplement_schedule.provenance.json',dict(time=now(),
        historical_source=str(path),historical_source_sha256=digest_bytes(raw),
        E1_snapshot='analysis/A1/E1_schedule_trace_capture.csv',E1_snapshot_sha256=result['E1_capture_sha256'],
        E1_derived_records='analysis/A1/E1_schedule_audit_snapshot.csv',
        E1_derived_records_sha256=sha(WORK/'analysis/A1/E1_schedule_audit_snapshot.csv'),
        separation='old Full sparse scatter and new E1 separate panels; no historical interpolation',
        historical_stats=stats,E1_stats=e1,SVG_native_text=True,AI_generated_assets=False))
    return result


def comparability_audit():
    gate_artifacts=lock_check()
    protocol=read_json(WORK/'protocol.json'); manifest=read_json(WORK/'run_manifest.json')
    source=(WORK/'source/historical_full.py').read_text()
    assert sha(FULL/'effective_source.py')==sha(WORK/'source/historical_full.py')
    from train_entry import build_beta_source
    generated={t:build_beta_source(config_for(t)[1]) for t in ('E1','E2','E3','E4')}
    assert len({digest_bytes(v.encode()) for v in generated.values()})==1
    actual=WORK/'runs/E1/revision_effective_source.py'
    if actual.exists(): assert actual.read_text()==generated['E1']
    expected={'E1':('uniform','dynamic','token'),'E2':('spatial','fixed','token'),
              'E3':('uniform','fixed','token'),'E4':('spatial','dynamic','bottleneck')}
    # The actual token cfg name is recorded; only its equivalence is needed.
    for task,(weight,mix,trc) in expected.items():
        _,cfg=config_for(task)
        assert cfg['weight_mode']==weight and cfg['mix_mode']==mix
        assert (cfg['trc']=='bottleneck')==(trc=='bottleneck')
        assert cfg['max_iterations']==30000 and cfg['batch_size']==24 and cfg['labeled_bs']==12
        assert cfg['seed']==42 and cfg['num_workers']==16 and cfg['validation_interval']==200
        assert cfg['pretrained_model']==protocol['pretrain']
    assert sha(protocol['pretrain'])==protocol['pretrain_sha256']
    evidence=WORK/'reports/supplementary_sources'
    evidence.mkdir(parents=True,exist_ok=True)
    ids=(DATA/'train_slices.list').read_text().splitlines()
    for split in ('train_slices','val','test'):
        put(evidence/f'{split}.list.txt',(DATA/f'{split}.list').read_text())
    labeled=ids[:136]
    union_csv(WORK/'reports/labeled_slice_identity.csv',[dict(index=i,patient=x.split('_')[0],slice_id=x) for i,x in enumerate(labeled)])
    labeled_hash=digest_bytes(('\n'.join(labeled)+'\n').encode())
    assert sorted({x.split('_')[0] for x in labeled})==protocol['labeled_patients']
    full_cp=FULL/'checkpoints/best.ckpt'
    full_cp_sha256=sha(full_cp)
    rows=[]
    loop=location(WORK/'source/historical_full.py','def train():')
    runtime=location(WORK/'scripts/revision_runtime.py','def ctw_maps')
    def add(task,aspect,before,after,kind='保持一致',note='',where=loop):
        rows.append(dict(experiment=task,aspect=aspect,classification=kind,full_value=before,control_value=after,
            full_run=str(FULL),full_checkpoint=str(full_cp),full_checkpoint_sha256=full_cp_sha256,
            control_run=str(WORK/'runs'/task),control_status=manifest['tasks'][task]['status'],
            evidence=where,note=note))
    losses={
        'mixed CE/Dice':'each student: .5*(two mixed-region Dice terms + two mixed-region CE terms), GTweight1/pseudoweight.5',
        'mixed/CPS':'student1 Dice(p1,detach argmax p2) + student2 Dice(p2,detach argmax p1), coefficient1 each',
        'FFT':'same paired mixed inputs; low-frequency spatial ifft(abs) mixing threshold30; cross-student pseudo-Dice coefficient1 each',
        'supervised weak/strong':'student1 each view .5*(CE+Dice), labels from12 labeled slices; strong cutout labels0',
        'ICT weak/strong':'student1 MSE to detached interpolated EMA probabilities; separate Beta(.1,.1) mixtures in both views',
        'ICT coefficient':'5*consistency(.1)*sigmoid_rampup(step//150,200), maximum .5 (not .1)',
        'total loss':'loss1+loss2+weak_supervised+strong_supervised+weak_ICT+strong_ICT+.05*CTW',
    }
    for task in ('E1','E2','E3','E4'):
        cfgpath,cfg=config_for(task)
        for key,value in [
            ('dataset root',str(DATA)),('current split hashes',json.dumps({s:sha(DATA/f'{s}.list') for s in ('train_slices','val','test')})),
            ('label cases/slices',json.dumps(dict(patients=protocol['labeled_patients'],slices=136,labeled_list_sha256=labeled_hash))),
            ('initialization file',protocol['pretrain']+' sha256='+protocol['pretrain_sha256']),
            ('initialization state','load same net and transplant same opt; new residual parameters retain their prescribed initializer'),
            ('base U-Net','same 1input/4classes U-Net[16,32,64,128,256]'),
            ('batch/input','B24=12 labeled+12 unlabeled,256x256; each group split6+6'),
            ('augmentation','same frozen WeakStrongAugment 2D; shared rot/flip/rotate and nearest resize; strong cutout(p.5,area.02-.4,ratio.3..1/.3) and ColorJitter(.8,.8,.8,.2)'),
            ('optimizer/LR','SGD .01,momentum.9,wd1e-4; group0 pretrain opt transplanted; constant LR; separate residual optimizer group'),
            ('seed/workers','seed42,deterministic=True,benchmark=False;16workers; same fresh sampler order algorithm'),
            ('SSL updates','30000 single run'),('precision','FP32; no AMP'),
            ('teacher targets','EMA forward no-grad; RV/MYO/LV hard-label LCC; separate strong targets; all mixed pixels incl cutout enter CPS/CTW'),
            ('teacher EMA','alpha.99; student1 state_dict, including residual layer and buffers'),
            ('validation','every200updates,split=val,student1 foreground macroDice; strict greater-than best; legacy duplicate val calls retained'),
            ('final evaluator','ACDC_v22_grid_directed_ASD; same unified evaluator as frozen Full reevaluation'),
            ('final units/postprocess','H5 original reconstruction grid; HD95 bidirectional95pct; ASD prediction-to-GT directed; no prediction LCC'),
            ('final empty policy','both empty all0; one empty Dice/IoU0 and surface distance volume-grid diagonal'),
            ('final averaging','equal RV/MYO/LV × frame volumes; A4 first combines frames within20 independent patients'),
            ('CTW coefficient/input/gradient','.05*mean(w*mean_class(abs(softmax(out1)-detach softmax(out2)))); weights detached; gradient onlystudent1'),
            ('mixed pairing/position','same reciprocal regions, image and target masks, shared rectangle/paired geometry; original randint excludes final possible start'),
        ]: add(task,key,value,value)
        for key,value in losses.items(): add(task,key,value,value)
        weight='per-image detached mean(w) broadcast over all mixed pixels' if task in ('E1','E3') else 'original spatial w=.25+minmax(backward absdifferences of detached mean max-prob)'
        add(task,'CTW spatial allocation','original spatial w=.25+r',weight,'预期改变' if task in ('E1','E3') else '保持一致',
            'E1/E3 share exactly revision_runtime.ctw_maps; preserve probability L1, denominator floor1e-6 and all participating pixels',runtime)
        scale='locked s97/256→97x97' if task in ('E2','E3') else 'original s=.18+.2*q'
        add(task,'TCSMix scale','s=.18+.2*q with round/clamp',scale,'预期改变' if task in ('E2','E3') else '保持一致',
            'E2/E3 share same function and fixed config value; log teacher q still measured',location(WORK/'scripts/revision_runtime.py','def generate_adamix'))
        add(task,'TRC residual layer','4tokens,dim16,641parameters,zero gamma',
            '1x1(16→18)/LN/GELU/1x1(18→16),647parameters,zero gamma' if task=='E4' else '4tokens,dim16,641parameters,zero gamma',
            '预期改变' if task=='E4' else '保持一致',
            'E4 replaces all two students + corresponding EMA; +6/student,+.936% residual params; no tokens/prototypes/attention',
            location(WORK/'scripts/revision_runtime.py','class BottleneckResidual'))
        add(task,'pretrain generation provenance','actual command/seed/updates NA','same known file reused; generation command/seed/updates still NA','无法核验',
            'pre_iterations=10000 is an argparse default, not evidence of how this file was generated')
        add(task,'historical train list immutable hash','NA at historical training date','current lists hashed and locked','无法核验',
            'historical namespace proves same path/first136 selection; no original-time train-list digest located; present hashes do not prove past bytes')
        add(task,'historical literal shell launch command','NA: original shell argv not located',
            shlex.join(manifest['tasks'][task].get('actual_command',manifest['tasks'][task]['command'])),'无法核验',
            'Historical parsed Namespace is available and preserved in protocol.json/train.log; it is not a surviving literal shell command')
        add(task,'historical imported dependencies/runtime','original-time full dependency hashes/torch-CUDA version NA',
            '32 frozen Python sources; current beta_fft environment; existing preflight hash gate intact','无法核验',
            'historical effective_source is preserved, but imported-module versions were copied for this revision; no original git history to prove every old dependency byte')
        add(task,'durability/diagnostics','historical direct save and batch-offset restart',
            'atomic save + after-val metadata refresh, full sampler IDs/RNG, safe stop, per-step detached audit','其他差异',
            'fresh sampler invokes exact parent algorithm; getters/logging draw no randomness. Worker augmentation/prefetch resume is not exact. Structural changes can change stochastic realization despite same seed.')
        add(task,'code/config/command',str(FULL/'effective_source.py')+' sha='+digest_bytes(source.encode())+'; historical parsed namespace in protocol.json',
            json.dumps(dict(config=str(cfgpath),config_sha256=sha(cfgpath),effective_source_sha256=digest_bytes(generated[task].encode()),
                runtime=str(WORK/'scripts/revision_runtime.py'),runtime_sha256=sha(WORK/'scripts/revision_runtime.py'),
                command=shlex.join(manifest['tasks'][task].get('actual_command',manifest['tasks'][task]['command']))),ensure_ascii=False),'其他差异',
            'separate output/entrypoint; shared frozen loop derivative. Pending controls are planned, not claimed already executed.')
        put(evidence/f'{task}_planned_effective_source.py',generated[task])
    e5path,e5=config_for('E5'); official=WORK/'external/AdaMix'
    commit=subprocess.run(['git','-C',str(official),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    assert commit==e5['official_commit']
    e5info=dict(status=manifest['tasks']['E5']['status'],repository='https://github.com/Senyh/AdaMix',commit=commit,
        source_recipe=str(official/'exp_acdc/train_adamix_ct.py'),source_sha256=sha(official/'exp_acdc/train_adamix_ct.py'),
        actual_entry=str(WORK/'scripts/train_entry.py')+' → official_adapter.run_official',
        adapter=str(WORK/'scripts/official_adapter.py'),adapter_sha256=sha(WORK/'scripts/official_adapter.py'),
        complete_command=shlex.join(manifest['tasks']['E5'].get('actual_command',manifest['tasks']['E5']['command'])),
        upstream_main_executed_directly=False,literal_upstream_training_step_compiled=True,
        budget='30000updates,scratch,no pretrain; native four loaders B16 each',
        differences='official native U-Net/two students(no EMA),AdamW1e-4 wd.01,DSCLossH(confidence.95),four native AdaMix modules patch8/topk16/hard/p.5,age curriculum7500; no betaFFT/TRC/CTW; upstream200epochs adapted to30000updates, val-loss selection changed to common every200 val student1 macroDice; same7cases136slices; retain official PIL quantization/augmentation')
    atomic_json(WORK/'reports/E5_OFFICIAL_METHOD_AUDIT.json',e5info)
    for aspect,value in e5info.items(): add('E5',aspect,'Full β-FFT protocol',str(value),'其他差异','E5 is a pinned official-method implementation with explicitly described orchestration adaptation; not an unchanged upstream CLI run')
    union_csv(WORK/'protocol_comparison.csv',rows)
    counts={t:{kind:sum(r['experiment']==t and r['classification']==kind for r in rows) for kind in ('保持一致','预期改变','其他差异','无法核验')} for t in expected}
    atomic_json(WORK/'reports/comparability_checks.json',dict(status='PASS_WITH_HISTORICAL_PROVENANCE_GAPS',time=now(),
        locked_existing_artifacts=len(gate_artifacts),expected_controls_verified=True,
        generated_loop_sources_equal=True,actual_E1_loop_matches=True,config_changed=False,
        no_unexpected_scientific_change_found_in_available_source=True,
        cannot_prove_original_time_list_or_imported_dependency_hashes=True,counts=counts))
    put(WORK/'COMPARABILITY_AUDIT.md',f"""# 原Full与新增对照可比性核查

更新时间{now()}。结论：可用配置与冻结循环中未发现E1–E4的额外科学变量改变；预期改变分别为E1均匀CTW、E2固定尺度、E3二者组合、E4普通残差层。不能宣称所有历史依赖、数据字节或恢复随机流都已证明完全一致。

protocol_comparison.csv逐项使用保持一致、预期改变、其他差异、无法核验四类，含真实run/checkpoint/hash、配置与完整队列命令。报告当前训练状态以state/queue.json为准。四个生成的循环源码hash相同，控制变量通过同一revision_runtime和独立cfg注入；实际E1源码与生成源码相同。原有{len(gate_artifacts)}个预检锁定文件哈希仍全部匹配，未修改训练入口、控制器或五项配置。

标注七病例：{', '.join(protocol['labeled_patients'])}，首136切片列表SHA256 {labeled_hash}。当前train/val/test列表及逐slice标签身份实际装入reports/supplementary_sources和labeled_slice_identity.csv。旧Full日志记录同路径、labelnum7与对应136规则，但历史训练当时的清单hash未保存；不能把当前hash冒充历史hash。

初始化文件{protocol['pretrain']}，SHA256 {protocol['pretrain_sha256']}，相同net/opt复用；预训练实际生成命令、seed、步数仍NA。历史pre_iterations=10000只是默认参数，不能补成已证实预算。E4新增普通层不从旧token权重初始化，其零gamma保证初始接口恒等。

逐loss定义在CSV并保留源码：双学生各自mixed CE/Dice、两向CPS、两向FFT pseudo-Dice；student1 weak/strong supervised与ICT；CTW仅student1。特别补清ICT有效权重=5×0.1×sigmoid_rampup(step//150,200)，最大0.5，而不是把args.consistency=.1写成有效最大权重。loss1+loss2+两视图supervised+两视图ICT+.05CTW表达式保持一致。CTW仍为类别均值概率L1、student2 detach、q/w detached、后向差分、逐图minmax下限1e-6，全部mixed像素参与。

E1与E3调用同一个逐图mean(w)广播实现；E2与E3调用同一个固定矩形实现和同一s；E4三模型同接口替换，647vs641，差6参数/部署student、18参数/三模型。所计残差MAC/pixel576vs640，差−10%，不含LN/激活/softmax；部署总参数差约0.000328%，不能把残差层百分比写成全网络百分比。

EMA=.99来自student1的整个state_dict，包括对应残差参数和buffers。B24/12标签、256²、SGD.01/momentum.9/wd1e-4、恢复预训练optimizer状态、恒定LR、30000SSLupdates、每200step验证student1前景macroDice严格提高选best保持原设置。历史名testloader实际仍读取val，本轮也保留重复val行为；ACDC测试不参与候选选择，测试只在best冻结后最终评价。

最终评价与原Full独立重评价采用同一evaluate.py，nearest输入/输出恢复H5原frame体积，无预测LCC；HD95为双向95百分位、ASD为prediction→GT单向，单位grid；both-empty全0、one-empty体积网格对角线。case/frame与class等权，A4另聚合为20独立患者。Full历史指标已精确重现，但这不证明缺失的历史训练依赖版本。

其他工程差异明确保留：原子checkpoint、val后刷新last元数据、sampler IDs/RNG、安全stop与detached逐step诊断。新鲜运行采样仍调用原parent iterator；诊断读取RNG而不抽取新随机数。resume的16worker增强随机流/预取不能完整恢复，不宣称位级精确。不同结构即使同seed也不保证相同dropout随机实现。原历史effective_source已冻结；历史所有import依赖的当时hash和运行环境版本未全部保存，无法核验，未额外重训Full。

E5单独核查：repo {e5info['repository']}，commit {commit}；方法配方来源{e5info['source_recipe']}，实际执行{e5info['actual_entry']}。完整命令：

    {e5info['complete_command']}

当前{e5info['status']}。执行的是官方核心训练step/网络/损失/AdaMix的同协议适配，未直接运行原main；差异为{e5info['differences']}。详情reports/E5_OFFICIAL_METHOD_AUDIT.json和official_adapter.py，不能写成未经修改的官方命令原样复跑。
""")
    return counts


def covid_audit():
    import numpy as np
    import nibabel as nib
    from scipy.ndimage import zoom
    evaluator=WORK/'source/train_COVID19/covid19_volume_evaluator.py'
    assert sha(evaluator)==sha(PROJECT/'train_COVID19/covid19_volume_evaluator.py')
    spec=importlib.util.spec_from_file_location('revision_spacing_evaluator',evaluator)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    manifest=PROJECT/'data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv'
    old_geometry={r['patient']:r for r in read_table(WORK/'reports/covid_geometry_audit.csv')}
    groups={}
    for row in read_table(manifest): groups.setdefault(row['patient_id'],[]).append(row)
    volumes={}; geometry=[]; slice_sources=[]
    for patient,items in sorted(groups.items()):
        items.sort(key=lambda r:int(r['slice_idx']))
        assert [int(r['slice_idx']) for r in items]==list(range(len(items)))
        image_path=PROJECT/items[0]['image_path']; gt_path=PROJECT/items[0]['mask_path']
        raw_image=nib.load(image_path); raw_gt=nib.load(gt_path)
        image=nib.as_closest_canonical(raw_image); native=nib.as_closest_canonical(raw_gt)
        assert image.shape==native.shape and np.allclose(image.affine,native.affine)
        assert sha(gt_path)==old_geometry[patient]['native_gt_sha256']
        native_gt=(np.asarray(native.dataobj)>0)
        targets=[]; spacing=None
        for row in items:
            p=Path(row['derived_path']); p=p if p.is_absolute() else PROJECT/p
            with np.load(p) as payload:
                target=payload['mask']>0; current=tuple(float(x) for x in payload['spacing'])
            if spacing is None: spacing=current
            assert current==spacing and target.shape==(256,256)
            expected=zoom(native_gt[:,:,int(row['slice_idx'])],(256/native.shape[0],256/native.shape[1]),order=0)
            assert np.array_equal(target,expected), 'Derived GT differs from native nearest mapping: '+row['sample_id']
            targets.append(target)
            slice_sources.append(dict(patient=patient,slice_idx=row['slice_idx'],derived_npz=str(p),sha256=sha(p),gt_key='mask',original_gt=str(gt_path)))
        gt=np.stack(targets); sx,sy,sz=(float(x) for x in image.header.get_zooms()[:3])
        assert np.allclose(spacing,(sx,sy,sz)) and gt.shape[0]==native.shape[2]
        corrected=(sx*(native.shape[0]-1)/255,sy*(native.shape[1]-1)/255,sz)
        g=dict(patient=patient,original_image=str(image_path),original_gt=str(gt_path),original_gt_sha256=sha(gt_path),
            original_image_shape=list(raw_image.shape),original_GT_shape=list(raw_gt.shape),canonical_shape_XYZ=list(image.shape),
            evaluation_shape_XYZ=[256,256,len(items)],prediction_stored_shape_ZXY=list(gt.shape),
            original_spacing_header=json.dumps([float(x) for x in raw_image.header.get_zooms()[:3]]),
            canonical_original_spacing_XYZ=json.dumps([sx,sy,sz]),old_spacing_XYZ=json.dumps(spacing),
            spacing_only_corrected_XYZ=json.dumps(corrected),spacing_only_corrected_ZXY=json.dumps((sz,corrected[0],corrected[1])),
            native_spacing_ZXY=json.dumps((sz,sx,sy)),original_orientation=str(nib.aff2axcodes(raw_image.affine)),
            canonical_orientation=str(nib.aff2axcodes(image.affine)),canonical_image_GT_affine_equal=True,
            Z_slices_unchanged=True,derived_GT_exactly_native_nearest=True,
            derived_GT_array_sha256=digest_bytes(np.ascontiguousarray(gt).tobytes()),
            native_GT_array_sha256_ZXY=digest_bytes(np.ascontiguousarray(native_gt.transpose(2,0,1)).tobytes()))
        geometry.append(g); volumes[patient]=dict(gt=gt,legacy_spacing=spacing,corrected_spacing=corrected,geometry=g)
    union_csv(WORK/'reports/covid_spacing_geometry.csv',geometry)
    union_csv(WORK/'reports/covid_derived_slice_sources.csv',slice_sources)
    registry=[]
    for folder in (PROJECT/'formal_test/covid19',PROJECT/'formal_test/covid19_seed1337',PROJECT/'formal_test/cross_dataset_ablation_v19_8/covid19'):
        for path in sorted(folder.glob('*/FORMAL_TEST_RECEIPT.json')):
            receipt=read_json(path); match=re.search(r'(?:COVID19_)(5|10)P',str(path))
            if not match: continue
            token=receipt.get('mask_token',False); mix=receipt.get('adamix',False)
            method='TRC+TCSMix' if token and mix else 'TRC' if token else 'TCSMix' if mix else 'baseline'
            registry.append(dict(run=path.parent.name,method=method,ratio=match[1]+'%',
                seed=receipt.get('formal_training_seed','NA'),receipt=str(path),receipt_sha256=sha(path),
                checkpoint=str(Path(receipt['checkpoint_path']) if Path(receipt['checkpoint_path']).is_absolute() else PROJECT/receipt['checkpoint_path']),
                checkpoint_sha256=receipt['checkpoint_sha256'],predictions=receipt['prediction_paths'],
                original_metrics=receipt['metric_result']['per_patient'],source_evaluator_sha256=receipt.get('evaluator_sha256','NA'),
                cohort='canonical1337' if 'CANON1337' in path.parent.name else 'older run, separately retained'))
    for percent in (5,10):
        path=PROJECT/f'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID{percent}_FINAL3/TEST_RESULT_FULL_PRECISION.json'
        receipt=read_json(path)
        registry.append(dict(run=f'COVID{percent}_FINAL3',method='Full',ratio=f'{percent}%',seed=1337,receipt=str(path),receipt_sha256=sha(path),
            checkpoint=str(PROJECT/receipt['CheckpointPath']),checkpoint_sha256=receipt['CheckpointSHA'],
            predictions=[dict(patient_id=p,path=str(WORK/f'runs/COVID{percent}_Full/evaluation/predictions/{p}.npz')) for p in volumes],
            original_metrics=read_json(path.parent/'patient_metrics.json')['per_patient'],source_evaluator_sha256=receipt['EvaluatorSHA'],
            seed_source=str(PROJECT/f'configs/three_innovation_ablation_transfer_v22_0/COVID{percent}_FINAL3.json'),cohort='canonical1337'))
    cache={(r['prediction'],r['patient'],r['policy']):r for r in read_table(WORK/'reports/covid_existing_prediction_reevaluation.csv')}
    initial_script=WORK/'reports/supplementary_sources/supplementary_audits_initial_version.py'
    results=[]; macros=[]; input_failures=[]
    for run in registry:
        original_receipt=Path(run['receipt'])
        put(WORK/'reports/supplementary_sources/covid_receipts'/f'{run["run"]}.json',original_receipt.read_text())
        if run.get('seed_source'):
            put(WORK/'reports/supplementary_sources/covid_receipts'/f'{run["run"]}_seed_config.json',Path(run['seed_source']).read_text())
        for item in run['predictions']:
            patient=item['patient_id']; v=volumes[patient]; g=v['geometry']
            prediction=Path(item['path']); prediction=prediction if prediction.is_absolute() else PROJECT/prediction
            identity=dict(method=run['method'],ratio=run['ratio'],run=run['run'],seed=run['seed'],cohort=run['cohort'],
                patient=patient,receipt=run['receipt'],receipt_sha256=run['receipt_sha256'],
                prediction=str(prediction),GT_source=g['original_gt'],derived_GT_sources=str(WORK/'reports/covid_derived_slice_sources.csv'),
                checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256'],evaluator=str(evaluator),evaluator_sha256=sha(evaluator))
            try:
                prediction_sha=sha(prediction)
                if item.get('output_sha256'): assert prediction_sha==item['output_sha256']
                with np.load(prediction) as payload: pred=payload['prediction']>0
                assert pred.shape==v['gt'].shape
                identity.update(prediction_sha256=prediction_sha,GT_sha256=g['original_gt_sha256'],
                    before_prediction_shape=json.dumps(list(pred.shape)),before_GT_shape=json.dumps(list(v['gt'].shape)),
                    original_shape_XYZ=json.dumps(g['canonical_shape_XYZ']),old_evaluation_shape_XYZ=json.dumps(g['evaluation_shape_XYZ']),
                    original_spacing_XYZ=g['canonical_original_spacing_XYZ'],old_evaluation_spacing_XYZ=g['old_spacing_XYZ'],
                    stored_array_order='ZXY',evaluation_array_order='XYZ',spacing_axis_mapping='(sx,sy,sz) matches XYZ; (sz,sx,sy) matches ZXY',
                    postprocess='threshold/argmax saved categorical prediction, binary>0; no new LCC/TTA/ensemble',
                    empty_policy='both empty Dice/IoU1,HD95/ASD0; one empty Dice/IoU0,HD95/ASDinf',
                    aggregation='3D whole patient metrics, macro equal4patients; ASD mean of concatenated two-direction surface samples',
                    resize_code=location(PROJECT/'train_COVID19/covid19_official20_v2.py','def resize_slice'),
                    reconstruction_code=location(evaluator,'    pred_vol = np.stack'),
                    metric_code=location(evaluator,'def binary_volume_metrics'))
                original=run['original_metrics'][patient]
                reusable=cache.get((str(prediction),patient,'legacy_replay'))
                corrected_cache=cache.get((str(prediction),patient,'corrected_derived_center_grid'))
                cachepath=WORK/'reports/covid_spacing_cache'/f'{digest_bytes((prediction_sha+g["derived_GT_array_sha256"]+str(v["corrected_spacing"])+sha(evaluator)).encode())}.json'
                if cachepath.exists():
                    calculated=read_json(cachepath); before=calculated['before']; after=calculated['after']; origin='own immutable case cache'
                    initial_origin=calculated.get('initial_calculation_origin',calculated['origin'])
                    after_order=calculated.get('actual_corrected_call_axis_order','ZXY' if 'previous verified' in initial_origin else 'XYZ')
                    calculation_script=calculated.get('calculation_script_sha256',sha(initial_script) if initial_script.exists() else 'NA')
                elif reusable and corrected_cache and reusable['prediction_sha256']==prediction_sha:
                    before={k:float(reusable[k]) for k in METRICS}; after={k:float(corrected_cache[k]) for k in METRICS}; origin='previous verified exact legacy replay and unchanged prediction hash'
                    initial_origin=origin; after_order='ZXY'; calculation_script=sha(WORK/'scripts/reevaluate_existing_covid.py')
                else:
                    xyz=pred.transpose(1,2,0); gtxyz=v['gt'].transpose(1,2,0)
                    before=module.binary_volume_metrics(xyz,gtxyz,v['legacy_spacing'])
                    after=module.binary_volume_metrics(xyz,gtxyz,v['corrected_spacing']); origin='same evaluator, direct old/corrected-spacing recomputation'
                    initial_origin=origin; after_order='XYZ'; calculation_script=sha(__file__)
                assert before['dice']==after['dice'] and before['iou']==after['iou']
                calculated=dict(before=before,after=after,time=now(),prediction_sha256=prediction_sha,
                    derived_GT_sha256=g['derived_GT_array_sha256'],origin=origin,initial_calculation_origin=initial_origin,
                    actual_corrected_call_axis_order=after_order,calculation_script_sha256=calculation_script)
                atomic_json(cachepath,calculated)
                delta=max(abs(before[k]-original[k]) for k in METRICS)
                row=dict(identity,policy='spacing_only_derived_center_grid',status='DONE',reason='',
                    corrected_spacing_XYZ=g['spacing_only_corrected_XYZ'],corrected_evaluation_shape_XYZ=g['evaluation_shape_XYZ'],
                    changed_prediction=False,changed_GT=False,changed_resampling=False,changed_postprocess=False,
                    Z_spacing_changed=False,change_scope='only physical spacing, identical XYZ prediction/GT arrays',
                    GT_array_sha256=g['derived_GT_array_sha256'],calculation_origin=origin,legacy_replay_max_difference=delta,
                    before_call_array_order='XYZ',before_call_spacing=g['old_spacing_XYZ'],
                    after_call_array_order=after_order,
                    after_call_spacing=g['spacing_only_corrected_XYZ'] if after_order=='XYZ' else g['spacing_only_corrected_ZXY'],
                    after_call_shape=json.dumps(g['evaluation_shape_XYZ'] if after_order=='XYZ' else list(v['gt'].shape)),
                    after_GT_source=str(WORK/'reports/covid_derived_slice_sources.csv'),after_GT_array_sha256=g['derived_GT_array_sha256'],
                    calculation_script_sha256=calculation_script,
                    unit_before='legacy mm label, in-plane spacing unscaled',unit_after='mm')
                row['evaluation_array_order']='before XYZ; after '+after_order
                for key in METRICS:
                    row['before_'+key]=before[key]; row['after_'+key]=after[key]; row['delta_'+key]=after[key]-before[key]
                results.append(row)
                native=cache.get((str(prediction),patient,'corrected_native_grid'))
                if native and native['prediction_sha256']==prediction_sha:
                    extra=dict(row,policy='native_grid_separate_reevaluation',corrected_spacing_XYZ=g['canonical_original_spacing_XYZ'],
                        corrected_evaluation_shape_XYZ=g['canonical_shape_XYZ'],changed_prediction=True,changed_GT=True,changed_resampling=True,
                        change_scope='categorical prediction nearest upsample to native canonical grid; GT replaced with unresized native GT; not spacing-only',
                        calculation_origin='previous same frozen prediction native-grid evaluation',
                        after_call_array_order='ZXY',after_call_spacing=g['native_spacing_ZXY'],
                        after_call_shape=json.dumps([g['canonical_shape_XYZ'][2],g['canonical_shape_XYZ'][0],g['canonical_shape_XYZ'][1]]),
                        evaluation_array_order='before XYZ; after ZXY',
                        after_GT_source=g['original_gt']+' (canonical, no resize)',
                        after_GT_array_sha256=g['native_GT_array_sha256_ZXY'],
                        calculation_script_sha256=sha(WORK/'scripts/reevaluate_existing_covid.py'))
                    for key in METRICS:
                        extra['after_'+key]=float(native[key]); extra['delta_'+key]=float(native[key])-before[key]
                    results.append(extra)
                print('COVID',run['run'],patient,'DONE',flush=True)
            except (FileNotFoundError,KeyError,AssertionError) as error:
                results.append(dict(identity,policy='spacing_only_derived_center_grid',status='BLOCKED',reason=str(error),
                    **{prefix+k:'NA' for prefix in ('before_','after_','delta_') for k in METRICS}))
                input_failures.append(dict(run=run['run'],patient=patient,reason=str(error)))
            union_csv(WORK/'covid_metrics_before_after.csv',results)
            progress=read_json(AUDIT_STATE)
            progress.update(heartbeat=now(),last_evaluated_run=run['run'],last_evaluated_patient=patient,
                completed_case_policy_rows=len(results))
            atomic_json(AUDIT_STATE,progress)
        for policy in ('spacing_only_derived_center_grid','native_grid_separate_reevaluation'):
            part=[r for r in results if r['run']==run['run'] and r.get('policy')==policy and r['status']=='DONE']
            if len(part)!=4: continue
            macro={k:run[k] for k in ('run','method','ratio','seed','cohort')}
            macro.update(policy=policy,patients=4,status='DONE')
            for prefix in ('before_','after_','delta_'):
                for k in METRICS: macro[prefix+k]=sum(r[prefix+k] for r in part)/4
            macros.append(macro)
    for ratio in ('5%','10%'):
        results.append(dict(method='NA: unresolved external manuscript rows',ratio=ratio,run='NA',patient='NA',policy='NA',
            status='BLOCKED',reason='No local mapping of manuscript external method identities/predictions/GT/space/evaluation; cannot infer values or apply a ratio',
            **{prefix+k:'NA' for prefix in ('before_','after_','delta_') for k in METRICS}))
    union_csv(WORK/'covid_metrics_before_after.csv',results)
    union_csv(WORK/'covid_metrics_before_after_macro.csv',macros)
    union_csv(WORK/'reports/covid_available_prediction_inventory.csv',[{k:v for k,v in r.items() if k not in ('original_metrics','predictions')} for r in registry])
    maxdiff=max(r.get('legacy_replay_max_difference',0) for r in results if r['status']=='DONE')
    atomic_json(WORK/'reports/covid_spacing_supplement_receipt.json',dict(status='DONE' if not input_failures else 'PARTIAL',
        time=now(),existing_runs=len(registry),completed_case_policy_rows=sum(r['status']=='DONE' for r in results),
        external_rows='BLOCKED identity/inputs unavailable',input_failures=input_failures,
        maximum_legacy_replay_difference=maxdiff,training_started=False,GPU_used=False,
        evaluator=str(evaluator),evaluator_sha256=sha(evaluator),manifest_sha256=sha(manifest)))
    table='|病例|canonical XYZ尺寸|旧评价XYZ尺寸|原spacing XYZ|旧spacing XYZ|spacing-only修正XYZ|\n|---|---|---|---|---|---|\n'
    for g in geometry: table+=f"|{g['patient']}|{g['canonical_shape_XYZ']}|{g['evaluation_shape_XYZ']}|{g['canonical_original_spacing_XYZ']}|{g['old_spacing_XYZ']}|{g['spacing_only_corrected_XYZ']}|\n"
    mtable='|method/run|预算|口径|Dice前→后|IoU前→后|HD95前→后|ASD前→后|\n|---|---|---|---|---|---|---|\n'
    for r in macros: mtable+='|'+r['method']+'/'+r['run']+'|'+r['ratio']+'|'+r['policy']+'|'+'|'.join(f"{r['before_'+k]:.9f}→{r['after_'+k]:.9f}" for k in METRICS)+'|\n'
    put(WORK/'COVID_SPACING_AUDIT.md',f"""# COVID-19 spacing与重评价审计

更新时间{now()}。独立CPU评价，没有新COVID训练、模型推理或GPU占用。共发现{len(registry)}个已有本地run，保留旧run与canonical1337 run各自身份，不按分数挑run、不把旧seed混入canonical主对照。外部稿件逐行身份/预测/空间信息仍缺失，NA/BLOCKED写入CSV，未猜测方法或按比例修正指标。

{table}

并非所有原平面都是512²：两例coronacases为512²，两例radiopaedia为630²，均变为256²；Z切片数分别200、256、66、93，原序号0..N−1逐一保留。原始NIfTI header/orientation/canonical尺寸与GT/image affine一致性在reports/covid_spacing_geometry.csv。615个实际derived NPZ的路径与SHA/GT键在covid_derived_slice_sources.csv；逐病例GT严格等于原canonical GT各slice nearest到256²，不做Z插值。

数组顺序：原canonical NIfTI XYZ；dataset/prediction文件存ZXY；旧evaluator reconstruct_patient_volume沿axis=2把2D切片堆回XYZ，spacing(sx,sy,sz)与XYZ对应，轴序正确。旧问题是resize后仍存原sx/sy，并非应缩放Z spacing。新spacing-only评价使用同一binary_volume_metrics、相同prediction/GT XYZ数组，只把sx改为sx×(Nx−1)/255、sy改为sy×(Ny−1)/255，sz严格保持原值。

比例根据实际SciPy zoom(grid_mode=False)的首末像素中心跨度，而不是统一×2或所有轴同时缩放。代码来源{location(PROJECT/'train_COVID19/covid19_official20_v2.py','def resize_slice')}、{location(PROJECT/'train_COVID19/covid19_official20_v2.py','spacing=np.array')}；重建{location(evaluator,'    pred_vol = np.stack')}；评价{location(evaluator,'def binary_volume_metrics')}。补充脚本{Path(__file__)}直接调用同一个冻结evaluator，见SHA清单。无需比例推测HD95/ASD，两项从surface EDT实算。

covid_metrics_before_after.csv逐方法/预算/run/患者/口径列明prediction与GT源、hash、shape、spacing、axis对应、代码位置、before/after/delta四指标和变化范围。Dice/IoU为比例；修正后距离mm，旧距离只保留原mm标签及geometry-error注释。原预测/GT/receipt不覆盖，legacy最大逐指标重放差值{maxdiff}。既有baseline/Full核验过的同预测hash结果复用，其余已有预测直接重评价，并逐病例缓存以便中断恢复。

spacing_only_derived_center_grid：prediction、GT、重采样、后处理全部不变，Dice/IoU严格逐病例相等，只有平面内spacing改正。新计算以XYZ调用；已核验缓存的derived修正原来以ZXY和相应(sz,sx,sy)调用，联合轴排列数学等价且预检已独立验证，CSV的before_call/after_call字段保留真实数组次序、spacing与shape，normalized XYZ字段用于几何对照，不混淆两者。native_grid_separate_reevaluation：仅对先前已完成该口径的baseline/Full保留另一个独立结果；实际调用为ZXY、spacing(sz,sx,sy)，预测nearest回原canonical网格，GT换为未缩放原GT，Dice/IoU可能变化，这是网格/GT重采样重评价，不能统称spacing修正，也不将这部分与spacing-only平均混用。其他本地方法本轮只需、并已做spacing-only；未新增native-grid评价要求。

后处理：读取既有categorical预测并binary>0，未加新LCC、TTA、ensemble或阈值搜索。empty-both时Dice/IoU1、HD95/ASD0；empty-one时Dice/IoU0、表面距离inf，保持既有程序而非换成有限惩罚。HD95为合并两向表面距离95百分位；ASD为合并两向样本均值，并非两方向均值再等权。指标先在完整3D患者上计算，再4患者等权macro，不把615切片作独立样本。未发生输入缺失的本地病例条数：{sum(r['status']=='DONE' for r in results)}；实际缺失原因见CSV和receipt。

{mtable}

论文应更新：区分原历史距离记录与正确spacing-only物理距离，原记录保持溯源；把512/630各自到256的平面变化及Z不变写清。原网格结果单独注明GT/预测重采样改变。尚不能证明外部文献方法使用同一口径，因此不能给外部整表统一加mm或比较混合单位。两种修正口径均保留，不据指标优劣选择一种。
""")
    return len(registry)


def current_status():
    state=read_json(WORK/'state/queue.json'); rows=[]
    for task,t in state['tasks'].items():
        p=WORK/f'runs/{task}/progress.json'
        progress=read_json(p) if p.exists() else {}
        rows.append(dict(task=task,status=t['status'],pid=t.get('pid','NA'),step=progress.get('step',t.get('latest_step',0)),
            budget_updates=30000,log=t.get('log',str(WORK/f'logs/{task}.log')),reason=t.get('reason',''),time=now()))
    union_csv(WORK/'reports/CURRENT_QUEUE_STATUS.csv',rows)
    atomic_json(WORK/'reports/SUPPLEMENT_QUEUE_SNAPSHOT.json',state)
    return rows


def handoff():
    counts=read_json(WORK/'reports/fixed_rectangle_audit.json')
    text=f"""{MARKER}

固定尺度：FIXED_RECTANGLE_AUDIT.md及fixed_rectangle_historical_records.csv明确旧Full150条step200..30000稀疏记录；s范围{counts['historical']['s_min']:.12f}–{counts['historical']['s_max']:.12f}、均值{counts['historical']['s_mean']:.12f}，记录点97²比例100%，实际面积均值14.3569946%、固定误差0。历史batch切片ID与每时点teacher快照缺失，不能声称全30000步平均；E1逐步曲线独立标新run，不能冒充旧Full。E2/E3配置未变。

COVID：COVID_SPACING_AUDIT.md、covid_metrics_before_after.csv和macro CSV覆盖所有已定位本地预测/run。spacing-only保留预测/GT/重采样/后处理，仅改sx/sy；Z不缩放，Dice/IoU应完全不变。原网格另存结果改变了GT/预测重采样，须单独说明。原平面包括512²和630²，不全是512²。CSV含完整源/hash/axis/shape/spacing与真实重算的HD95/ASD；外部方法身份/输入缺失仍NA，不能根据比例估计。

协议：COMPARABILITY_AUDIT.md和protocol_comparison.csv逐项说明保持一致/预期改变/其他差异/无法核验。补清ICT实际最大系数5×.1=.5，不把参数consistency=.1当作最大有效权重。原预训练生成预算/seed、原训练当时清单与全部import依赖hash无法核验，不能写完全一致或位级精确续训。E4残差参数+.936%不是全网络增幅。E5实际执行train_entry→official_adapter，官方方法核心来自锁定commit的exp_acdc/train_adamix_ct.py；是有明示预算/验证/划分适配的官方方法实现，并未直接运行未经改动原main。

新增文件及原始统计实际进入证据包，保持五项单次结果包括下降。当前任务状态见reports/CURRENT_QUEUE_STATUS.csv与SUPPLEMENT_QUEUE_SNAPSHOT.json；未完成测试指标仍NA。原queue继续自动训练/评价/A1–A6/汇总。审计跟随进程位于同tmux的audits窗口，在queue最终归档后追加本节并再次生成一致快照包，避免最终summarizer覆盖本次写作交接。
"""
    put(WORK/'EDITOR_SUPPLEMENT_CN.md',text)
    for name in ('EDITOR_HANDOFF_CN.md','FINAL_REPORT_CN.md'):
        path=WORK/name
        if path.exists(): put(path,path.read_text().split(MARKER)[0]+text)


def deliver():
    log=WORK/'logs/supplement_delivery.log'
    with log.open('a') as f:
        subprocess.run([str(PYTHON),str(WORK/'scripts/summarize_results.py')],cwd=WORK,stdout=f,stderr=subprocess.STDOUT,check=True)
        handoff()
        subprocess.run([str(PYTHON),str(WORK/'scripts/package_results.py')],cwd=WORK,stdout=f,stderr=subprocess.STDOUT,check=True)


def process_alive(pid):
    try: os.kill(int(pid),0); return True
    except (OSError,TypeError,ValueError): return False


def main():
    p=argparse.ArgumentParser(); p.add_argument('--watch',action='store_true'); p.add_argument('--live-only',action='store_true'); args=p.parse_args()
    with (WORK/'state/supplementary_audits.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Supplement audit already active')
        status=dict(status='RUNNING',pid=os.getpid(),started=now(),script_sha256=sha(__file__),GPU_used=False,training_configuration_changed=False)
        def persist(phase):
            status.update(phase=phase,heartbeat=now()); atomic_json(AUDIT_STATE,status)
        try:
            persist('fixed_rectangle'); scale_audit()
            persist('comparability'); comparability_audit()
            if not args.live_only:
                persist('covid_existing_predictions'); covid_audit()
            current_status(); persist('initial_evidence_delivery'); deliver()
            status['initial_reports_complete']=True; status['initial_reports_completed']=now()
            if not args.watch:
                status['status']='DONE'; persist('complete'); return
            persist('FOLLOWING_EXISTING_QUEUE')
            last_refresh=time.monotonic()
            while True:
                state=read_json(WORK/'state/queue.json')
                terminal=state['status'] in ('FINISHED','FINISHED_WITH_FAILURES','FAILED')
                controller_command=Path(f'/proc/{state.get("controller_pid")}/cmdline')
                real_controller=controller_command.exists() and str(WORK/'scripts/run_queue.py').encode() in controller_command.read_bytes()
                if terminal and not real_controller:
                    persist('final_refresh'); scale_audit(); comparability_audit(); current_status(); deliver()
                    status.update(status='DONE',completed=now()); persist('final_evidence_delivered'); return
                if time.monotonic()-last_refresh>=600:
                    scale_audit(); comparability_audit(); current_status(); handoff()
                    last_refresh=time.monotonic()
                persist('FOLLOWING_EXISTING_QUEUE'); time.sleep(30)
        except BaseException as error:
            import traceback
            status.update(status='FAILED',reason=str(error),traceback=traceback.format_exc(),failed=now())
            atomic_json(AUDIT_STATE,status); raise


if __name__=='__main__': main()
