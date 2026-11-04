"""Write the four-gap closure report and extend its bounded source index."""
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]

def readcsv(p):
    p=Path(p)
    with p.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t' if p.suffix=='.tsv' else ','))

def j(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def csvout(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

index=readcsv(OUT/'SOURCE_INDEX.csv');by_path={r['original_path']:r for r in index}
def source(p,copy=True,lines=None):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    key=str(p)
    if key not in by_path:
        r={'source_id':f'S{len(index)+1:04d}','original_path':key,'exists':p.is_file(),
           'sha256':sha(p) if p.is_file() else '', 'bytes':p.stat().st_size if p.is_file() else '',
           'packed_path':'','packed_kind':'INDEX_ONLY','expected_sha256':'','expected_sha_matches':'','original_lines':''}
        if p.suffix in ['.json','.py','.csv','.tsv','.yaml','.list','.md','.txt'] and p.is_file():r['original_lines']=len(p.read_text(errors='replace').splitlines())
        index.append(r);by_path[key]=r
    r=by_path[key]
    if lines is not None:
        a=p.read_text().splitlines();dest=OUT/'evidence/line_excerpts'/(r['source_id']+'_'+p.name+'.txt')
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text('ORIGINAL: '+key+'\nSHA256: '+r['sha256']+'\n'+'\n'.join(f'{n}: {a[n-1]}' for n in sorted(set(lines)) if 1<=n<=len(a))+'\n')
        r['packed_path']=str(dest.relative_to(OUT));r['packed_kind']='LINE_NUMBERED_EXCERPT'
    elif copy and not r['packed_path'] and p.is_file():
        assert p.suffix in ['.json','.py','.csv','.tsv','.yaml','.list','.md','.txt','.sha256'] and p.stat().st_size<1200000
        dest=OUT/'evidence'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        r['packed_path']=str(dest.relative_to(OUT));r['packed_kind']='BYTE_IDENTICAL_COPY'
    return r

def cite(p,needles=None,lines=None):
    p=Path(p)
    if not p.is_absolute():p=ROOT/p
    r=source(p,copy=False)
    a=p.read_text(errors='replace').splitlines()
    r['original_lines']=len(a)
    if lines is None:
        lines=[i for i,s in enumerate(a,1) if any(n in s for n in (needles or []))][:7]
    lines=[n for n in lines if 1<=n<=len(a)]
    # Every cited textual source must be reviewable from this package.
    if r['packed_kind']=='LINE_NUMBERED_EXCERPT' or not r['packed_path']:
        if not r['packed_path'] and p.stat().st_size<1200000:
            source(p,copy=True)
        else:
            prior=[]
            if r['packed_path']:
                prior=[int(m.group(1)) for m in re.finditer(r'(?m)^(\d+): ',(OUT/r['packed_path']).read_text())]
            source(p,copy=False,lines=sorted(set(prior+lines)))
    ln=','.join(map(str,lines)) if lines else '文件/二进制元数据'
    return f'`{p}`，行 {ln}；SHA256 `{r["sha256"]}`（{r["source_id"]}）'

fullroot='runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT'
five='experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3'
r2='tcsmix_control_revision_20261006/formal_R2_fixed92_20261007'
external='external_comparators/2026_reproduction'
records=readcsv(OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv')
meta=j(OUT/'CHECKPOINT_CPU_METADATA.json');budget=j(OUT/'ACDC_TRAINING_BUDGET_RECOMPUTATION.json')
r2check=j(OUT/'R2_RECORD_VERIFICATION.json');empty=j(OUT/'EMPTY_CASE_CHECK_RECEIPT.json')
promise=j(OUT/'PROMISE_BASELINE_CPU_RECALC_RECEIPT.json');summary=j(OUT/'COLLECTION_SUMMARY.json')
split=j(OUT/'ACDC_SSL_SPLIT_RECONSTRUCTION.json')

# All small input evidence needed by the paper session is carried in the package.
for p in [
    'formal_E4_E5_20261008/tasks/E5/reports/INITIALIZATION_ACTUAL.json',
    'formal_E4_E5_20261008/tasks/E5/protocol.json',
    'formal_E4_E5_20261008/scripts/official_adapter.py',
    'formal_E4_E5_20261008/external/AdaMix/exp_acdc/data/dataset.py',
    external+'/audits/HIEN_PROTOCOL_AUDIT.md',external+'/audits/PMEUL_PROTOCOL_AUDIT.md',
    external+'/manifests/UPSTREAM_REPOSITORIES.tsv',
    'revision_runs_20261006/reports/covid_geometry_audit.csv',
    'revision_runs_20261006/reports/covid_metric_unit_audit.csv',
    'revision_runs_20261006/covid_derived_slice_sources.csv',
]:
    if (ROOT/p).exists():source(p)
for m in meta:
    p=Path(m['path']);r=source(p,copy=False);r['expected_sha256']=m['sha256'];r['expected_sha_matches']=r['sha256']==m['sha256']
    r['packed_kind']='CPU_METADATA_ONLY_WEIGHT_EXCLUDED'
for p in ['audits/three_innovation_ablation_transfer_v22_0/V22_0_FORMAL_START_MANIFEST.json',
          'audits/acdc10_third_innovation_codex_prep_top6_v21_3/ROUND1_TOP6_CONTROLLER_DRY_CONSUMPTION.json']:
    a=(ROOT/p).read_text().splitlines()
    nums=set(range(1,7))
    if 'DRY_CONSUMPTION' in p:
        for i,s in enumerate(a,1):
            if 'V21_3_X03_CONTOUR_RELIABILITY_WEIGHT' in s:nums.update(range(i,min(len(a)+1,i+35)))
    else:
        for i,s in enumerate(a,1):
            if any(n in s for n in ['MaskAdaMixSourceSHA256','train_promise12/train_promise12_bcp_fft.py','X03SourceSHA256','created_at']):nums.update(range(max(1,i-1),i+2))
    source(p,copy=False,lines=nums)
for p in ['train_ACDC/beta_fft_label3.py','train_ACDC/beta_fft_label7.py']:
    source(p)

# SHA-only indexes of the masks examined during lightweight CPU verification.
mask_index=[]
for table in ['EMPTY_CASE_COUNTS_FROM_SAVED_MASKS.csv','PROMISE_BASELINE_SAVED_PRED_CPU_RECALC.csv']:
    for r in readcsv(OUT/table):
        for kind in ['prediction','GT']:
            if r.get(kind+'_sha256'):
                mask_index.append({'kind':kind,'path':r[kind+'_source'],'sha256':r[kind+'_sha256'],
                                   'audit_table':table,'copied_into_package':False})
mask_index=list({(r['kind'],r['path']):r for r in mask_index}.values())
csvout('SAVED_MASK_INPUT_SHA256.csv',mask_index)

# Recompute geometry, without reading image voxels or calculating distance ratios.
geom=readcsv(ROOT/'revision_runs_20261006/reports/covid_spacing_geometry.csv');geocheck=[]
for r in geom:
    shape=jstr=json.loads(r['canonical_shape_XYZ']);old=json.loads(r['old_spacing_XYZ']);new=json.loads(r['spacing_only_corrected_XYZ'])
    computed=[old[0]*(shape[0]-1)/255,old[1]*(shape[1]-1)/255,old[2]]
    geocheck.append({'patient':r['patient'],'native_canonical_shape_XYZ':r['canonical_shape_XYZ'],
        'evaluation_shape_XYZ':r['evaluation_shape_XYZ'],'prediction_shape_ZXY':r['prediction_stored_shape_ZXY'],
        'old_spacing_XYZ':r['old_spacing_XYZ'],'corrected_spacing_XYZ':r['spacing_only_corrected_XYZ'],
        'recomputed_spacing_XYZ':json.dumps(computed),'spacing_max_error':max(abs(x-y) for x,y in zip(computed,new)),
        'Z_spacing_unchanged':old[2]==new[2],'Z_slices_unchanged':r['Z_slices_unchanged'],
        'HD95_recomputed_this_round':False,'existing_corrected_metric_source':'revision_runs_20261006/covid_metrics_before_after_macro.csv'})
csvout('COVID_GEOMETRY_RECOMPUTATION.csv',geocheck)
covmac=readcsv(ROOT/'revision_runs_20261006/covid_metrics_before_after_macro.csv')
covrows=[]
for row in records:
    if row['dataset']!='COVID-19':continue
    run=row['historical_run']
    if run.startswith('COVID19_') and 'BASELINE_CANON1337' in run:run+='_CONFIRMATORY_TEST_ONCE'
    matches=[r for r in covmac if r['run']==run and r['policy']=='spacing_only_derived_center_grid']
    if matches:
        r=matches[0];covrows.append({**r,'original_result_status':r['status'],
            'row_id':row['row_id'],'paper_method':row['paper_method'],'label_budget':row['label_budget'],
            'historical_run':row['historical_run'],'version':'spacing_only_derived_center_grid','status':'VERIFIED_EXISTING_RESULT',
            'source_sha256':sha(ROOT/'revision_runs_20261006/covid_metrics_before_after_macro.csv')})
    else:
        covrows.append({'row_id':row['row_id'],'paper_method':row['paper_method'],'label_budget':row['label_budget'],
            'historical_run':row['historical_run'],'version':'spacing_only_derived_center_grid','status':'NOT_FOUND',
            'reason':'No existing spacing-corrected result located for this exact reproduced run; old common adapter uses same unscaled derived spacing. No new volume distance evaluation started.'})
csvout('COVID_RESULT_VERSION_COVERAGE.csv',covrows)

# Protocol facts beside each row. Derived plans are not marked as observed execution.
facts=[]
for row in records:
    p=row['config_path']
    if 'external_comparators/2026_reproduction/configs' in p:
        c=j(p)
        facts.append({'row_id':row['row_id'],'paper_method':row['paper_method'],'dataset':row['dataset'],
            'seed':c['seed'],'max_iterations':c['max_iterations'],'optimizer':json.dumps(c.get('optimizer',{})),
            'initialization':json.dumps(c.get('pretraining',{})),
            'batch_size':c['batch_size'],'labeled_batch_size':c['labeled_batch_size'],'train_patch':str(c['patch_size']),
            'inference_size':str(c['inference_size']), 'model':c['network'],
            'losses':json.dumps({k:v for k,v in c.items() if 'loss' in k or k in ['computed_but_not_optimized','lambda','temperature']},ensure_ascii=False),
            'selection':c['selection_rule'], 'protocol_source':p,
            'scope':'Receipt config SHA verified; helper SHA matches prelaunch source freeze; method-specific settings retained.'})
    elif row['historical_run'] in ['ACDC5_FINAL3','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT']:
        m=next(m for m in meta if m.get('args',{}).get('run_root','').endswith(row['historical_run']))
        a=m['args'];facts.append({'row_id':row['row_id'],'paper_method':row['paper_method'],'dataset':row['dataset'],
            'seed':a['seed'],'max_iterations':m['global_step'],'optimizer':'SGD(momentum=0.9,weight_decay=1e-4)',
            'initialization':a['pretrained_model']+'/unet_best_model.pth; model1/model2/EMA load net; base SGD momentum transplanted',
            'batch_size':a['batch_size'],'labeled_batch_size':a['labeled_bs'],'train_patch':str(a['image_size']),
            'inference_size':'256x256 then native reconstruction','model':'beta-FFT UNet + TRC(4,16)',
            'losses':'historical beta-FFT mixed/CPS/FFT + supervised/ICT + CTW coefficient0.05; LR base/router0.01, no entry scheduler',
            'selection':'student1 val macro Dice every200, strict greater; step'+str(row['checkpoint_selected_step']),
            'protocol_source':m['path'],'scope':'Actual last checkpoint namespace + entry + step trace; pretrain generation history remains missing.'})
csvout('PROTOCOL_FACTS_BY_ROW.csv',facts)

verdicts=[
    {'item':'A','status':'PARTIAL','can_close':'NO_ALL_ROWS; YES_LOCATED_METRICS_AND_GEOMETRY','verified':f'24 available comparator rows +5 supplements; {summary["aggregate_tests"]} macro checks; PROMISE baseline precision recovered; 17 actual single-empty records accounted for; COVID geometry/units clarified',
     'missing':'Current manuscript Table2-4 full row roster; some early evaluator/import SHA provenance; corrected distances for HiEn/PMEUL/UCAD existing runs'},
    {'item':'B','status':'PARTIAL','can_close':'YES_SSL_5_10; NO_PRETRAIN_GENERATION_OR_TOTAL_5','verified':'Two Full runs joint30000, each student30000, full states/EMA/SGD groups and LR; label3/label7 SHA and optimizer state; current SSL labels',
     'missing':'Actual pretraining generation commands/logs/seed/selection step/frozen label subset/update budget; 5% combined pretrain+SSL total'},
    {'item':'C','status':'PARTIAL','can_close':'YES_CTW_AND_COVID_ENTRY; NO_ALL_IMPORTED_DEPENDENCIES','verified':'Main Full archived entry + prelaunch CTW SHA + live loss endpoints; COVID5/10 actual e1006e entry; earlier COVID dependency hashes; current/snapshot equality',
     'missing':'Main Full-specific historical augment/DiceLoss runtime hash; FINAL3-specific historical COVID adapter/FFT import receipt/archive'},
    {'item':'D','status':'VERIFIED','can_close':'YES','verified':'R2 original120 unique case-class rows /40 ED-ES frames /20 patient rows; all macros and per-patient means reproduced; pair control89.98 preserved','missing':''},
]
csvout('CLOSURE_DECISIONS.csv',verdicts)

def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rows)

keymetrics=[]
for row in records:
    if row['paper_method'] in ['完整方法 / 论文 R1','更换混合控制信号','固定混合区域（92×92）','均匀权重替换 CTW','普通残差替换 TRC','AdaMix-CT 共同协议比较']:
        keymetrics.append([row['paper_method'],row['historical_run'],row['dice_display'],row['iou_display'],row['hd95_display'],row['asd_display'],
            '完整方法（外部方法比较）' if row['historical_run'].startswith('E5') else ('论文主结果/原八组合' if row['paper_method'].startswith('完整方法') else ('更换混合控制信号，89.98%' if not row['historical_run'].startswith('R1') else '该配置自身：模块替代配对控制'))])

manifestcites='\n\n'.join(cite(p,needles=['CheckpointPath','CheckpointSHA','TestManifest','EvaluatorPath','Dice']) for p in [
    'final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json',
    'final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json',
    'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/TEST_COMPLETION_RECEIPT.json'])
geometry_table=table(['患者','canonical XYZ','评价 XYZ','原 spacing XYZ','修正 spacing XYZ'],[
    [r['patient'],r['canonical_shape_XYZ'],r['evaluation_shape_XYZ'],r['old_spacing_XYZ'],r['spacing_only_corrected_XYZ']] for r in geom])
correction_table=table(['方法/预算','旧 HD95 / ASD（原 mm 标签）','spacing-only HD95 / ASD（mm）'],[
    [r['paper_method']+' '+r['label_budget'],r['before_hd95_mm']+' / '+r['before_asd_mm'],r['after_hd95_mm']+' / '+r['after_asd_mm']] for r in covrows if r['status']=='VERIFIED_EXISTING_RESULT'])

report=f'''# REVIEW_CLOSURE_EVIDENCE

核查完成时间：{datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}；主机 n1。
项目根目录：`{ROOT}`。本轮仅核对 A–D 四项缺口，CPU 读取既有检查点、解析日志/CSV、复算现有统计与十例 PROMISE 保存预测。未构建分割模型、未做 GPU 前向，未启动/恢复训练、未重选检查点、未修改原源码、数据、检查点或历史结果。E3 保持取消；不新增 seed。
适用 AGENTS.md：已检查项目祖先路径及所给证据子树，未找到；记录于 `AUDIT_SCOPE.json`。

论文题目：**Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation**。统一模块名 TRC — Token-Guided Residual Conditioning、TCSMix — Teacher-Confidence-Scaled Mixing、CTW — Confidence-Transition Weighting。保留 ACDC、PROMISE12、COVID-19。

## 四项关闭结论

{table(['缺口','状态','能否关闭','仍缺的具体证据'],[[v['item'],v['status'],v['can_close'],v['missing'] or '无'] for v in verdicts])}

`VERIFIED` 是列明范围内有现存证据及复核；`PARTIAL` 不代表实验失败；`NOT_FOUND` 仅表示本轮在授权项目记录链中未定位到该项。新的 CPU 复算与历史原始评价分开标注，未把复算当历史执行记录。完整 SHA、原路径、打包路径见 `SOURCE_INDEX.csv`；二进制权重只列 SHA 和 CPU 元数据，不进入压缩包。

## A — 表 2–4 的行映射、真实协议与评价口径：PARTIAL

### A1. 已定位行：VERIFIED；最终逐行覆盖：PARTIAL

`TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv` 一行一个实际结果，共 **24 条主表候选记录**：β-FFT/完整方法的 10 个任务预算行、HiEn/PMEUL 的 10 个行、UCAD 的 4 个行；另列 **5 条补实验**。每行分别登记 run、checkpoint、实际/记录 SHA、split manifest、验证选点、评价来源、真实评价 driver 与 metric implementation、单位、聚合和协议缺口。没有新增 local/published 分类；均沿用作者确认的复现身份。

服务器 v23 文件名是 TABLE1 ACDC、TABLE2 PROMISE、TABLE3 COVID、TABLE4 八组合，并不是当前论文表 2–4 的逐行排版索引。当前论文最终逐行名单未在本轮引用链中定位到，不能用这 24 行宣布“表 2–4 所有行闭合”，也没有凭空添加其他方法。完整八组合原 TSV 同时入包，保留每个真实运行。

主 Full / 论文 R1 绑定原 X03 检查点29600，**不是**服务器 R1_v2_B。原成绩和补实验配对关系如下（距离均 grid）：

{table(['论文名称','服务器历史编号','Dice%','IoU%','HD95','ASD','比较参照/范围'],keymetrics)}

E4、C1、R2 对照均使用 89.98% 更换混合控制信号配置；不能把这条参照换成主 Full 90.35% 后计算模块单因素收益。R2 结果只支持该控制信号配置的动态控制与 fixed92 比较。E5 是整个方法比较，不能将整个差值归因于 TCSMix。

关键原始 receipt 的实际路径、SHA 与引用行：

{manifestcites}

### A2. 已有病例记录复算与空例：VERIFIED

`AGGREGATE_RECOMPUTATION.csv` 对 **{summary['aggregate_tests']} 项指标均值**作复算，全部与相应历史汇总在 1e-12 内一致；没有重新推理或选择 checkpoint。`NONFINITE_RECORDS.csv` 为已检查原始数值的非有限值清单，本轮为 **0 条**。这只覆盖实际列明的病例表，不能泛指论文尚未映射的所有方法。

COVID两条canonical baseline额外登记了八条已存在的2026-10-06病例legacy重放记录，其`before_*`均值与原receipt逐项一致，检查点SHA一致、四例Dice均大于0、数值有限。来源是现有保存预测的历史重评价CSV，**不是新找回的原训练期病例表**；筛选run/spacing-only及`before_*`列条件写入映射表和 `COVID_BASELINE_CASE_REPLAY_VERIFICATION.csv`。

PMEUL ACDC5 的 120 条类别记录中有 **18 条 Dice=0**。读取既有预测 NPZ 与对应 GT label 数组后，确认 **17 条预测为空且 GT 非空**，分布于 9 个帧病例；另1条 patient024_frame02 / LV 双方非空、零重叠。17条的 HD95 与 ASD 均严格等于原数组 `norm(shape)` 惩罚，全部计入普通均值，没有删除或用有限值替换。原始来源、mask SHA、像素数、形状和逐行对照见 `EMPTY_CASE_COUNTS_FROM_SAVED_MASKS.csv` 与 `SAVED_MASK_INPUT_SHA256.csv`；包内不含这些数组。

ACDC 主 Full/β-FFT、模块补实验、HiEn/PMEUL：前景 RV=1、MYO=2、LV=3；每个 ED/ES 体分别算三类，40帧体×3类等权平均；20患者表另作两帧均值。2D nearest/order0 到256²，预测回到 H5 原平面，Z不重采样；不做推理后 LCC。MedPy HD95 为合并双方向表面距离的95百分位；**ASD 是 prediction→GT 单方向表面距离均值**。两者不传 voxel spacing，因此为重建数组 **grid** 距离。双空四指标0，单空 Dice/IoU0、距离 `norm(shape)`。教师训练时 LCC 与推理后处理不是同一流程。

{cite('tools/v22_0_acdc_test_frozen_checkpoint.py',lines=[35,44,45,47,49,54,62,68,73,84])}

### A3. PROMISE12 基线精度和实际规则补回：VERIFIED 数值；PARTIAL 历史依赖绑定

找到 `audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json`，并对其对应 Case40–49 的十份既有预测做 **{promise['seconds']:.3f} 秒 CPU 复算**：

Dice **0.8110316082469234**；IoU **0.6843582268075924**；HD95 **3.669142478708649 grid**；ASD **1.5342924140259098 grid**。

四项与历史精度记录逐项差 **0.0**。原舍入 TSV 0.8110/0.6844/3.67/1.53 保留；本轮更新新索引中的精度来源，不覆盖旧表。新复算 CSV 和 receipt 明确是当前 CPU 复算，历史精度由现有 audit JSON 支持，未冒充训练期完整 test 行。

{cite('audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json',needles=['created_at','checkpoint_iter','checkpoint_sha256','0.811031','0.684358','3.669142','1.534292'])}

该基线实际定位到的评价实现为 `audits/run_v19_5_promise20_and_mscmr_final_r3.py:119–176`：对现存 staged `(Z,224,224)` 预测/GT计算前景，10病例等权宏平均，HD95 同为 MedPy 双方向合并95百分位，ASD单方向，grid，无额外重采样/后处理。**基线双空 Dice/IoU=1，距离0；单空 Dice/IoU=0、距离 inf**。Full 的 v22.1 程序则双空 Dice/IoU=0、单空网格对角线惩罚；HiEn/PMEUL 共用 adapter 为后一规则。不能将所有 PROMISE 行笼统写成完全相同空例协议。十例基线的保存预测与 GT 均非空，故该空例差异未改变这十例结果。历史基线执行时导入依赖 SHA 仍未定位到，原入口/依赖版本绑定不能由当前复算替代。

{cite('audits/run_v19_5_promise20_and_mscmr_final_r3.py',lines=[119,124,126,128,136,151,159,164,179])}

{cite('tools/v22_1_postfull_test_launcher.py',lines=[172,183,186,189,193,200,208,219])}

### A4. COVID spacing、轴顺序、网格、单位：VERIFIED 既有几何/修正范围

原 canonical NIfTI 为 XYZ；slice 存 XY，dataset/保存预测体为 **ZXY**；原评价器按 `axis=2` 将 slice 堆回 **XYZ**，spacing `(sx,sy,sz)` 与 XYZ 对应，轴顺序本身正确。平面512²或630²到256²，但旧 NPZ仍写原 sx/sy，**未给缩小后的 XY 网格修正 spacing**；Z切片逐一保留，不能同时缩放三轴。原距离的“mm”标签保留为历史标签，不能无注释当正确物理距离。

{geometry_table}

固定原预测、GT和后处理，按 SciPy zoom 默认 `grid_mode=False` 的首末像素中心跨度，sx乘 `(Nx−1)/255`、sy乘 `(Ny−1)/255`，**sz保持原值**。`COVID_GEOMETRY_RECOMPUTATION.csv` 本轮只对四例 spacing 公式复算，误差0；距离值取现有 EDT 实算结果，未通过比例推测。

上表使用XYZ表达同一个物理网格。既有修正CSV同时登记实际函数调用：原评价用XYZ/`(sx,sy,sz)`，部分修正执行直接用保存的ZXY数组/`(sz,sx',sy')`；这只是数组与spacing同时对应的轴重排，未改变预测、GT或重采样。不能将XYZ spacing直接传给ZXY数组；逐例`before_call_array_order`、`after_call_array_order`和`*_call_spacing`原样保留。

{cite('train_COVID19/covid19_official20_v2.py',lines=[80,82,149,154,156,159,162,163,172])}

{cite('train_COVID19/covid19_volume_evaluator.py',lines=[31,40,42,45,53,60,61,65,83,97,102])}

原旧网格及修正结果分别登记，以下为与本表 baseline/Full 精确 run 匹配的 spacing-only 原有结果：

{correction_table}

四行 Dice/IoU 在 spacing-only 口径下不变。`native_grid_separate_reevaluation` 另外改变预测重采样及 GT 网格，Dice/IoU会变化，**不能称纯 spacing 修正，也不混入 spacing-only 比较**。逐患者预测/GT来源、数组/hash和旧/新 spacing 见入包的 `covid_metrics_before_after.csv`。现有重评价 receipt 的 legacy最大逐指标差0.0，时间2026-10-06T21:14:54.761639+08:00，geometry修正记录2026-10-06T21:19:54.932521+08:00。

{cite('revision_runs_20261006/reports/covid_reevaluation_receipt.json',needles=['legacy_maximum','geometry','time','evaluator_sha256'])}

HiEn/PMEUL 的 receipt 中 `evaluator` 是 canonical参考路径；真实执行链是 `adapters/evaluate.py → common/metrics.py → common/data.py`。这三个文件的当前 SHA 全部与预启动 `FINAL_SOURCE_FREEZE.sha256` 匹配。COVID 路径从 NPZ读取旧 spacing、`moveaxis(ZXY,0,-1)` 回XYZ，直接算双方向 EDT，与上述旧几何问题相同。**HiEn/PMEUL/UCAD 精确运行的已完成 spacing 修正结果未定位到**，`COVID_RESULT_VERSION_COVERAGE.csv` 为其记 NOT_FOUND；本轮没有启动体距离重评价，也没用比例补值。修正距离主表尚不能整表关闭；作者需要在写作中明确原评价版本或待齐的统一修正版本，不混列。

{cite(external+'/adapters/common/data.py',lines=[202,206,210,218])}

{cite(external+'/adapters/common/metrics.py',lines=[60,70,79,117,130,132,138])}

COVID 为4患者体等权平均、单前景；HD95为双方向表面距离合并后95百分位，**ASD为双方向表面样本合并后的均值**，与ACDC/PROMISE单方向ASD不同。双空 Dice/IoU1、HD95/ASD0。单空代码先返回 `[inf]`；当前 NumPy **{empty['numpy_version']}** 的隔离 CPU 检查得到 HD95 **NaN**、ASD **inf**（percentile 的非有限插值），不是两个有限惩罚。程序用普通mean，没有过滤/插补；若发生会向病例/宏平均传播。该结果仅证明当前纯 metric 函数的行为，不证明历史 NumPy 版本；本轮已查现存 COVID病例记录均无单空/非有限值。详见 `METRIC_EMPTY_EDGE_CPU_CHECK.csv`，未修改评价代码。

### A5. 比较真实协议与选择流程：VERIFIED 所列范围

`PER_RUN_PROTOCOLS.csv` 保留每条配置的完整结构；`PROTOCOL_FACTS_BY_ROW.csv` 提取配置/检查点实际证据，不把 derived config 当观测执行。共同使用绑定的划分和对应评价口径，方法特有网络、初始化、优化器和增强按实际记录保留。

HiEn 在所列配置中使用 ImageNet ResNet34 初始化、3通道复制、训练patch224²、自己的监督/特征一致性损失与教师更新。PMEUL使用自己的MCNet双decoder及损失；具体字段逐run入包，不泛称所有行等参数或等初始化。HiEn/PMEUL formal TEST receipts 明确 `test_used_for_selection:false`、严格加载零missing/unexpected，配置SHA匹配。RUN_MATRIX 中早期 PENDING_GATES 标签不作完成证据。

本轮仅CPU读取这些**已选定**的检查点元数据，未重新验证或选模。HiEn的COVID10%/5%测试检查点分别为step3500/5300、角色均为teacher；ACDC10%/5%及PROMISE20%分别为step10500/19800/25500、角色student。PMEUL对应ACDC10%/5%、COVID10%/5%、PROMISE20%为27200/23400/14200/600/8000。UCAD四份已选检查点的`iter`分别为ACDC5%10600、ACDC10%23000、COVID5%3000、COVID10%15600；其旧receipt未记该字段，本轮元数据和当前SHA不能冒充旧receipt绑定。逐份键、SHA、角色及读取边界见 `EXTERNAL_SELECTED_CHECKPOINT_CPU_METADATA.json`。

E5实际初始化 receipt（2026-10-08T20:27:21.020838+08:00）记录 fresh、seed42、随机初始化、预训练0、无教师；官方 commit `85361fb91cf43a69a158d41dfa9e7ee9fce36db9`。AdamW lr1e-4、weight_decay0.01、betas0.9/0.999、无调度；4流×6=24图，12GT，严格63无标注患者；迭代30000，验证200步，student1验证Dice选点27200。官方CT增强保留颜色扰动/自动对比度/直方图均衡以及随机翻转/旋转，适配器只绑定共同数据来源和划分。与 Full 的预训练来源、SGD、配对弱强增强、EMA及目标不同，比较为完整方法层面。

{cite('formal_E4_E5_20261008/tasks/E5/reports/INITIALIZATION_ACTUAL.json',needles=['fresh','random_seed','pretrain','teacher','official_commit','official_entry','batch_definition'])}

{cite('formal_E4_E5_20261008/tasks/E5/protocol.json',needles=['max_iterations','optimizer','lr','weight_decay','primary_selection','scheduler'])}

{cite('formal_E4_E5_20261008/external/AdaMix/exp_acdc/data/dataset.py',lines=[76,78,79,80,87,88,89])}

作者已更正 ACDC 测试未参与候选方法选择。本轮沿用 **验证集选 checkpoint、随后固定 checkpoint测试** 的实际流程，不根据旧rank/成绩排序推断选择因果。保留两个历史文字冲突：X03原TEST receipt（2026-08-19T12:55:05.155768+00:00）写 `"TEST_SELECTION_EXPOSURE": true`；其生成器 `tools/v21_5_acdc10_test_frozen_checkpoint.py:203` **固定写 True**，并非方法选择事件追踪。20260904助手派生汇总写“CLASS_B material TEST contamination and evaluator mismatch”；它也是判断标签，不是新的原始选择事件。相反，正式启动 readiness、Full completion 记录 test_access_count0，入口的 db_test 实际split=val，真正test另由冻结评价入口消费。原字段、身份、时间、原句与生成位置均保留于 `HISTORICAL_ANNOTATION_CONFLICTS.csv`；这些标签不足以推翻作者更正，本轮没有发现新的具体选择事件与其冲突。

{cite(fullroot+'/full_completion_report.json',needles=['test_access_count','validation_points'])}

{cite('tools/v21_5_acdc10_test_frozen_checkpoint.py',lines=[145,203,207])}

## B — ACDC 5%/10% 预训练、初始化与更新预算：PARTIAL

### B1. 现有预训练权重和继承状态：VERIFIED；实际生成历史：NOT_FOUND

label3、label7目录目前各只有 `unet_best_model.pth`。CPU读取均仅含 `net,opt`，无step、seed、args、实际标签清单或选模指标；SGD state都有82个 momentum_buffer、142个参数引用，lr0.01/momentum0.9/weight_decay1e-4；SGD缓冲没有更新计数，不能从内容推回10000步或选模时刻。

label3 SHA256：`{meta[0]['sha256']}`。
label7 SHA256：`{meta[1]['sha256']}`。
实际路径分别为 `{meta[0]['path']}` 与 `{meta[1]['path']}`；本轮CPU元数据与SHA见 `CHECKPOINT_CPU_METADATA.json`，不导出权重。

`beta_fft_label3.py` / `beta_fft_label7.py::pre_train` 的代码目标pre_iterations10000、验证200步、最佳验证Dice保存net/opt，只能证明**代码定义/目标**；两者当前 `__main__` 中 pre_train 均注释，label7脚本还保留labelnum3及label3路径，不能以脚本文件名证明实际生成协议。没有定位到这两份权重对应的原始启动命令、生成log、实际seed、选模步数/分数、真实更新预算及冻结标签子集。因此这些字段保持 NA/NOT_FOUND，不报40000总预算。

{cite('train_ACDC/beta_fft_label3.py',needles=['pre_iterations','def pre_train','save_best_path','pre_train(args','pre_snapshot_path ='])}

{cite('train_ACDC/beta_fft_label7.py',needles=['pre_iterations','default=3','def pre_train','pre_snapshot_path =','pre_train(args'])}

`audit/beta_fft_clean_supervised_pretrain_final_config.json` 明确 `training_started:false`，输出是另一套 clean_label12_seed42；这是后来准备配置，不是label3/label7生成记录。20260618 clean label7审计针对clean split复用，不能将其“不适用clean复用”扩大为主Full无效。

### B2. 半监督实际预算、学生/教师/SGD、LR：VERIFIED

5% Full：labelnum3、seed1337，last global_step30000，student1验证选点28400、Dice0.8892521589382639。
10%主Full：labelnum7、seed42，last global_step30000，student1验证选点29600、Dice0.8999080952059592。

两份 dynamics CSV各30001行，但只有 step1–30000这 **30000个不同更新步**；最后重复的step30000是 `effective_source.py:549` 写入的零loss检查点终结标记，不是第30001次更新。两条轨迹的loss/grad有限性记录均通过。每个循环各执行optimizer1.step、optimizer2.step，故证据支持 **每学生30000次半监督更新**、30000次student1→EMA更新，不能把两学生计数相加当一个60000-step训练预算；这是“完整轨迹＋入口＋checkpoint”证据，不是独立optimizer调用计数器。

last 含 model_1、model_2、ema_model、两个完整SGD、epoch/global_step、best_metrics、random_state、args；EMA=0.99。新run加载label3或label7 `net` 到两学生和教师，TRC新参数按机制初始化；两学生继承pretrain opt的base momentum/超参数，TRC参数单列组。保存状态各optimizer90 momentum_buffers，group0参数引用142、group1 8。没有丢弃SGD历史状态后假称完全从头初始化。

原Full及ACDC5有效入口**没有LR schedule更新语句**；pretrain opt记录lr0.01，base组继承该值、router组由args.base_lr0.01构建，两份last各组仍lr0.01。该运行证据对应 **恒定0.01**，不能沿用其他入口的poly说法。

{cite(fullroot+'/effective_source.py',lines=[21,22,276,277,279,291,292,315,316,317,318,320,321,443,444,445,461,549])}

{cite('train_ACDC/v14_mask_adamix_components.py',lines=[159,169,176,180,182,185,187,193])}

ACDC5 derived config中“30000 inferred from run family”原句保留；本轮确认30000来自独立真实log、连续轨迹和CPU last读取，不引用那句推断作为运行预算。

{cite(five+'/train.log',lines=[1,3,61219,61224])}

**5%总预算仍 NA**：已经确认半监督30000，但label3实际预训练总更新数未找回；写成“pretrain未知＋self-train30000”，不能以默认10000补成总40000。

### B3. 标签子集：VERIFIED 当前SSL重建；PARTIAL 历史绑定；NOT_FOUND 预训练子集

当前实际引用train_slices.list共1312切片/70患者，val20ED/ES帧/10患者，test40帧/20患者。入口patients_to_slices固定5%首68片、10%首136片。按该现存manifest CPU重建：

5%：patient038、patient050、patient099（68片）；10%：patient021、patient038、patient049、patient050、patient058、patient099、patient100（136片）。逐slice索引和当前manifest SHA见 `ACDC_SSL_LABEL_PREFIX_RECONSTRUCTION.csv`。这是**现存SSL入口/manifest的重建**；原Full checkpoint没有保存train manifest hash，不能由此证明最初pretrain也用这套标签清单；历史生成子集仍NOT_FOUND。

## C — 主 Full 增强/DiceLoss/CTW 与 COVID 适配器/FFT 的历史版本：PARTIAL

### C1. 归档入口与CTW机制：VERIFIED

原主Full和ACDC5有效入口当前SHA一致：`{sha(ROOT/fullroot/'effective_source.py')}`。它证明运行目录生成入口、参数和调用位置，**不自动证明所有 import 内容**。CTW组件有20260817预启动冻结SHA，当前文件与该值完全相同，实际training log持续记录X03 loss/weighted loss端点：

CTW：`{ROOT/'train_ACDC/v21_3_acdc10_third_innovation_components.py'}`。
SHA256 `f28abc48fedff330f9aa7470072ebc16a607c6c59fd14942c2adb111b8ce748a`。

具体计算：q为两学生detach概率的最大类置信度均值；对q作水平/垂直绝对后向差分，首位置0；逐图min–max、分母下限1e-6得到r；w=0.25+r，范围[0.25,1.25]；误差是p1与detach(p2)的类别平均概率L1，空间batch取mean，外系数0.05，只由这项更新student1。不是教师置信度梯度图、不是全边界GT监督。

{cite('train_ACDC/v21_3_acdc10_third_innovation_components.py',lines=[28,32,35,38,39,67,68,99,100,101])}

{cite('audits/acdc10_third_innovation_codex_prep_top6_v21_3/ROUND1_TOP6_CONTROLLER_DRY_CONSUMPTION.json',needles=['created_at','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT','source_sha256'])}

### C2. 增强与DiceLoss静态定义：VERIFIED；原 Full 导入绑定：PARTIAL

主入口import `dataloaders.dataset.WeakStrongAugment` 与 `utils.losses.DiceLoss`。当前二者与20260907已有证据快照逐byte相同，但这份依赖快照晚于原Full运行；未定位到原Full启动时针对两文件的模块SHA/依赖副本/loaded __file__+SHA收据。

现存2D增强：weak共享图/标签随机90°旋转及翻转，或[-20,20)角度order0旋转；图与标签nearest到256²。strong在该共享几何上加cutout(p0.5，面积比例0.02–0.4)和ColorJitter(0.8,0.8,0.8,0.2)；cutout区标签写背景0。不能凭args.magnitude6/s_param6推断该路径用了另一套RandAugment。

现存DiceLoss对batch+空间维求和，按四类含背景等权平均；分母为 `sum(p²)+sum(y²)`，非mask smooth1e-5、mask smooth1e-10。mask先按类别复制，分别施加到交集和两分母项；不能把它写成线性分母Dice或只前景Dice。训练DiceLoss与测试Dice统计定义不同。

{cite('train_ACDC/dataloaders/dataset.py',lines=[218,229,245,250,254,278,440,443,446,449,450])}

{cite('train_ACDC/utils/losses.py',lines=[156,169,171,173,174,183,186,188,193,203,208,211])}

SHA分别 `{sha(ROOT/'train_ACDC/dataloaders/dataset.py')}` 和 `{sha(ROOT/'train_ACDC/utils/losses.py')}`；`SOURCE_VERSION_BINDING.csv` 区分静态定义、晚期依赖归档和原run绑定。论文可准确描述“可核查实现”，但不要写“所有历史依赖已完全冻结证明”。

### C3. COVID5/10实际入口：VERIFIED；FINAL3 导入依赖：PARTIAL

已读取并核验两份实际归档入口：

`{ROOT/'experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/train_promise12_bcp_fft.py'}`；
`{ROOT/'experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/train_promise12_bcp_fft.py'}`。

二者SHA均 **e1006ebe9731d7665a886dc5f897f130c5818c132fa575cc16c9ea845cb27386**，与20260819正式启动manifest相同。不是“找不到COVID训练入口”。入口导入COVID dataset adapter，并从 `utils.displacement` 调用逐对应样本FFT；归档入口自身self_train lineage只写training_core/pretrain/manifest SHA，没有写所有导入模块SHA。

还找回早期COVID seed42家族的 `audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json`：adapter SHA `{sha(ROOT/'train_COVID19/covid19_dataset_adapter.py')}`，FFT helper SHA `{sha(ROOT/'train_promise12/utils/displacement.py')}`，均与当前文件一致。20260812 seed1337 critical source清单也包含adapter/hash；但**早期family的绑定不能自动移给COVID5/10_FINAL3**，尚缺FINAL3当时的依赖副本或runtime imported-file/hash收据。

{cite('audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json',needles=['authorization_timestamp','formal_full_runs','dataset_adapter_sha256','fft_helper_sha256','training_core_sha256'])}

当前所附adapter强视图为 `image_batch.clone()`、label也clone；在共用BCP mask及配对索引下两份mix输入重复。FFT helper按同一i交换对应样本的低频部分，阈值30；若两输入相同，则交换项相消，不能声称当前这条链产生独立频域扰动。入口随后仍让两学生在该额外输入上做互监督。**静态行为已确认，早期绑定已有SHA；将其确定归给FINAL3全历史仍受上述import版本缺口限制**。不修改算法、不重训、不否定其他已核实模块。

{cite('train_COVID19/covid19_dataset_adapter.py',lines=[167,174,176,177,178,179])}

{cite('train_promise12/utils/displacement.py',lines=[32,50,52,53,59,60,63,64,73])}

{cite('experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/train_promise12_bcp_fft.py',lines=[29,2323,2443,2449,2929,2930,2933,2937,2938])}

## D — R2 fixed92 的120条记录及20患者表：VERIFIED，可关闭

原始文件原样入包：

{cite(r2+'/results/case_class_metrics.csv',lines=[1,2,121])}

{cite(r2+'/results/R2_patient_metrics.csv',lines=[1,2,21])}

120个独立(case,class)键；40个ED/ES帧体、20患者；每类40行，每患者2帧×3类=6行。20患者表逐项与原120行的分组均值相比，最大误差 **{r2check['patient_table_max_error']}**；所有数值有限。120行宏平均、3类宏平均、40帧均值及20患者均值在浮点误差内一致（每患者恰好两帧，因此等权患者均值与等权帧均值一致，不推广到不均衡帧数队列）。

原receipt未舍入结果：Dice **0.9020444131484706**、IoU **0.8270690413278287**、HD95 **1.880012540213665 grid**、ASD **0.5087759056993733 grid**。学生1验证选点29000，checkpoint SHA **b56292a270ccf6679b5597e9d556eda686df91778e496e715174aaa367b710de**。论文显示90.20/82.71/1.880/0.509；对89.98控制的Dice差 **+{r2check['delta_Dice_pp']:.15f}个百分点**，不归到原主Full动态TCSMix。

`R2_RECORD_VERIFICATION.json`、`R2_PATIENT_RECOMPUTATION.csv`、`R2_CLASS_RECOMPUTATION.csv` 和 `AGGREGATE_RECOMPUTATION.csv` 提供逐行检查。两个原始CSV有完整SHA、包内路径；无需登录服务器或取得影像/权重即可复核这项统计。

## 可直接采用的论文句子（按证据范围）

1. “论文主 Full（R1）为原始动态TCSMix完整方法；其ACDC10%成绩为Dice90.35%、IoU82.93%、HD951.423和ASD0.429，距离为重建网格单位。”
2. “更换混合控制信号的89.98%运行是普通残差替代TRC、均匀权重替代CTW及fixed92实验的共同配对参照；这组结果不用于估计原主Full的单因素模块增益。”
3. “ACDC5%和10%完整方法均完成30000次半监督联合迭代，每学生每迭代更新一次；使用验证集学生1平均前景Dice选定检查点，固定该检查点后报告测试结果。”
4. “ACDC完整方法继承既有监督预训练权重及base SGD动量状态；可核查半监督入口/末尾状态对应lr0.01和EMA0.99。预训练生成日志及其确切更新预算未在本轮定位到，故不将推断的预训练步数计作已核实总预算。”
5. “ACDC和PROMISE12的ASD按prediction到GT的单方向表面距离计算；COVID-19的ASD则为双方向表面样本合并后的均值，跨任务的距离统计定义分别说明。”
6. “COVID-19病例按完整轴向序列重建并作患者宏平均；XY由512或630缩至256而Z不变。修正评价仅更新XY spacing，原预测与GT不变；原网格重评价另列，不能与纯spacing修正混用。”此句须与明确选择的结果版本配套，不能据此把未修正的外部行标成修正mm。
7. “AdaMix-CT在共同划分、联合迭代预算和验证/测试口径下比较，保留其随机初始化、AdamW与方法规定的增强；因此结果是完整方法比较。”
8. “R2 fixed92包含20患者的40个ED/ES帧体及120个类别记录；对共同混合控制信号参照的Dice差为+0.219989个百分点，单次实验不证明多随机种子显著性。”
9. “现存COVID适配实现的两视图由clone得到，配对FFT交换不产生独立输入扰动，额外两学生互监督仍保留。”这句表述**可核查实现**；若正文要无条件归于FINAL3历史全过程，依赖绑定缺口需先关闭。

## 剩余材料与收尾边界

仅需补**已有记录**：当前论文表2–4逐行方法/预算名单；label3/label7实际生成与选模日志/命令/label manifest/update budget；原Full增强/DiceLoss及COVID_FINAL3 adapter/FFT的历史导入快照或加载SHA收据；HiEn/PMEUL/UCAD既有spacing修正结果（若不存在，当前记NOT_FOUND）。不建议用新训练补造历史来源。

本轮已关闭D、ACDC5/10半监督预算与状态、PROMISE基线精度复算、所列病例记录均值、实际单空处理、COVID入口及几何定位、CTW组件SHA链。A/B/C总体仍PARTIAL，缺失项不填推断。作者需要决定论文距离表采用哪一个已明确的评价版本，并将最终逐行名单与本索引对应；未据测试成绩选版本或改参数。

证据包入口为本文件；索引 `SOURCE_INDEX.csv`，四项状态 `CLOSURE_DECISIONS.csv`，完整性清单 `SHA256SUMS.txt`。只包含必要小型源码/片段、配置、receipt、原病例统计、复算和校验记录；不含影像、预测数组、权重、数据集、凭据或其他项目材料。
'''
(OUT/'REVIEW_CLOSURE_EVIDENCE.md').write_text(report)
csvout('SOURCE_INDEX.csv',index)

# Source citations are backed by exported files where possible; missing generation records stay explicit.
coverage=[
    {'scope':'Final current manuscript Table2-4 exact row roster','status':'NOT_FOUND','search':'given final_report tables and bound comparison handoffs; v23 numbering differs from current manuscript', 'not_inferred':True},
    {'scope':'label3/label7 actual pretrain generation command/log/selected step/labels/update budget','status':'NOT_FOUND','search':'train_ACDC/Pre_model label3/7, source entries, project pretrain audit/command records and related runs; only net,opt remain','not_inferred':True},
    {'scope':'Main Full historical augment/DiceLoss imported module receipt','status':'PARTIAL','search':'X03 run archives, contemporary candidate audits, later draft8 dependency archive, current source','not_inferred':True},
    {'scope':'COVID_FINAL3-specific adapter/FFT historical loaded SHA','status':'PARTIAL','search':'archived e1006e entries, transfer launch manifests, prior COVID source identity and current snapshot; earlier family hashes found','not_inferred':True},
]
csvout('SEARCH_AND_MISSING_EVIDENCE.csv',coverage)
(OUT/'README_CN.md').write_text('本包只处理A–D四项收尾证据。先读 REVIEW_CLOSURE_EVIDENCE.md，再查 CLOSURE_DECISIONS.csv 与 SOURCE_INDEX.csv。\n原路径用于溯源，packed_path 可在解压后直接打开。INDEX_ONLY / CPU_METADATA_ONLY_WEIGHT_EXCLUDED 文件不随包导出；其中检查点仅登记SHA/元数据。\nscripts/ 中都是只读CPU核查脚本；新输出仅写本目录。不要将当前CPU复算当作历史运行记录。\n')
print('Report written:',OUT/'REVIEW_CLOSURE_EVIDENCE.md')
print('Sources',len(index),'report bytes',len(report.encode()))
