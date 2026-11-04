#!/usr/bin/env python3
import csv, gzip, hashlib, json, re, shutil, subprocess
from pathlib import Path
TASK=Path(__file__).resolve().parents[1];CAT=TASK/'catalog'
def save(n,x):(CAT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
weights=json.loads((CAT/'weights.json').read_text());files=json.loads((CAT/'files.json').read_text());experiments=json.loads((CAT/'experiments.json').read_text());audit=json.loads((CAT/'reports/checkpoint_audit.json').read_text())
bysha={w['sha256']:w for w in weights};bindings={h:[] for h in bysha}
for f in files:
    name=f['backup_path'];p=CAT/name
    if p.stat().st_size>3*1024**2 or not any(s in name.lower() for s in ['result','receipt','evaluation','report','manifest']):continue
    try:
        content=(gzip.open(p,'rt').read() if f.get('compression')=='gzip' else p.read_text())
        for h in set(re.findall(r'\b[0-9a-f]{64}\b',content)) & bysha.keys():bindings[h].append(name)
    except (UnicodeError,OSError):pass
for e in experiments:
    own=[w for w in weights if e['id'] in w['experiment_ids'] and w['source'].startswith(e['run_root']+'/')]
    last=next((w for w in own if w['source'].endswith('last.ckpt')),None) or next((w for w in own if 'full_state_final' in w['source']),None)
    direct=[f for f in files if f.get('source','').startswith(e['run_root']+'/')]
    e['records']=[f['backup_path'] for f in direct]
    aliases={'acdc5':'ACDC5_FINAL3','acdc10':'V21_3_X03_CONTOUR_RELIABILITY_WEIGHT','promise20':'PROMISE20_FINAL3','covid5':'COVID5_FINAL3','covid10':'COVID10_FINAL3'}
    alias=aliases.get(e['id'],e['id'])
    stdout=[f for f in files if '/controller_formal_logs/' in f.get('source','') and alias.lower() in Path(f['source']).name.lower()]
    e['training_logs']=[f['backup_path'] for f in direct+stdout if f['backup_path'].endswith(('.log','.log.gz','log.txt','log.txt.gz'))]
    e['source_snapshots']=[f['backup_path'] for f in direct if f['backup_path'].endswith('.py')]
    e['configs']=[f['backup_path'] for f in direct if f['backup_path'].endswith(('.json','.yaml','.yml'))]
    if e.get('portable_config'):e['configs'].append(e['portable_config'])
    if last:
        args=audit[last['sha256']].get('args') or {}
        if args:
            e['recorded_module_arguments']={k:v for k,v in args.items() if any(t in k.lower() for t in ['mask_token','adamix','innovation','ctw','control','rectangle','residual'])}
            if e.get('modules') is None:
                innovation=args.get('innovation_x_id')
                e['modules']={'TRC':args.get('mask_token_enabled'),'TCSMix':args.get('adamix_enabled'),
                              'CTW':bool(args.get('enable_v21_3_innovation_x')) and innovation=='V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',
                              'recorded_innovation_id':innovation,'recorded_coefficient':args.get('v21_3_innovation_x_weight')}
        e['last_training_state']={'source':last['source'],'sha256':last['sha256'],'iteration':audit[last['sha256']].get('iteration'),
                                  'status':audit[last['sha256']].get('resume_status')}
        configpath=CAT/'derived_configs'/(e['id']+'.json');configpath.parent.mkdir(exist_ok=True)
        configdata={'origin':'Checkpoint args and lineage export; not a newly invented training record','checkpoint':last['source'],
                    'checkpoint_sha256':last['sha256'],'args':audit[last['sha256']].get('args'),'lineage':audit[last['sha256']].get('lineage'),
                    'portable_config':e.get('portable_config')}
        configpath.write_text(json.dumps(configdata,ensure_ascii=False,indent=2)+'\n')
        bp=str(configpath.relative_to(CAT));e['configs'].insert(0,bp)
        files=[f for f in files if f['backup_path']!=bp]
        files.append(dict(source='derived_configs/'+configpath.name,backup_path=bp,sha256=sha(configpath),size=configpath.stat().st_size,
                          experiment_ids=[e['id']],kind='derived_configuration',association_basis='Export of actual checkpoint metadata; not an original config file'))
    e['weight_evidence_bindings']=[{'weight':w['source'],'sha256':w['sha256'],'records_containing_exact_sha':sorted(set(bindings[w['sha256']]))} for w in own]
    e['code_version']['snapshots']=[{'path':f['backup_path'],'source_sha256':f.get('source_sha256'),'backup_sha256':f['sha256']} for f in files if f['backup_path'] in e['source_snapshots']]
    # Every run-specific model object is linked to its own log/config/source records. Shared material remains marked as shared context.
    for w in own:w['run_records']=e['records']
save('experiments.json',experiments)
save('files.json',files)
# Avoid repeating the long text index inside each weight. The explicit experiment IDs are the join key.
for w in weights:w.pop('run_records',None)
save('weights.json',weights)
with (CAT/'experiments.csv').open('w',newline='') as out:
    fields=['id','priority','dataset','labeled_ratio','seed','modules','run_root','expected_best_sha256','paper_best_hash_status','validation','paper_metrics','reported_test','last_training_state','weights','training_logs','configs','source_snapshots','notes']
    writer=csv.DictWriter(out,fields,extrasaction='ignore');writer.writeheader()
    for e in experiments:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in e.items() if k in fields})
with (CAT/'weights.csv').open('w',newline='') as out:
    fields=['source','sha256','size','object','asset','experiment_ids','backup_time','source_mtime','stable_copy','cpu_load','iteration','resume_status']
    writer=csv.DictWriter(out,fields,extrasaction='ignore');writer.writeheader()
    for w in weights:
        row=dict(w);row.update({k:audit[w['sha256']].get(k) for k in ['cpu_load','iteration','resume_status']});row['experiment_ids']=';'.join(row['experiment_ids']);writer.writerow(row)
for p in (TASK/'tools').glob('*.py'):shutil.copyfile(p,CAT/'tools'/p.name) if (CAT/'tools').exists() else None
collection=json.loads((CAT/'reports/collection.json').read_text());collection.update(text_files=sum(f['kind']=='text' for f in files),text_and_scalar_references=len(files),
    unique_text_and_scalar_contents=len({f['sha256'] for f in files}),stored_catalog_content_bytes=sum((CAT/f['backup_path']).stat().st_size for f in files),
    scalar_source_files=41,scalar_rows=sum(x['scalar_rows'] for x in json.loads((CAT/'reports/tensorboard_audit.json').read_text())),
    cpu_load_pass=sum(x['cpu_load']=='PASS' for x in audit.values()),cpu_load_total=len(audit),paper_metrics_reproduced=False)
save('reports/collection.json',collection)
limitations={
 'release_upload':'Pending GitHub API login. SSH push works, but no Release or remote asset exists yet. Planned asset URLs are not verified downloads.',
 'paper_metrics':'No training, inference, validation, or test evaluation was run. Existing result receipts were preserved and five selected checkpoint hashes matched.',
 'original_git':'Training tree has no .git. Original training commits and historical uncommitted diffs cannot be reconstructed. Publication baseline and current differences, plus per-run source hashes, are saved.',
 'pretraining':'label3/label7 initialization bytes are saved and bound. Their original generating commands, logs and actual pretraining update counts were not found in the existing confirmed records.',
 'historical_dependencies':'Current dependency files are archived. Historical Full augmentation/DiceLoss and COVID FINAL3 adapter/FFT dependency bindings remain limited by existing evidence; see retained author review.',
 'resume':'Main last states are at 30000 completed updates. No original-budget updates remain. ACDC best full states (28400/29600) and binary 25000 full states are also saved for recovery within the original 30000-update budget.',
 'exactness':'ACDC worker RNG/prefetch was not fully checkpointed; no claim of bitwise exact resumption. Binary sampler/RNG and formula-LR state are present, but no training step was executed in this backup task.',
 'candidates':'E1 is an interrupted candidate; aborted ARM100 is distinct from the evaluated fresh restart. Both retained as candidates, never promoted to paper results.',
 'controls':'R1_v2_B/formal_R1 is the independent control-signal replacement (89.98%), separate from original Full/paper R1 (90.35%).',
 'mechanism':'User-specified 2026-10-17 CTW directional summary differs from retained computed CSV ordering. Both provenance states remain as documented by the author-confirmed file.',
 'privacy':'No raw images, labels, NIfTI/HDF5/NPY/NPZ data, image figures, event binaries, .env, authentication caches or private keys are included. Public dataset case IDs are retained only for reproducible split/metric joins. Email redaction is documented.',
 'comparison_rows':'This backup does not invent the missing external-comparator run/checkpoint mappings reported in the historical review.'}
save('reports/limitations.json',limitations)
rows=['| 主实验 | 论文最佳步数 | 论文 Dice (%) | 最后训练状态 | 日志/配置 |','|---|---:|---:|---|---|']
for e in experiments:
    if e['priority']=='P0':rows.append(f"| {e['id']} | {e['validation']['step']} | {float(e['paper_metrics']['dice'])*100:.4f} | {e['last_training_state']['iteration']}；{e['last_training_state']['status']} | 已保存；见 experiments.json |")
readme='''# CoMixW 关键实验备份 · 2026-11-04

本目录保存 5 个主实验和关键消融/对照的清单、代码、脱敏日志、配置、指标原始表及恢复工具。大型权重放在本次独立 Release；在 API 登录完成前，附件仅保留本地，链接标为计划地址，不表示已上传。

原训练目录只读，未删除文件、未修改训练逻辑、未启动训练或评估。权重均在稳定后复制并核对 SHA-256。`PAPER_AUTHOR_CONFIRMED_FACTS.md` 的固定结果优先于旧历史记录：COVID 主 Full HD95 为 22.79/11.88 mm；原始 ACDC10 Full 为 90.35%，控制信号替换为 89.98%。

'''+ '\n'.join(rows)+'''

`experiments.csv/json`：实验、比例、种子、模块、验证/测试来源、训练日志、配置、实际源代码和恢复状态。`weights.csv/json`：原始权重路径、对象 SHA、大小、附件、实验引用和 CPU 检查。`files.json`：各文本/标量文件的来源、原 SHA、备份 SHA、压缩方式及实验关联。共享历史材料有明确标记，不能当作每个实验实际使用它的证明。

`reports/checkpoint_audit.json` 和 `reports/model_compatibility.json` 分别记录 CPU 反序列化与主实验严格模型结构加载；`reports/tensorboard_audit.json` 记录事件内容检查。文件校验、模型加载、论文指标复现是三个独立状态。本次没有重跑评估，不声称复现了论文指标。

原训练目录没有 Git 历史；出版基线 commit 和可取得的差异见 `reports/git_provenance.json`。历史依赖、初始化生成记录、未完成候选和恢复限制见 `reports/limitations.json`。所有原始医学数据须从数据提供方另行取得。

## 新服务器下载、验证和恢复文件

以下命令以仓库根目录为起点；仅在 Release 上传成功后，`--download` 才能取得附件。当前可从原备份服务器复制 `assets/` 目录使用相同命令验证恢复。

```bash
git clone --branch backup/comixw-critical-2026-11-04 https://github.com/rgm-medseg/Mixing-Meets-Weighting.git
cd Mixing-Meets-Weighting
COMIXW_CATALOG="$PWD/backups/2026-11-04"
python "$COMIXW_CATALOG/tools/restore_backup.py" --catalog "$COMIXW_CATALOG" --assets "$PWD/backup-assets" --download --verify-only
python "$COMIXW_CATALOG/tools/restore_backup.py" --catalog "$COMIXW_CATALOG" --assets "$PWD/backup-assets" --destination "$PWD/restored"
```

`restore_backup.py` 验证全部附件 SHA 后解包，再核对每个对象 SHA，恢复原相对路径。重复权重使用硬链接引用同一对象。文本中的大日志/CSV 以确定性 gzip 保存，恢复时自动解压。当前各附件均小于 2 GiB，无须分卷拼接。

## 环境与评估

原环境为 Python 3.10、PyTorch 2.4.1+cu121；历史及本次环境快照均已保存。先安装匹配硬件的 PyTorch 和出版代码依赖，再准备原划分与预处理格式的数据。

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r restored/publication/requirements-core.txt
export DATA_ROOT=/datasets
cd restored/publication
python tools/evaluate_paper.py --config configs/paper/acdc5.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3/checkpoints/best.ckpt --data-root "$DATA_ROOT" --output-dir /work/eval/acdc5
python tools/evaluate_paper.py --config configs/paper/acdc10.json --checkpoint ../project/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt --data-root "$DATA_ROOT" --output-dir /work/eval/acdc10
python tools/evaluate_paper.py --config configs/paper/promise20.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/promise20
python tools/evaluate_paper.py --config configs/paper/covid5.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/covid5
python tools/evaluate_paper.py --config configs/paper/covid10.json --checkpoint ../project/experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/unet_best_model.pth --data-root "$DATA_ROOT" --output-dir /work/eval/covid10
```

`DATA_ROOT` 下应有 `ACDC/`、`PROMISE12/`、`COVID19_OFFICIAL20/`；具体预处理和数据来源见 `records/publication/README.md`、`tools/prepare_data.py` 及保留的原脚本。上述评估命令未在本次任务执行；请将新结果另存，不覆盖原始 receipt。

## 恢复训练

五组最后状态均为 30000 步，已完成原预算。下面工具默认选择仍有原预算更新的已保存状态：ACDC 验证最佳完整状态；PROMISE/COVID 第 25000 步完整状态。先打印实际 argv、初始化复制和数据链接计划，再明确加入 `--execute` 执行。

```bash
cd /work/Mixing-Meets-Weighting
COMIXW_CATALOG="$PWD/backups/2026-11-04"
python "$COMIXW_CATALOG/tools/resume_main.py" --catalog "$COMIXW_CATALOG" --restored "$PWD/restored" --setting covid5 --data-root /datasets --output-root /work/resumed
python "$COMIXW_CATALOG/tools/resume_main.py" --catalog "$COMIXW_CATALOG" --restored "$PWD/restored" --setting covid5 --data-root /datasets --output-root /work/resumed --execute
```

`--setting` 可选 `acdc5/acdc10/promise20/covid5/covid10`。工具使用保留的历史训练入口、原 run identity 和原 30000 步预算，并检查二分类训练核心、初始化与清单 SHA；输出必须是新目录。ACDC 缺少完整 worker RNG/prefetch 状态，不能承诺逐步一致。恢复工具的参数构造和 CPU 状态已检查，本次没有执行训练恢复。

## Release 上传重试

SSH 只能推送 Git，创建/上传 Release 还需要 GitHub API 登录。工具从 `GH_TOKEN/GITHUB_TOKEN`、GitHub CLI 登录或用户私有凭据文件读取；凭据不进入本目录。`--login` 会在终端隐藏输入并保存到备份外的权限 600 文件。

```bash
python backups/2026-11-04/tools/upload_release.py --catalog backups/2026-11-04 --assets /path/to/assets --download-cache /path/to/remote-verification --login
```

上传工具核对账号与仓库写权限、远端备份 commit 和 tag；不覆盖已有附件。全部附件上传后重新从 GitHub 下载逐项核对 SHA，并实测解包恢复。结果写入 `reports/remote_verification.json`；存在失败即报告并保留已上传附件和本地包，不购买存储服务。
'''
(CAT/'README.md').write_text(readme)
print(json.dumps(collection,ensure_ascii=False,indent=2))
