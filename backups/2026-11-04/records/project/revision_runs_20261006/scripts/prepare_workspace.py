"""Freeze local sources and resolved protocol; no historical output is modified."""
import ast
import json
import re
import shutil
import socket
import subprocess
from common import *

def main():
    for name in ('train_ACDC', 'train_COVID19'):
        shutil.copytree(PROJECT/name, WORK/'source'/name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pth','Pre_model'))
    shutil.copy2(FULL/'effective_source.py', WORK/'source/historical_full.py')
    shutil.copy2(BASE/'source_snapshot.py', WORK/'source/historical_baseline.py')
    for task, run, trc in [('Historical_Full',FULL,True),('Historical_baseline',BASE,False)]:
        cfg = dict(task=task, model_family='beta_fft', trc='token' if trc else 'off',
                   checkpoint=str(run/'checkpoints/best.ckpt'), dataset_root=str(DATA),
                   prediction_dir=str(WORK/f'runs/{task}/evaluation/predictions'),
                   evaluation_dir=str(WORK/f'runs/{task}/evaluation'), seed=42, max_iterations=30000)
        atomic_json(WORK/f'configs/{task}.json',cfg)
    first = (FULL/'train.log').read_text().splitlines()[0]
    match = re.search(r'Namespace\((.*)\)$', first)
    parsed = ast.parse('f(' + match[1] + ')',mode='eval').body
    args = {kw.arg:ast.literal_eval(kw.value) for kw in parsed.keywords}
    slices = (DATA/'train_slices.list').read_text().splitlines()
    ids = sorted({s.split('_')[0] for s in slices[:136]})
    all_train = sorted({s.split('_')[0] for s in slices})
    split_ids={s:sorted({x.split('_')[0] for x in (DATA/f'{s}.list').read_text().splitlines()}) for s in ('val','test')}
    if len(ids)!=7 or any(set(all_train)&set(v) for v in split_ids.values()) or set(split_ids['val'])&set(split_ids['test']):
        raise RuntimeError('ACDC split/patient contract failed')
    protocol=dict(dataset_root=str(DATA), original_namespace=args, seed=42,
                  labeled_patients=ids, labeled_slice_count=136, train_patients=all_train,
                  val_patients=split_ids['val'], test_patients=split_ids['test'],
                  updates=30000, batch=24, labeled_batch=12, image_size=[256,256],
                  validation_interval=200, selection='strictly higher foreground macro Dice on val; student 1',
                  optimizer='SGD momentum .9, weight_decay 1e-4; transplant pretrain optimizer state',
                  base_lr=.01, lr_schedule='constant after pretrain optimizer transplant (historical loop has no LR update)',
                  pretrain=str(PROJECT/'train_ACDC/Pre_model/label7/unet_best_model.pth'),
                  pretrain_sha256=sha(PROJECT/'train_ACDC/Pre_model/label7/unet_best_model.pth'),
                  pretrain_history='NA: exact generating log/command not located; file net/opt reused identically, official pretrain function is only a code clue',
                  precision='FP32; no AMP in historical parent', ema=.99, ctw_lambda=.05,
                  ctw_participating_pixels='all mixed pixels, including cutout; no exclusion mask',
                  evaluation=dict(id='ACDC_v22_grid_directed_ASD', dice_iou='fractions; macro over RV/MYO/LV and frame volumes',
                                  surface_unit='original H5 reconstruction grid, spacing=(1,1,1)',
                                  empty_both='all metrics 0', empty_one='Dice/IoU 0, HD95/ASD volume-grid diagonal',
                                  asd='MedPy directed prediction-to-GT ASD', resize='nearest order=0 input and output', postprocess='none'),
                  resume_boundary='main RNG and sampled batch IDs recoverable; 16 worker RNG/prefetch not fully recoverable, not bitwise exact',
                  author_correction='ACDC test not used to choose candidate methods; old rank labels do not establish selection causality',
                  external_sources=dict(AdaMix='https://github.com/Senyh/AdaMix',paper='https://arxiv.org/abs/2407.21586'))
    atomic_json(WORK/'protocol.json',protocol)
    atomic_json(WORK/'resolved_paths.json',dict(host=socket.gethostname(),project=str(PROJECT),workspace=str(WORK),
                python=str(PYTHON),data=str(DATA),full=str(FULL),baseline=str(BASE),gpu=0,persistence='NFS /home',created=now()))
    scale=read_json(WORK/'configs/fixed_scale.json')
    manifest={'created':now(),'session':'beta_fft_revision_20261006','gpu':0,'tasks':{},'analyses':{},'archive_status':'PENDING'}
    for task, name, weight, mix, trc in [('E1','uniform','uniform','dynamic','token'),('E2','fixed','spatial','fixed','token'),
                                        ('E3','fixed_uniform','uniform','fixed','token'),('E4','bottleneck','spatial','dynamic','bottleneck'),
                                        ('E5','adamix_ct','official','official','off')]:
        cfg=dict(task=task,model_family='official_adamix_ct' if task=='E5' else 'beta_fft',weight_mode=weight,
                 mix_mode=mix,trc=trc,fixed_side=scale['fixed_side'],bottleneck_width=18,
                 dataset_root=str(DATA),seed=42,max_iterations=30000,validation_interval=200,save_interval=200,
                 batch_size=16 if task=='E5' else 24, labeled_bs=16 if task=='E5' else 12,
                 pretrained_model=protocol['pretrain'],run_root=str(WORK/'runs'/task),num_workers=2 if task=='E5' else 16,
                 protocol_hash=sha(WORK/'protocol.json'),precision='fp32',
                 official_commit=subprocess.check_output(['git','-C',str(WORK/'external/AdaMix'),'rev-parse','HEAD'],text=True).strip() if task=='E5' else None,
                 budget_note='30000 co-training updates, scratch init, no pretraining, official AdamW 1e-4, two main+two auxiliary loaders, each B16' if task=='E5' else '30000 SSL updates, reuse exact historical pretraining file',
                 estimated_seconds=None)
        path=WORK/f'configs/{task}_{name}.yaml'
        atomic_json(path,cfg)
        cmd=[str(PYTHON),str(WORK/'scripts/train_entry.py'),'--config',str(path)]
        manifest['tasks'][task]=dict(status='PENDING',command=cmd,cwd=str(WORK),interpreter=str(PYTHON),gpu=0,
                config=str(path),config_sha256=sha(path),inputs=[str(WORK/'protocol.json'),str(DATA/'train_slices.list'),
                str(DATA/'val.list')]+([protocol['pretrain']] if task!='E5' else [str(WORK/'external/AdaMix')]),
                output=str(WORK/'runs'/task),budget=cfg['budget_note'],pid=None,latest_step=0,retries=0)
    atomic_json(WORK/'run_manifest.json',manifest)
    sources=[FULL/'effective_source.py',FULL/'train.log',BASE/'train.log',BASE/'source_snapshot.py',
             PROJECT/'train_ACDC/v14_mask_adamix_components.py',PROJECT/'train_ACDC/v21_3_acdc10_third_innovation_components.py',
             PROJECT/'train_ACDC/dataloaders/dataset.py',PROJECT/'tools/v22_0_acdc_test_frozen_checkpoint.py',
             PROJECT/'tools/v21_5_acdc10_test_frozen_checkpoint.py',PROJECT/'train_COVID19/covid19_dataset_adapter.py',
             PROJECT/'train_COVID19/covid19_volume_evaluator.py',PROJECT/'tools/v22_1_postfull_test_launcher.py',
             PROJECT/'final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv',
             PROJECT/'final_report/paper_internal_evidence_v23_0/X03_WEIGHT_SENSITIVITY_FULL_PRECISION.tsv',
             PROJECT/'configs/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_001.json',
             PROJECT/'final_report/PAPER_COMPARISON_TABLE_INPUTS_V19_5_R3.tsv',
             PROJECT/'final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv',
             PROJECT/'final_report/COVID19_FORMAL_TEST_PER_PATIENT.tsv',
             PROJECT/'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/patient_metrics.json',
             PROJECT/'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID10_FINAL3/patient_metrics.json']
    sources += [DATA/f'{s}.list' for s in ('train','train_slices','val','test')]
    sources += [run/'checkpoints'/p for run in (FULL,BASE) for p in ('best.ckpt','last.ckpt')]
    rows=[]
    for p in sources:
        rows.append(dict(path=str(p),sha256=sha(p) if p.exists() else 'NA',kind='checkpoint' if p.suffix=='.ckpt' else 'source',
                         run_id=str(FULL.relative_to(PROJECT)) if str(FULL) in str(p) else str(BASE.relative_to(PROJECT)) if str(BASE) in str(p) else 'NA'))
        if p.exists() and p.suffix not in ('.ckpt','.pth'):
            rel=p.relative_to(PROJECT) if p.is_relative_to(PROJECT) else Path('data_manifests')/p.name
            dest=WORK/'reports/evidence_sources'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest)
    write_csv(WORK/'reports/source_index.csv',rows)
    print(json.dumps(dict(labeled=ids,train=len(all_train),val=len(split_ids['val']),test=len(split_ids['test']),prepared=True)))

if __name__=='__main__':
    main()
