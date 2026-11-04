#!/usr/bin/env python3
"""Read-only source collection. All mutations stay under the backup workspace."""
import csv, hashlib, io, json, os, re, shutil, subprocess, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

TASK = Path(__file__).resolve().parents[1]
ROOT = Path('/home/renguanming/jupyterlab/rgm/beta-FFT-main')
PUB = ROOT.parent / 'beta_fft_publication_20261010/release'
CAT = TASK / 'catalog'
PAY = TASK / 'payload'
for p in [CAT, PAY/'objects', CAT/'records/project', CAT/'records/publication', CAT/'reports']:
    p.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
EXP = []
def add(id, root, priority='P1', dataset='ACDC', ratio='10%', modules=None, evidence=None, notes='', expected_sha=None):
    EXP.append(dict(id=id, run_root=root, priority=priority, dataset=dataset, labeled_ratio=ratio,
                    modules=modules, seed=None, evidence=evidence or [], notes=notes, expected_best_sha256=expected_sha,
                    weights=[], records=[], code_version={'original_git_commit':None,'status':'Original experiment directory has no .git; effective source snapshots and hashes retained.'}))

paper=list(csv.DictReader((PUB/'docs/paper_results.tsv').open(),delimiter='\t'))
mainpaths={
 'acdc5':'experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3',
 'acdc10':'runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',
 'promise20':'experiments/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3',
 'covid5':'experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3',
 'covid10':'experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3'}
for r in paper:
    k=r['setting']; ds='ACDC' if 'acdc' in k else ('PROMISE12' if 'promise' in k else 'COVID-19')
    ratio=('20%' if k=='promise20' else ('10%' if '10' in k else '5%'))
    add(k,mainpaths[k],'P0',ds,ratio,dict(TRC=True,TCSMix=True,CTW=True),
        [r['receipt_source'].split(' (')[0]],expected_sha=r['checkpoint_sha256'])
    EXP[-1].update(paper_metrics={x:r[x] for x in ['dice','iou','hd95','asd','distance_unit']},
                   validation={'step':r['validation_step'],'dice':r['validation_dice'],'source':'publication docs/paper_results.tsv'},
                   seed=42 if k=='acdc10' else 1337,
                   init_path=('train_ACDC/Pre_model/label7/unet_best_model.pth' if k=='acdc10' else
                              'train_ACDC/Pre_model/label3/unet_best_model.pth' if k=='acdc5' else
                              mainpaths[k]+'/pre_train/unet_best_model.pth'),
                   portable_config='records/publication/configs/paper/'+k+'.json')

arms=list(csv.DictReader((ROOT/'final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv').open(),delimiter='\t'))
for r in arms:
    arm=r['Arm']; path=('runs/beta_fft_acdc_official_full_seed42_fresh_ckpt' if arm=='000' else
                       'runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42' if arm=='110' else
                       mainpaths['acdc10'] if arm=='111' else
                       'experiments/three_innovation_ablation_transfer_v22_0/ACDC10_ARM_'+arm+('_fresh_restart_after_prelaunch_hardening' if arm=='100' else ''))
    add('ACDC10_ARM_'+arm,path,modules=dict(zip(['TRC','TCSMix','CTW'],[c=='1' for c in arm])),
        evidence=['final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv'],expected_sha=r['CheckpointSHA'])
    EXP[-1].update(seed=42,validation={'dice':r['BestValDice'],'step':r['BestValIter']},
                   reported_test={k:v for k,v in r.items() if k.startswith('Test')},init_path='train_ACDC/Pre_model/label7/unet_best_model.pth')

for group in ['experiments/paper_internal_evidence_v23_0','experiments/internal_evidence_expansion_v24_0']:
    for p in sorted((ROOT/group).iterdir()):
        if p.is_dir() and 'ACDC' in p.name:
            add(p.name,str(p.relative_to(ROOT)),notes='Coefficient or seed experiment; retain existing best/last and raw evidence, no new selection.',
                evidence=['final_report/'+group.split('/')[-1]])
            EXP[-1]['seed']=int(p.name.rsplit('SEED',1)[1]) if 'SEED' in p.name else 42
            EXP[-1]['init_path']='train_ACDC/Pre_model/label7/unet_best_model.pth'
for p in sorted((ROOT/'runs/acdc10_third_innovation_codex_prep_top6_v21_3').glob('V21_3_X*')):
    if 'X03_' not in p.name:
        add(p.name,str(p.relative_to(ROOT)),evidence=['final_report/acdc10_top6_all_test_v21_5'],notes='Module replacement candidate, original reported evidence retained.')
        EXP[-1].update(seed=42,init_path='train_ACDC/Pre_model/label7/unet_best_model.pth')
controls={
 'R1_v2_B_control_signal':'tcsmix_control_revision_20261006/formal_R1_20261007/runs/R1_full_v2B',
 'R2_fixed92':'tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/runs/R2_fixed92',
 'C1_CTW_uniform':'tcsmix_control_revision_20261006/formal_C1_CTW_uniform_20261008/runs/C1_CTW_mean_uniform_v2B',
 'E4_plain_residual':'formal_E4_E5_20261008/tasks/E4/runs/E4_plain_residual_v2B',
 'E5_AdaMix_common_protocol':'formal_E4_E5_20261008/tasks/E5/runs/E5_AdaMix_CT_common_protocol',
 'F97_main_signal':'mechanism_strengthening_20261013/runs/F97_main_signal_20261013',
 'E1_interrupted_candidate':'revision_runs_20261006/runs/E1',
 'ACDC10_ARM_100_aborted_candidate':'experiments/three_innovation_ablation_transfer_v22_0/ACDC10_ARM_100'}
for k,p in controls.items():
    add(k,p,evidence=[p.split('/runs/')[0]],notes=('UNVERIFIED / incomplete historical candidate; do not use as a paper result.' if 'candidate' in k else 'Independent control experiment; not original Full_main.'))
    EXP[-1].update(seed=42,init_path=None if k.startswith('E5') else 'train_ACDC/Pre_model/label7/unet_best_model.pth')

EXCLUDE=re.compile(r'(MSCMR|ISIC|BRATS|CHAOS|POLYP|MSD_TASK|LA_HEART)',re.I)
SKIP_PARTS={'__pycache__','.git','.claude','.ipynb_checkpoints','archives','versions','upstream','smoke','smoke_archive','isolated_short','scratch_resume_canary','scratch_closed_loop','failure_injection','transfer_canary','transfer_failure_injection','_engineering_stall_archive','external'}
TEXT={'.py','.sh','.json','.jsonl','.yaml','.yml','.toml','.txt','.log','.csv','.tsv','.md','.tex','.list','.cfg','.ini','.patch','.diff','.sha256','.sha256sum'}
paths={}; weightpaths={}; events={}; exclusions=[]; changes=[]
def selected(p):
    return not any(s in SKIP_PARTS for s in p.parts) and not EXCLUDE.search(str(p)) and not p.name.startswith('.env')
def collect(base,ids):
    if not base.exists():return
    seq=[base] if base.is_file() else base.rglob('*')
    for p in seq:
        if not p.is_file() or p.is_symlink() or not selected(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p.relative_to(PUB)):continue
        if p.name.startswith('events.out.tfevents'):
            events.setdefault(p,set()).update(ids);continue
        if p.suffix in TEXT or p.name in {'COMMANDS','SHA256SUMS','requirements-core.txt'}:
            paths.setdefault(p,set()).update(ids)
for e in EXP:
    run=ROOT/e['run_root']; collect(run,[e['id']])
    ws=[]
    if run.exists():
        for p in run.rglob('*'):
            if p.is_file() and p.suffix in {'.pth','.pt','.ckpt'} and selected(p.relative_to(ROOT)):
                # Keep selected best, last, final, and latest complete states. Improvement snapshots are not paper-selected.
                if any(k in p.name.lower() for k in ['best','last','final','latest','full_state_iter_30000']):ws.append(p)
    for p in ws:weightpaths.setdefault(p,set()).add(e['id'])
    if e.get('init_path'):
        p=ROOT/e['init_path']
        if p.exists():weightpaths.setdefault(p,set()).add(e['id'])
    for v in e['evidence']:collect(ROOT/v,[e['id']])

allids=[e['id'] for e in EXP]; acdcids=[e['id'] for e in EXP if e['dataset']=='ACDC']
shared=['configs/three_innovation_ablation_transfer_v22_0','configs/acdc10_third_innovation_codex_prep_top6_v21_3',
 'configs/paper_internal_evidence_v23_0','configs/internal_evidence_expansion_v24_0',
 'scripts/three_innovation_ablation_transfer_v22_0','scripts/paper_internal_evidence_v23_0','scripts/internal_evidence_expansion_v24_0',
 'audits/three_innovation_ablation_transfer_v22_0','audits/three_innovation_postfull_test_v22_1','audits/paper_internal_evidence_v23_0','audits/internal_evidence_expansion_v24_0',
 'runtime/three_innovation_ablation_transfer_v22_0/controller_formal_logs','runtime/paper_internal_evidence_v23_0','runtime/internal_evidence_expansion_v24_0',
 'final_report/three_innovation_postfull_test_v22_1','final_report/three_innovation_ablation_transfer_v22_0','final_report/paper_internal_evidence_v23_0',
 'final_report/internal_evidence_expansion_v24_0','final_report/acdc10_top6_all_test_v21_5',
 'revision_runs_20261006','tcsmix_control_revision_20261006','formal_E4_E5_20261008',
 'mechanism_strengthening_20261013','mechanism_direction_20261017','mechanism_direction_requested_basis_20261017',
 'paper_offline_followup_20261007','offline_A2_A5_20261007',
 'review_ABC_followup_20261009_231554_BJT','review_closure_evidence_20261009_192409_BJT',
 'PAPER_AUTHOR_CONFIRMED_FACTS.md','AGENTS.md','Table_2_3_4_Row_Checklist_20261009.txt',
 'requirements.txt','requirements_beta_fft_frozen.txt','environment_beta_fft.yml']
for v in shared:collect(ROOT/v,allids)
# Training code and all local Python/config dependencies; no data or raw binaries.
for v in ['train_ACDC','train_promise12','train_COVID19','common_evaluation','common_protocols/acdc','common_protocols/promise12']:
    collect(ROOT/v, acdcids if v=='train_ACDC' else allids)
for p in (ROOT/'tools').glob('*.py'):
    if re.search(r'v2[234]_|covid|promise|acdc',p.name,re.I):collect(p,allids)
for v in ['configs','docs','splits','tools','scripts','requirements-core.txt','README.md']:
    collect(PUB/v,allids)
collect(ROOT.parent/'beta_fft_publication_20261010/MIGRATION_MINIMAL.md',allids) if False else None

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def scrub(s):
    edits=[]
    pats=[('private_key',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
          ('github_token',r'\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
          ('api_key',r'\bsk-(?:proj-|live-)?[A-Za-z0-9_-]{32,}\b'),
          ('url_credentials',r'(?<=https://)[^\s/@:]+:[^\s/@]+@'),
          ('email',r'\b[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')]
    for kind,pat in pats:
        s,n=re.subn(pat,'[REDACTED_'+kind.upper()+']',s)
        if n:edits.append(dict(kind=kind,count=n))
    # Preserve numerical module-token settings; redact actual non-placeholder credential assignments only.
    pat=r'(?im)(\b(?:password|passwd|access_token|api_key|authorization)\s*[:=]\s*)([\x22\x27]?)([^\s,\x22\x27]{8,})'
    def assigned(m):
        if re.match(r'(?i)(none|null|false|true|os\.|\$|<|your_|getenv|bearer\b)',m[3]):return m[0]
        edits.append(dict(kind='credential_assignment',count=1));return m[1]+m[2]+'[REDACTED]'
    s=re.sub(pat,assigned,s)
    return s,edits
fileindex=[]; dedup={}
for n,(p,ids) in enumerate(sorted(paths.items(),key=lambda x:str(x[0]))):
    try:
        raw=p.read_bytes(); txt=raw.decode('utf-8')
    except (UnicodeDecodeError,OSError) as ex:
        exclusions.append(dict(path=str(p),reason=type(ex).__name__));continue
    if '\0' in txt:
        exclusions.append(dict(path=str(p),reason='binary content in nominal text file'));continue
    new,edits=scrub(txt)
    if p.is_relative_to(PUB):target=CAT/'records/publication'/p.relative_to(PUB); origin='publication/'+str(p.relative_to(PUB))
    else:target=CAT/'records/project'/p.relative_to(ROOT);origin=str(p.relative_to(ROOT))
    # Mixed historical tables are retained as historical evidence, not promoted to current results.
    target.parent.mkdir(parents=True,exist_ok=True); data=new.encode(); sha=hashlib.sha256(data).hexdigest()
    if not target.exists():
        if sha in dedup:os.link(dedup[sha],target)
        else:target.write_bytes(data);dedup[sha]=target
    if edits:changes.append(dict(path=origin,edits=edits))
    fileindex.append(dict(source=origin,source_sha256=hashlib.sha256(raw).hexdigest(),backup_path=str(target.relative_to(CAT)),
                          sha256=sha,size=len(data),experiment_ids=sorted(ids),kind='text'))
    if n%500==0:print('text',n,len(paths),flush=True)

# Originals are only read if old and unchanged through copy. No automatic retry of a changing checkpoint.
weights=[]
for n,(p,ids) in enumerate(sorted(weightpaths.items(),key=lambda x:str(x[0]))):
    before=p.stat(); age=time.time()-before.st_mtime
    if age<120:
        exclusions.append(dict(path=str(p.relative_to(ROOT)),reason='checkpoint mtime younger than 120 seconds'));continue
    sha=digest(p); target=PAY/'objects'/(sha+'.bin'); target.parent.mkdir(exist_ok=True)
    if not target.exists():
        tmp=target.with_suffix('.tmp');shutil.copyfile(p,tmp)
        after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or digest(tmp)!=sha:
            tmp.unlink();exclusions.append(dict(path=str(p.relative_to(ROOT)),reason='changed during copy'));continue
        tmp.replace(target)
    row=dict(source=str(p.relative_to(ROOT)),sha256=sha,size=before.st_size,object='objects/'+target.name,
             experiment_ids=sorted(ids),backup_time=NOW,source_mtime=datetime.fromtimestamp(before.st_mtime,ZoneInfo('Asia/Shanghai')).isoformat(),stable_copy=True)
    weights.append(row)
    print('weight',n+1,len(weightpaths),p.name,flush=True)

byid={e['id']:e for e in EXP}
for r in weights:
    for k in r['experiment_ids']:byid[k]['weights'].append(r['source'])
for r in fileindex:
    # Link direct run files and named config/receipts; shared materials have a complete many-to-many file index.
    for k in r['experiment_ids']:
        e=byid[k]
        if r['source'].startswith(e['run_root']+'/') or k.lower() in r['source'].lower():e['records'].append(r['backup_path'])
for e in EXP:
    matches=[w for w in weights if w['sha256']==e.get('expected_best_sha256')]
    e['paper_best_hash_status']='MATCH' if matches else ('NO_EXPECTED_SHA' if not e.get('expected_best_sha256') else 'MISSING_OR_MISMATCH')
    e['best_weight_objects']=[w['object'] for w in matches]
    e['training_logs']=[r for r in e['records'] if r.endswith(('.log','log.txt'))]
    e['configs']=[r for r in e['records'] if r.endswith(('.json','.yaml','.yml'))]
    e['source_snapshots']=[r for r in e['records'] if r.endswith('.py')]

def save(n,d): (CAT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
save('experiments.json',EXP);save('files.json',fileindex);save('weights.json',weights)
save('reports/redactions.json',changes);save('reports/exclusions.json',exclusions)
save('reports/tensorboard_sources.json',[dict(source=str(p.relative_to(ROOT)),experiment_ids=sorted(ids),size=p.stat().st_size,sha256=digest(p)) for p,ids in events.items()])
save('reports/collection.json',dict(backup_time=NOW,experiment_count=len(EXP),main_experiment_count=5,
    weight_references=len(weights),unique_weight_objects=len({w['sha256'] for w in weights}),
    unique_weight_bytes=sum((PAY/'objects'/(h+'.bin')).stat().st_size for h in {w['sha256'] for w in weights}),text_files=len(fileindex),
    original_project_is_git_repository=False,source_root=str(ROOT),publication_root=str(PUB),
    medical_arrays_uploaded=False,raw_tensorboard_uploaded=False,training_started=False,evaluation_started=False))
with (CAT/'experiments.csv').open('w') as f:
    fields=['id','priority','dataset','labeled_ratio','seed','modules','run_root','expected_best_sha256','paper_best_hash_status','validation','paper_metrics','reported_test','weights','training_logs','configs','source_snapshots','notes']
    writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader()
    for e in EXP:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in e.items() if k in fields})
print(json.dumps(json.loads((CAT/'reports/collection.json').read_text()),ensure_ascii=False,indent=2),flush=True)
