#!/usr/bin/env python3
import csv, difflib, hashlib, json, os, shutil, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
TASK=Path(__file__).resolve().parents[1];CAT=TASK/'catalog';ROOT=Path('/home/renguanming/jupyterlab/rgm/beta-FFT-main');PUB=ROOT.parent/'beta_fft_publication_20261010/release'
files=json.loads((CAT/'files.json').read_text());weights=json.loads((CAT/'weights.json').read_text());experiments=json.loads((CAT/'experiments.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(CAT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def text(p,source,ids,section='project'):
    dest=CAT/'records'/section/source;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():return
    shutil.copyfile(p,dest)
    files.append(dict(source=('publication/' if section=='publication' else '')+source,backup_path=str(dest.relative_to(CAT)),
                      sha256=sha(dest),source_sha256=sha(p),size=dest.stat().st_size,experiment_ids=ids,kind='text',external_original_path=str(p)))
acdcids=[e['id'] for e in experiments if e['dataset']=='ACDC'];allids=[e['id'] for e in experiments]
for section,base in [('project',ROOT),('publication',PUB)]:
    rel='train_ACDC/beta_fft_acdc_repro.py.bak_clean_split_minimal';text(base/rel,rel,acdcids,section)
text(ROOT.parent/'beta_fft_publication_20261010/MIGRATION_MINIMAL.md','backup_provenance/MIGRATION_MINIMAL.md',allids)
for p in (ROOT.parent/'KnowSAM-main/ACDC').glob('*.list'):text(p,'backup_splits/acdc/'+p.name,acdcids)
for p in (ROOT/'data/COVID19_OFFICIAL20/manifests').glob('*'):
    if p.is_file() and p.suffix in {'.tsv','.json','.txt','.list'}:text(p,str(p.relative_to(ROOT)),['covid5','covid10'])
for p in (ROOT/'configs/covid19_official20').rglob('*'):
    if p.is_file() and p.suffix in {'.json','.yaml','.yml','.list'}:text(p,str(p.relative_to(ROOT)),['covid5','covid10'])
promise=ROOT/'data_staging/promise12/20260624_141558/data_promise12'
for p in promise.glob('*.list'):text(p,str(p.relative_to(ROOT)),['promise20'])
for key in ['promise20','covid5','covid10']:
    e=next(x for x in experiments if x['id']==key)
    p=ROOT/e['run_root']/'self_train/self_train_full_state_iter_25000.pth'
    st=p.stat();h=sha(p);dst=TASK/'payload/objects'/(h+'.bin')
    if not dst.exists():shutil.copyfile(p,dst)
    st2=p.stat()
    assert (st.st_size,st.st_mtime_ns)==(st2.st_size,st2.st_mtime_ns) and sha(dst)==h
    weights.append(dict(source=str(p.relative_to(ROOT)),sha256=h,size=st.st_size,object='objects/'+dst.name,experiment_ids=[key],
                        backup_time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_mtime=datetime.fromtimestamp(st.st_mtime,ZoneInfo('Asia/Shanghai')).isoformat(),stable_copy=True,
                        role='penultimate full-state checkpoint, enables resuming remaining original-budget updates'))
    e['weights'].append(str(p.relative_to(ROOT)))

baseline=subprocess.check_output(['git','-C',str(PUB),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(PUB),'status','--porcelain'],text=True)
(CAT/'reports/publication_uncommitted.patch').write_text(subprocess.check_output(['git','-C',str(PUB),'diff','HEAD'],text=True))
diffs=[];comparable=[]
for f in files:
    if f['source'].startswith('publication/'):continue
    p=ROOT/f['source'];q=PUB/f['source']
    if p.is_file() and q.is_file() and f['source'].startswith(('train_ACDC/','train_promise12/','train_COVID19/','tools/')):
        a=q.read_text(errors='replace').splitlines(keepends=True);b=p.read_text(errors='replace').splitlines(keepends=True)
        comparable.append(f['source'])
        if a!=b:diffs.extend(difflib.unified_diff(a,b,fromfile='publication/'+f['source'],tofile='original-current/'+f['source']))
(CAT/'reports/current_source_vs_publication.patch').write_text(''.join(diffs))
save('reports/git_provenance.json',dict(publication_baseline_commit=baseline,publication_worktree_status=status,
     original_training_directory_git_commit=None,uncommitted_diff_status='Original directory has no .git; historical uncommitted diff cannot be reconstructed. Per-run source snapshots retained.',
     comparison_patch_kind='Current original files versus publication baseline; not a historical training diff.',comparable_files=comparable))
for e in experiments:e['code_version']['publication_baseline_commit']=baseline
for f in files:
    direct=[e['id'] for e in experiments if f.get('source','').startswith(e['run_root']+'/')]
    f['association_basis']='direct run output' if direct else 'shared source, protocol, evidence or historical context; not proof every experiment consumed this file'
    if direct:f['experiment_ids']=direct
save('files.json',files);save('weights.json',weights);save('experiments.json',experiments)
collection=json.loads((CAT/'reports/collection.json').read_text());collection.update(weight_references=len(weights),unique_weight_objects=len({w['sha256'] for w in weights}),
    unique_weight_bytes=sum((TASK/'payload/objects'/(h+'.bin')).stat().st_size for h in {w['sha256'] for w in weights}),text_files=len(files))
save('reports/collection.json',collection)
print(json.dumps(collection,indent=2))
