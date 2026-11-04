#!/usr/bin/env python3
import gzip, hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path

TASK=Path(__file__).resolve().parents[1];CAT=TASK/'catalog';ASSETS=TASK/'assets';ASSETS.mkdir(exist_ok=True)
REPO='rgm-medseg/Mixing-Meets-Weighting';TAG='comixw-critical-backup-2026-11-04'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(n,v):(CAT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
rawfiles=json.loads((CAT/'files.json').read_text());mapping={};compressed={};by_path={}
for f in rawfiles:
    key=f['backup_path']
    if key in by_path:
        previous=by_path[key]
        assert previous['sha256']==f['sha256']
        previous['experiment_ids']=sorted(set(previous['experiment_ids'])|set(f['experiment_ids']))
    else:by_path[key]=f
files=list(by_path.values())
for f in files:
    p=CAT/f['backup_path']
    if p.stat().st_size<1024**2 or p.suffix=='.py':continue
    target=p.with_name(p.name+'.gz');content_sha=f['sha256']
    if content_sha in compressed:
        os.link(compressed[content_sha],target)
    else:
        with p.open('rb') as inp,target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0,compresslevel=6) as out:
            shutil.copyfileobj(inp,out,8*1024**2)
        compressed[content_sha]=target
    mapping[f['backup_path']]=str(target.relative_to(CAT));p.unlink()
    f.update(content_sha256=content_sha,backup_path=str(target.relative_to(CAT)),compression='gzip',sha256=sha(target),stored_size=target.stat().st_size)
save('files.json',files)
experiments=json.loads((CAT/'experiments.json').read_text())
for e in experiments:
    for k in ['records','training_logs','configs','source_snapshots']:e[k]=[mapping.get(p,p) for p in e.get(k,[])]
save('experiments.json',experiments)
weights=json.loads((CAT/'weights.json').read_text());groups={};assignment={}
for w in weights:
    h=w['sha256']
    if h in assignment:continue
    ids=set(v for a in weights if a['sha256']==h for v in a['experiment_ids'])
    p0=next((k for k in ['acdc5','acdc10','promise20','covid5','covid10'] if k in ids),None)
    if p0:group='P0-'+p0
    elif any('ARM_' in k for k in ids):group='P1-acdc-factorial'
    elif any('WEIGHT_' in k for k in ids):group='P1-acdc-coefficients'
    elif any('SEED' in k for k in ids):group='P1-acdc-seeds'
    elif any(k.startswith('V21_3_') for k in ids):group='P1-acdc-module-replacements'
    else:group='P1-acdc-controls'
    groups.setdefault(group,[]).append(h);assignment[h]=group+'.tar.gz'
assets=[]
for group,hashes in sorted(groups.items()):
    p=ASSETS/(group+'.tar.gz')
    if p.exists():raise FileExistsError(p)
    print('PACK',group,len(hashes),flush=True)
    with p.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=3) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tf:
        for h in sorted(hashes):
            src=TASK/'payload/objects'/(h+'.bin');info=tf.gettarinfo(str(src),arcname='objects/'+src.name)
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
            with src.open('rb') as f:tf.addfile(info,f)
    if p.stat().st_size>=1900000000:raise RuntimeError('Bundle requires splitting before upload: '+str(p))
    assets.append(dict(name=p.name,kind='weights',size=p.stat().st_size,sha256=sha(p),object_count=len(hashes),
                       download_url=f'https://github.com/{REPO}/releases/download/{TAG}/{p.name}',upload_status='PENDING_AUTHENTICATION'))
for w in weights:w['asset']=assignment[w['sha256']]
save('weights.json',weights);save('assets.json',assets)
(CAT/'SHA256SUMS.assets').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets))
print('PACK COMPLETE',sum(a['size'] for a in assets),len(assets),flush=True)
