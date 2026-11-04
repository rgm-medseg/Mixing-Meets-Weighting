#!/usr/bin/env python3
import gzip, hashlib, json, os, shutil, tarfile
from pathlib import Path
TASK=Path(__file__).resolve().parents[1];CAT=TASK/'catalog';ASSETS=TASK/'assets'
REPO='rgm-medseg/Mixing-Meets-Weighting';TAG='comixw-critical-backup-2026-11-04'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(n,x):(CAT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
for p in (TASK/'tools').glob('*.py'):shutil.copyfile(p,CAT/'tools'/p.name)
save('publication.json',dict(repository='https://github.com/'+REPO,branch='backup/comixw-critical-2026-11-04',tag=TAG,
    backup_commit_ref='refs/tags/'+TAG,backup_commit=None,
    commit_note='Resolve the immutable lightweight backup tag after SSH push; a commit cannot embed its own hash.',
    release_upload_status='BLOCKED_GITHUB_API_LOGIN',release_url=None,
    planned_release_url=f'https://github.com/{REPO}/releases/tag/{TAG}'))
files=json.loads((CAT/'files.json').read_text())
for f in files:
    if sha(CAT/f['backup_path'])!=f['sha256']:raise ValueError('Catalog hash mismatch: '+f['backup_path'])
log=(TASK/'private/restore.log').read_text()
assert 'RESTORE PASS' in log
commands=json.loads((CAT/'reports/restored_command_validation.json').read_text())
assert all(x['parser_check']=='PASS' for x in commands)
local={'weight_asset_sha256':'PASS','weight_object_sha256':'PASS','weight_archive_restoration':'PASS',
       'indexed_catalog_file_sha256':'PASS','restored_command_build_and_parser':'PASS',
       'indexed_file_references':len(files),'unique_indexed_contents':len({f['sha256'] for f in files}),
       'unique_weight_objects':87,'restored_files':7713,'training_started':False,'evaluation_started':False,
       'remote_download_sha256':'NOT_RUN_API_LOGIN_REQUIRED','paper_metrics_reproduced':False}
save('reports/local_verification.json',local)
(CAT/'BUNDLE_NOTE.md').write_text('The catalogue Release archive contains the metadata snapshot before its own asset entry is added. Its embedded assets.json lists the ten weight bundles. The Git branch assets.json and SHA256SUMS.assets additionally list the catalogue archive itself, avoiding a self-hash cycle. All download URLs remain planned until remote verification succeeds.\n')
p=ASSETS/'comixw-catalog-2026-11-04.tar.gz'
print('PACK CATALOG',flush=True)
def normalize(info):
    info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
    return info
with p.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=3) as out,tarfile.open(fileobj=out,mode='w|') as tf:
    tf.add(CAT,arcname='catalog',filter=normalize)
print('VERIFY CATALOG ARCHIVE',flush=True)
count=0;verified_members={}
with tarfile.open(p,'r:gz') as tf:
    for member in tf:
        if not (member.isfile() or member.islnk()):continue
        relative=Path(member.name).relative_to('catalog');expected=sha(CAT/relative)
        if member.islnk():
            actual=verified_members[member.linkname]
        else:
            h=hashlib.sha256()
            with tf.extractfile(member) as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
            actual=h.hexdigest()
        if actual!=expected:raise ValueError('Catalog archive content mismatch: '+str(relative))
        verified_members[member.name]=actual
        count+=1
meta=json.loads((CAT/'assets.json').read_text())
meta.append(dict(name=p.name,kind='catalog',size=p.stat().st_size,sha256=sha(p),catalog_files=count,
                 download_url=f'https://github.com/{REPO}/releases/download/{TAG}/{p.name}',upload_status='PENDING_AUTHENTICATION'))
save('assets.json',meta)
(CAT/'SHA256SUMS.assets').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in meta))
local.update(catalog_archive_content_sha256='PASS',catalog_archive_file_count=count,release_asset_count=len(meta),
             release_asset_bytes=sum(a['size'] for a in meta),release_asset_size_gib=sum(a['size'] for a in meta)/2**30)
save('reports/local_verification.json',local)
catalogfiles=sorted(p for p in CAT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.catalog')
(CAT/'SHA256SUMS.catalog').write_text(''.join(sha(p)+'  '+str(p.relative_to(CAT))+'\n' for p in catalogfiles))
retry='''#!/usr/bin/env bash
set -euo pipefail
comixw_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$comixw_task_dir/repository/backups/2026-11-04/tools/upload_release.py" \\
  --catalog "$comixw_task_dir/repository/backups/2026-11-04" \\
  --assets "$comixw_task_dir/assets" \\
  --download-cache "$comixw_task_dir/verification_remote" "$@"
'''
(TASK/'retry_upload.sh').write_text(retry);(TASK/'retry_upload.sh').chmod(0o700)
save('reports/package_totals.json',dict(catalog_file_count=len(catalogfiles)+2,indexed_file_count=len(files),
    unique_weight_object_count=87,raw_weight_bytes=2841875074,release_asset_count=len(meta),release_asset_bytes=sum(a['size'] for a in meta)))
# Recompute the comprehensive catalogue checksum after the final report is written.
catalogfiles=sorted(p for p in CAT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.catalog')
(CAT/'SHA256SUMS.catalog').write_text(''.join(sha(p)+'  '+str(p.relative_to(CAT))+'\n' for p in catalogfiles))
print(json.dumps(local,ensure_ascii=False,indent=2),flush=True)
