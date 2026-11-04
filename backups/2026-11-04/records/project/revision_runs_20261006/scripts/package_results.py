"""Archive review evidence only; never raw clinical arrays, credentials or weights."""
import os
import hashlib
import shutil
import tarfile
import tempfile
from common import *

def main():
    candidates=[]
    roots=['scripts','configs','reports','analysis','logs','state','source']
    extensions={'.py','.sh','.json','.yaml','.yml','.md','.csv','.tsv','.txt','.svg','.png','.log','.patch'}
    for name in roots:
        for p in (WORK/name).rglob('*'):
            if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts and not any(part.startswith('evidence_snapshot_') for part in p.parts) and p!=WORK/'reports/archive_receipt.json':
                candidates.append(p)
    for p in WORK.glob('*'):
        if p.is_file() and p.suffix in extensions and p.name!='SHA256SUMS.txt': candidates.append(p)
    # Ship pinned official method implementation, not dependency binaries or git auth config.
    for p in (WORK/'external/AdaMix').rglob('*'):
        if p.is_file() and (p.suffix in {'.py','.md'} or p.name.startswith(('LICENSE','COPYING')) or p.name=='requirements.txt') and '.git' not in p.parts and '__pycache__' not in p.parts: candidates.append(p)
    for p in (WORK/'runs').rglob('*'):
        if p.is_file() and p.suffix in {'.json','.csv','.py'} and 'smoke' not in p.parts and 'smoke_archive' not in p.parts: candidates.append(p)
    candidates=sorted(set(candidates))
    output=WORK/'revision_evidence_20261006.tar.gz'; tmp=output.with_name(output.name+'.tmp')
    # Controller heartbeat and training logs can change while packaging. Hash
    # the same frozen bytes that enter the archive, rather than live paths.
    with tempfile.TemporaryDirectory(prefix='evidence_snapshot_',dir=WORK/'state') as directory:
        snapshot=Path(directory)
        relative=[]
        for p in candidates:
            rel=p.relative_to(WORK); dest=snapshot/rel
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
            relative.append(rel)
        info=Path('PACKAGE_SNAPSHOT.json')
        atomic_json(snapshot/info,dict(time=now(),status='EVIDENCE_SNAPSHOT',queue_status=read_json(WORK/'state/queue.json').get('status') if (WORK/'state/queue.json').exists() else 'NOT_STARTED',raw_arrays=False,weights=False,hash_scope='SHA256SUMS hashes archive snapshot bytes; live heartbeat/logs continue changing; outer archive hash is external'))
        relative.append(info)
        listing='\n'.join(sha(snapshot/rel)+'  '+str(rel) for rel in sorted(relative))+'\n'
        (snapshot/'SHA256SUMS.txt').write_text(listing)
        (WORK/'SHA256SUMS.txt').write_text(listing)
        relative.append(Path('SHA256SUMS.txt'))
        with tarfile.open(tmp,'w:gz') as archive:
            for rel in relative: archive.add(snapshot/rel,arcname=str(Path(WORK.name)/rel),recursive=False)
    os.replace(tmp,output)
    with tarfile.open(output) as archive:
        names=archive.getnames()
        assert not any(n.endswith(('.ckpt','.pt','.pth','.h5','.nii','.nii.gz','.npz','.npy')) for n in names)
        assert not any('/.git/' in n or 'python_deps/' in n for n in names)
        sums=archive.extractfile(WORK.name+'/SHA256SUMS.txt').read().decode()
        for line in sums.splitlines():
            digest,rel=line.split('  ',1)
            actual=hashlib.sha256(archive.extractfile(WORK.name+'/'+rel).read()).hexdigest()
            assert digest==actual, 'Archive checksum mismatch: '+rel
    (WORK/'revision_evidence_20261006.tar.gz.sha256').write_text(sha(output)+'  '+output.name+'\n')
    atomic_json(WORK/'reports/archive_receipt.json',dict(status='DONE',archive=str(output),sha256=sha(output),members=len(relative),time=now(),raw_medical_arrays=False,weights=False,all_internal_sha256_verified=True))
    print(f'Archived {len(relative)} files: {output} ({output.stat().st_size} bytes)')

if __name__=='__main__': main()
