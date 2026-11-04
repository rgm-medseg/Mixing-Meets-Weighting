"""Final stable deliverable manifest and compact evidence archive (full arrays stay in workspace)."""
import json,tarfile
from common import *
def main():
    assert (W/'receipts/fixed_test_complete.json').exists()
    rows=[]
    for folder in ['scripts','configs','results','figures','receipts','skills']:
        for p in sorted((W/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.name not in ['pipeline_status.json','delivery_archive.json']:rows.append(dict(path=str(p.relative_to(W)),bytes=p.stat().st_size,sha256=sha(p)))
    for p in sorted(W.glob('*')):
        if p.is_file() and p.suffix in ['.md','.txt','.tex','.sh']:rows.append(dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    csvout(W/'OUTPUT_MANIFEST.csv',rows)
    # Include both fixed-case image/gradient archives and one full TRC feature/residual archive.
    selected=[W/'arrays/joint'/f'{mode}_b0_p0.npz' for mode in ['fixed_uniform','fixed_CTW','dynamic_uniform','dynamic_CTW']]
    selected.extend([W/'arrays/trc/patient002_frame01_z5.npz',W/'arrays/trc_features/patient002_frame01_z5.npz'])
    archive=W/'mechanism_strengthening_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as out:
        for r in rows:out.add(W/r['path'],arcname=r['path'])
        out.add(W/'OUTPUT_MANIFEST.csv',arcname='OUTPUT_MANIFEST.csv')
        for p in selected:out.add(p,arcname=str(p.relative_to(W)))
    js(W/'receipts/delivery_archive.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),full_arrays='all patient/region/feature/logit-gradient arrays retained under arrays/ in the shared workspace',new_checkpoints='runs/F97_main_signal_20261013/checkpoints; retained in workspace, excluded from compact archive'))
    print(str(archive),flush=True)
if __name__=='__main__':main()
