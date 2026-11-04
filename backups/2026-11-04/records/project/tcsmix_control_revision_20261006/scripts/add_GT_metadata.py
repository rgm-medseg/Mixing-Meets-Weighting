"""Append label-presence context to existing offline rows; no model forward."""
import hashlib
import shutil
import numpy as np
from common import *
from offline_diagnostics import samples

def main():
    dst=WORK/'reports/control_diagnostics_before_GT_metadata.csv';assert not dst.exists()
    path=WORK/'control_diagnostics.csv';shutil.copyfile(path,dst)
    before=sha(WORK/'analysis/diagnostic_samples.csv');items=samples();assert before==sha(WORK/'analysis/diagnostic_samples.csv')
    lookup={v['id']:v for v in items};summary=[]
    for v in items:
        if v['gt'] is None:continue
        gt=v['gt'];summary.append(dict(sample=v['id'],cohort=v['cohort'],patient=v['patient'],GT_sha256=hashlib.sha256(gt.tobytes()).hexdigest(),
            pixels=gt.size,foreground_pixels=int((gt>0).sum()),**{f'n_{c}':int((gt==c).sum()) for c in [1,2,3]}))
    write_csv(WORK/'analysis/diagnostic_GT_summary.csv',summary)
    rows=read_table(path)
    for row in rows:
        batch=[lookup[k] for k in row['samples'].split(';')]
        if any(v['gt'] is None for v in batch):
            row.update(GT_foreground_pixels='NA: unlabeled GT not used',GT_foreground_fraction='NA',GT_present_classes='NA')
        else:
            gt=np.stack([v['gt'] for v in batch]);row.update(GT_foreground_pixels=int((gt>0).sum()),GT_foreground_fraction=float((gt>0).mean()),
                GT_present_classes=';'.join(str(c) for c in [1,2,3] if (gt==c).any()),**{f'GT_n_{c}':int((gt==c).sum()) for c in [1,2,3]})
    write_csv(path,rows,list(dict.fromkeys(k for r in rows for k in r)))
    atomic_json(WORK/'reports/GT_metadata_append_receipt.json',dict(time=now(),model_forwards=0,optimizer_calls=0,
        reproduced_input_manifest_sha256=before,original_diagnostic_predictions_and_scores_changed=False,unlabeled_GT_loaded=False))

if __name__=='__main__':main()
