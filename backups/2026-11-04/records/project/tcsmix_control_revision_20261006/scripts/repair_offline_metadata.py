"""Correct duplicated single-sample support counts; probabilities/predictions unchanged."""
import shutil
from common import *

def main():
    path=WORK/'control_diagnostics.csv'
    raw=WORK/'reports/control_diagnostics_before_unique_pixel_counts.csv'
    assert not raw.exists()
    shutil.copyfile(path,raw);rows=read_table(path);changed=0
    for row in rows:
        if row['scope']=='single_sample_response':
            assert int(row['n_samples'])==1 and int(row['pixels'])==2*256*256
            for field in ['pixels','foreground_pixels','n_1','n_2','n_3']:
                assert int(row[field])%2==0;row[field]=str(int(row[field])//2)
            changed+=1
        assert int(row['pixels'])==int(row['n_samples'])*256*256
        assert sum(int(row[f'n_{c}']) for c in [1,2,3])==int(row['foreground_pixels'])
    write_csv(path,rows)
    cp=WORK/'reports/offline_checkpoint_inventory.json'
    shutil.copyfile(cp,WORK/'reports/offline_checkpoint_inventory_before_scope_clarification.json')
    identities=read_json(cp)
    for item in identities:
        if item['run']=='Historical_Full':item['scope']='existing original Full spatial-CTW post-step teacher'
        elif item['run']=='E1_uniform':item['scope']='existing E1 uniform-CTW post-step teacher, not Full'
    atomic_json(cp,identities)
    atomic_json(WORK/'reports/offline_metadata_correction.json',dict(time=now(),single_rows_corrected=changed,
        problem='single response reused signals(l,l,y,y), doubling reported supports while q/margins/background fraction were unchanged',
        fix='divide pixels, foreground_pixels and per-class counts by 2 for single_sample_response only; batch rows unchanged',
        probabilities_predictions_sizes_quality_changed=False,GT_context_changed=False,
        min_unique_nonzero_support_train=min(int(r[f'n_{c}']) for r in rows if r['mode']=='train_snapshot' for c in [1,2,3] if int(r[f'n_{c}'])>0),
        min_unique_nonzero_support_eval=min(int(r[f'n_{c}']) for r in rows if r['mode']=='eval' for c in [1,2,3] if int(r[f'n_{c}'])>0),
        original_csv=str(raw),original_sha256=sha(raw),corrected_csv_sha256=sha(path),model_forwards=0,optimizer_calls=0,
        inventory_scope_fix='Full explicitly spatial CTW, E1 explicitly uniform CTW; same checkpoint identities'))
    print(f'Corrected unique supports in {changed} single-sample rows; no model forward or update.')

if __name__=='__main__':main()
