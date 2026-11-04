"""Engineering rehearsal with labeled TEMPORARY fixtures; no training/inference/R2 results."""
import shutil,tempfile
from common import *

def main():
    import analyze_R2 as analysis,package_R2 as package,summarize_R2 as summary
    formal=WORK
    with tempfile.TemporaryDirectory(prefix='R2_engineering_fixture_') as temporary:
        fixture=Path(temporary)
        for name in ['results','analysis','reports','state','configs','scripts','source','logs','runs','archives','handoff_sources']:(fixture/name).mkdir()
        for module in [analysis,package,summary]:module.WORK=fixture
        for p in (formal/'reports').glob('Historical_*'):shutil.copyfile(p,fixture/'reports'/p.name)
        cfg=read_json(formal/'configs/R2_fixed92.yaml');cfg['run_root']=str(fixture/'runs/fixture_NOT_FORMAL')
        atomic_json(fixture/'configs/R2_fixed92.yaml',cfg)
        atomic_json(fixture/'state/status.json',dict(status='ENGINEERING_FIXTURE_NOT_R2',stage='TEST',joint_step=30000))
        shutil.copyfile(formal/'R2_AREA_LOCK.json',fixture/'R2_AREA_LOCK.json')
        # Intentional fixture: use frozen R1 metric rows as stand-in to verify zero pair difference.
        shutil.copyfile(PARENT_R1/'results/case_class_metrics.csv',fixture/'results/case_class_metrics.csv')
        shutil.copyfile(PARENT_R1/'results/EVALUATION_RECEIPT.json',fixture/'results/EVALUATION_RECEIPT.json')
        rows=read_table(PARENT_R1/'results/R1_control_trace.csv')
        for row in rows:
            row.update(rectangle_h=92,rectangle_w=92,actual_area_pixels=8464,actual_area=8464/65536,s=.359375,s_actual=.359375,s_dynamic_B=.18+.2*float(row['q_B']),controller_version='v2_fixed')
        write_csv(fixture/'results/R2_control_trace.csv',rows)
        final=analysis.trajectory();assert len(final)==30000
        assert analysis.compare()['paired_rows']==480
        deltas=read_table(fixture/'analysis/paired_case_class_differences.csv')
        assert all(float(r[k+'_delta'])==0 for r in deltas if r['method']=='R2_fixed92' and r['reference']=='R1_v2_B' for k in ['dice','iou','hd95','asd'])
        areas=read_table(fixture/'analysis/scale_distribution.csv')
        assert all(r['n_sizes']=='1' and float(r['actual_area_mean'])==8464/65536 for r in areas if r['method']=='R2_fixed92')
        summary.main();assert (fixture/'FINAL_REPORT_R2_CN.md').exists()
        package.package(start=True)
        receipt=read_json(fixture/'startup_archive_receipt.json');assert receipt['status']=='VERIFIED'
        proof=dict(status='PASS',time=now(),training_updates=0,inference_calls=0,fixture_scope='temporary metadata only; R1 metrics used as explicitly declared placeholder to verify pairing; ALL fixtures deleted; not formal R2 results',tested=['30000-step fixed trajectory statistics and per-stage matching','four-method aggregate and 480 pair rows','R2-minus-R1 fixture exactly zero','summary generation with partial stage metadata','archive internal compressed readback and external SHA256'],fixture_files_retained=False)
    atomic_json(formal/'reports/ANALYSIS_ENGINEERING_REHEARSAL.json',proof)
    print('CPU metadata rehearsal PASS; no formal training/inference or fabricated result persisted.',flush=True)

if __name__=='__main__':main()
