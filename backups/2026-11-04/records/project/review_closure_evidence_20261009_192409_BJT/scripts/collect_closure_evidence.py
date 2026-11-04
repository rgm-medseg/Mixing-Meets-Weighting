"""CPU/text evidence audit. Writes exclusively inside this new handoff directory."""
import ast
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
INDEX = {}
HASH_CACHE = {}
METRICS = ('dice', 'iou', 'hd95', 'asd')

def path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def sha(p):
    p = path(p)
    if not p.is_file():
        return ''
    if str(p) not in HASH_CACHE:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1024 * 1024), b''):
                h.update(b)
        HASH_CACHE[str(p)] = h.hexdigest()
    return HASH_CACHE[str(p)]

def readjson(p):
    return json.loads(path(p).read_text())

def readcsv(p):
    p = path(p)
    with p.open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if p.suffix == '.tsv' else ','))

def writecsv(name, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

def writejson(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True)+'\n')

def register(p, expected='', copy=True):
    p = path(p)
    if p.name == 'run_v19_5_promise20_and_mscmr_final_r3.py':
        copy = False  # Export only the relevant PROMISE evaluator lines below.
    key = str(p)
    if key not in INDEX:
        exists = p.is_file()
        row = {'source_id': f'S{len(INDEX)+1:04d}', 'original_path': key,
               'exists': exists, 'sha256': sha(p) if exists else '',
               'bytes': p.stat().st_size if exists else '', 'packed_path': '',
               'packed_kind': 'INDEX_ONLY', 'expected_sha256': expected,
               'expected_sha_matches': sha(p) == expected if exists and expected else '',
               'original_lines': len(p.read_text(errors='replace').splitlines()) if exists and p.suffix in ['.py','.json','.csv','.tsv','.yaml','.list','.md','.txt'] else ''}
        INDEX[key] = row
        if exists and p.is_relative_to(OUT):
            row['packed_path'] = str(p.relative_to(OUT))
            row['packed_kind'] = 'NEW_CPU_AUDIT_ARTIFACT'
        elif copy and exists and p.suffix in ['.py','.json','.csv','.tsv','.yaml','.list','.md','.txt','.sha256'] and p.stat().st_size < 1200000:
            try:
                rel = p.relative_to(ROOT)
            except ValueError:
                # Only explicitly bound split manifests outside the project are eligible.
                if p.name not in ['train.list','train_slices.list','val.list','test.list']:
                    raise ValueError('outside-project export not allowed: '+str(p))
                rel = Path('bound_ACDC_manifests') / p.name
            dest = OUT / 'evidence' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            row['packed_path'] = str(dest.relative_to(OUT))
            row['packed_kind'] = 'BYTE_IDENTICAL_COPY'
    elif expected:
        INDEX[key]['expected_sha256'] = expected
        INDEX[key]['expected_sha_matches'] = sha(p) == expected
    return INDEX[key]['source_id']

def excerpts(p, numbers, name=None):
    p = path(p)
    sid = register(p, copy=False)
    lines = p.read_text(errors='replace').splitlines()
    numbers = sorted({n for n in numbers if 1 <= n <= len(lines)})
    dest = OUT / 'evidence' / 'line_excerpts' / (name or (sid+'_'+p.name+'.txt'))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('ORIGINAL: '+str(p)+'\nSHA256: '+sha(p)+'\n'+'\n'.join(f'{n}: {lines[n-1]}' for n in numbers)+'\n')
    INDEX[str(p)]['packed_path'] = str(dest.relative_to(OUT))
    INDEX[str(p)]['packed_kind'] = 'LINE_NUMBERED_EXCERPT'
    return sid

def spans(p, *needles):
    a=path(p).read_text(errors='replace').splitlines()
    return [i for i,s in enumerate(a,1) if any(n in s for n in needles)]

def metricdict(d):
    return {k: float(d.get(k, d.get(k.upper(), d.get({'dice':'Dice','iou':'IoU','hd95':'HD95','asd':'ASD'}[k], d.get(k+'_mm', float('nan')))))) for k in METRICS}

fullroot = 'runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT'
fiveroot = 'experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3'
post = 'final_report/three_innovation_postfull_test_v22_1/raw_test'
r2root = 'tcsmix_control_revision_20261006/formal_R2_fixed92_20261007'
runrows=[]
rawsets=[]

def add_run(method, dataset, budget, run, cp, cp_sha, ev, ev_sha, test, test_sha, source, metrics,
            selected='', config='', train='', val='', raw='', protocol='', notes='', raw_precision=True,
            receipt_kind='FORMAL_EVALUATION_RECEIPT', table_scope='TABLE2_4_CANDIDATE'):
    row={'row_id': f'ROW{len(runrows)+1:02d}', 'paper_method':method,'dataset':dataset,'label_budget':budget,
         'paper_identity':'REPRODUCED_AUTHOR_CONFIRMED','table_scope':table_scope,'historical_run':run,
         'checkpoint':str(path(cp)) if cp else '', 'checkpoint_sha256_recorded':cp_sha,
         'checkpoint_sha256_current':sha(cp) if cp else '', 'checkpoint_sha_match':sha(cp)==cp_sha if cp and cp_sha else '',
         'checkpoint_selected_step':selected,'checkpoint_selection':'VALIDATION_BEST',
         'split':'TEST','test_manifest':str(path(test)) if test else '', 'test_manifest_sha256_recorded':test_sha,
         'test_manifest_sha256_current':sha(test) if test else '', 'test_manifest_sha_match':sha(test)==test_sha if test and test_sha else '',
         'evaluator_reference_path':str(path(ev)) if ev else '', 'evaluator_sha256_recorded':ev_sha,
         'evaluator_sha256_current':sha(ev) if ev else '', 'evaluator_sha_match':sha(ev)==ev_sha if ev and ev_sha else '',
         'actual_evaluation_driver':str(path(ev)) if ev else '', 'protocol':protocol,'config_path':str(path(config)) if config else '',
         'train_manifest':str(path(train)) if train else '', 'validation_manifest':str(path(val)) if val else '',
         'result_source':str(path(source)),'source_sha256':sha(source),'receipt_kind':receipt_kind,
         'case_records':str(path(raw)) if raw else '', 'raw_precision_available':raw_precision,'notes':notes,
         'distance_unit':'grid' if dataset in ['ACDC','PROMISE12'] else 'legacy_nominal_mm_on_256_grid_UNSCALED_XY',
         'aggregation':'40_ED_ES_volumes_x_3_foreground_classes_macro' if dataset=='ACDC' else ('10_case_macro' if dataset=='PROMISE12' else '4_patient_volume_macro'),
         'ASD_definition':'directed_prediction_to_GT' if dataset in ['ACDC','PROMISE12'] else 'pooled_bidirectional_surface_distances_mean',
         'postprocessing':'none in located metric drivers; legacy PROMISE baseline NOT_VERIFIED' if dataset=='PROMISE12' else 'none in located metric drivers'}
    row['mapping_status']='VERIFIED' if all(row[k] is True for k in ['checkpoint_sha_match','test_manifest_sha_match','evaluator_sha_match']) else 'PARTIAL'
    normalized_metrics = metricdict(metrics)
    for k in METRICS:
        v = metrics.get(k, normalized_metrics[k])
        row[k+'_recorded_text']=str(v)
        row[k+'_raw']=repr(float(v)) if raw_precision else ''
        row[k+'_display']=f'{float(v)*100:.2f}' if k in ['dice','iou'] else f'{float(v):.3f}'
    runrows.append(row)
    for p,expected in [(cp,cp_sha),(ev,ev_sha),(test,test_sha),(source,''),(config,''),(train,''),(val,''),(raw,'')]:
        if p:register(p,expected,copy=Path(p).suffix not in ['.pth','.ckpt','.pt'])
    if raw and path(raw).exists():rawsets.append((row['row_id'],raw,metricdict(metrics)))
    return row

acdc_test='/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/test.list'
acdc_sha='98161bdfaad072bf5713d7345751ab5a2bf3e5468b8edb03fd9ebfedc4503671'
acdc_train='/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/train_slices.list'
acdc_val='/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/val.list'

# ACDC own rows: real receipts and the original table remain separate sources.
b5source='formal_test/acdc_v19_4/ACDC_5P_BASELINE_CONFIRMATORY_TEST/FORMAL_TEST_RECEIPT.json'
b5=readjson(b5source)
add_run('β-FFT','ACDC','5%',b5['run_id'],b5['checkpoint_path'],b5['checkpoint_sha256'],b5['evaluator'],b5['evaluator_sha256'],
        b5['test_list'],b5['test_list_sha256'],b5source,b5['aggregate'],selected=b5['checkpoint_global_iteration'],
        train=acdc_train,val=acdc_val,raw=b5['case_class_metrics_tsv'],protocol='ACDC_native_reconstruction_grid_directed_ASD')
f5source=post+'/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json';f5=readjson(f5source)
add_run('完整方法','ACDC','5%','ACDC5_FINAL3',f5['CheckpointPath'],f5['CheckpointSHA'],f5['EvaluatorPath'],f5['EvaluatorSHA'],
        f5['TestManifest'],f5['TestManifestSHA'],f5source,{k: f5[{'dice':'Dice','iou':'IoU','hd95':'HD95','asd':'ASD'}[k]] for k in METRICS},
        selected=f5['BestValIter'], config='configs/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3.json',
        train=acdc_train,val=acdc_val,raw=post+'/ACDC5_FINAL3/case_class_metrics.tsv',protocol='ACDC_native_reconstruction_grid_directed_ASD',
        notes='Derived config budget is inference; actual 30000 is independently proven by log/last checkpoint in section B.')
b10=next(r for r in readcsv('final_report/ACDC_5P_10P_TEST_RESULTS.tsv') if r['method_id']=='ACDC_10P_BASELINE')
b10receipt='tcsmix_control_revision_20261006/formal_R1_20261007/reports/Historical_baseline_EVALUATION_RECEIPT.json'
br=readjson(b10receipt)
add_run('β-FFT','ACDC','10%',b10['run_id'],b10['checkpoint_path'],b10['checkpoint_sha256'],
        'tools/v22_0_acdc_test_frozen_checkpoint.py',sha('tools/v22_0_acdc_test_frozen_checkpoint.py'),acdc_test,acdc_sha,
        'final_report/ACDC_5P_10P_TEST_RESULTS.tsv',{k:b10[k] for k in METRICS},selected=b10['best_iteration'],train=acdc_train,val=acdc_val,
        raw='tcsmix_control_revision_20261006/formal_R1_20261007/reports/Historical_baseline_case_class_metrics.csv',
        protocol='ACDC_native_reconstruction_grid_directed_ASD',receipt_kind='LEGACY_TABLE_PLUS_20261006_RECEIPT',
        notes='Later receipt confirms same checkpoint and exact metrics; original early evaluator binding remains partial.')
register(b10receipt)
f10source='final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json';f10=readjson(f10source)
add_run('完整方法 / 论文 R1','ACDC','10%','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',f10['CheckpointPath'],f10['CheckpointSHA'],
        f10['EvaluatorPath'],f10['EvaluatorSHA'],f10['TestManifest'],f10['TestManifestSHA'],f10source,f10['Metrics'],
        selected=f10['ValidationBestIteration'],config='configs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT.json',
        train=acdc_train,val=acdc_val,raw=f10['CaseClassMetrics'],protocol='ACDC_native_reconstruction_grid_directed_ASD',
        notes='90.35 main Full; TEST_SELECTION_EXPOSURE is a hard-coded evaluator field, not a candidate selection event (see conflict ledger).')

# PROMISE12 baseline: precision audit located, independently reproduced from saved masks.
pr=readcsv('final_report/PROMISE20_FINAL_TEST_RESULTS.tsv')[0]
promise_root='data_staging/promise12/20260624_141558/data_promise12'
precision_source='audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json'
pr_precision=readjson(precision_source)['baseline_test']
promise_split=readjson('audits/PROMISE20_TEST_SPLIT_AND_EXPOSURE_AUDIT.json')
add_run('β-FFT','PROMISE12','20%',str(Path(pr['checkpoint_path']).parent),pr['checkpoint_path'],pr['checkpoint_sha256'],
        'audits/run_v19_5_promise20_and_mscmr_final_r3.py', '',promise_root+'/test.list',promise_split['test_list_sha256'],precision_source,{k:pr_precision[k] for k in METRICS},
        selected=pr['checkpoint_iter'],train=promise_root+'/train.list',val=promise_root+'/val.list',
        raw=OUT/'PROMISE_BASELINE_SAVED_PRED_CPU_RECALC.csv',
        protocol='PROMISE12_224_staged_grid_legacy_metric_both_empty1_one_emptyinf',raw_precision=True,
        receipt_kind='HISTORICAL_PRECISION_AUDIT_PLUS_NEW_CPU_SAVED_MASK_RECALC',
        notes='Full-precision confirmatory audit found and reproduced exactly from ten saved predictions. Evaluator/dependency runtime SHA is not in the historical audit. Earlier training-time complete test row remains unavailable.')
register('audits/PROMISE20_TEST_SPLIT_AND_EXPOSURE_AUDIT.json')
shared_evaluator='audits/run_v19_5_promise20_and_mscmr_final_r3.py'
excerpts(shared_evaluator,list(range(22,74))+list(range(119,187))+list(range(246,268)))
pfsource=post+'/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json';pf=readjson(pfsource)
add_run('完整方法','PROMISE12','20%','PROMISE20_FINAL3',pf['CheckpointPath'],pf['CheckpointSHA'],pf['EvaluatorPath'],pf['EvaluatorSHA'],
        pf['TestManifest'],pf['TestManifestSHA'],pfsource,{k:pf[{'dice':'Dice','iou':'IoU','hd95':'HD95','asd':'ASD'}[k]] for k in METRICS},
        selected=pf['BestValIter'],config='configs/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3.json',
        train=promise_root+'/train.list',val=promise_root+'/val.list',raw=post+'/PROMISE20_FINAL3/case_metrics.tsv',
        protocol='PROMISE12_224_staged_grid_directed_ASD')
register('external_comparators/2026_reproduction/paper_blocker_remediation_20260904/outputs/promise20_final3_canonical_reeval/EVALUATION_RECEIPT.json')

covtest='data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv'
covval='data/COVID19_OFFICIAL20/manifests/VAL_ALL_SLICES.tsv'
for ratio in ['5','10']:
    source=f'formal_test/covid19_seed1337/COVID19_{ratio}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json'
    d=readjson(source)
    add_run('β-FFT','COVID-19',ratio+'%',d['source_training_run'],d['checkpoint_path'],d['checkpoint_sha256'],
            d['evaluator_path'],d['evaluator_sha256'],covtest,d['test_all_slices_sha256'],source,d['metric_result']['macro'],
            selected=d['best_iter'],config=f'configs/promise10_covid1337/COVID19_{ratio}P_BASELINE_CANON1337_R1.json',
            train=f'data/COVID19_OFFICIAL20/manifests/COVID19_{ratio}P_LABELED.tsv',val=covval,
            protocol='COVID_legacy_256_XYZ_with_unscaled_native_XY_spacing',notes='Receipt unit mm is nominal; spacing/grid mismatch documented separately.')
    source=post+f'/COVID{ratio}_FINAL3/TEST_RESULT_FULL_PRECISION.json';d=readjson(source)
    add_run('完整方法','COVID-19',ratio+'%',f'COVID{ratio}_FINAL3',d['CheckpointPath'],d['CheckpointSHA'],d['EvaluatorPath'],d['EvaluatorSHA'],
            d['TestManifest'],d['TestManifestSHA'],source,{k:d[{'dice':'Dice','iou':'IoU','hd95':'HD95','asd':'ASD'}[k]] for k in METRICS},
            selected=d['BestValIter'],config=f'configs/three_innovation_ablation_transfer_v22_0/COVID{ratio}_FINAL3.json',
            train=f'data/COVID19_OFFICIAL20/manifests/COVID19_{ratio}P_LABELED.tsv',val=covval,
            raw=post+f'/COVID{ratio}_FINAL3/patient_metrics.json',protocol='COVID_legacy_256_XYZ_with_unscaled_native_XY_spacing',
            notes='Archived entry exists, SHA e1006e...; dependency history audited in C; spacing-corrected results are separate.')

# All ten HiEn/PMEUL rows. Matrix PENDING_GATES is stale and is not completion evidence.
external='external_comparators/2026_reproduction'
for p in sorted((ROOT/external/'results').glob('*/EVALUATION_RECEIPT.json')):
    source=str(p.relative_to(ROOT));d=readjson(source);job=d['job_id'];config=f'{external}/configs/runs/{job}.json';c=readjson(config)
    row=add_run(d['method'],{'ACDC':'ACDC','PROMISE12':'PROMISE12','COVID19':'COVID-19'}[d['dataset']],c.get('label_ratio', c.get('setting','')) or ('20%' if 'promise12' in job else ('10%' if '10p' in job else '5%')),
            job,d['checkpoint'],d['checkpoint_sha256'],d['evaluator'],d['evaluator_sha256'],c['data']['test_manifest'],c['data']['test_manifest_sha256'],
            source,d['metrics'],config=config,train=c['data']['labeled_manifest'],val=c['data']['val_manifest'],
            raw=str(p.parent.relative_to(ROOT))+'/case_metrics.tsv',protocol='Frozen_common_metric_adapter_'+d['dataset'],
            notes='Canonical evaluator path in receipt is a reference; executed driver adapters/evaluate.py calls adapters/common/metrics.py. Frozen SHA manifest binds the helper source.')
    row['actual_evaluation_driver']=str(path(external+'/adapters/evaluate.py'))
    row['actual_metric_implementation']=str(path(external+'/adapters/common/metrics.py'))
    row['config_sha256_recorded']=d['config_sha256'];row['config_sha_match']=sha(config)==d['config_sha256']
    register(config,d['config_sha256'])
    trainreceipt=f'{external}/checkpoints/{job}/TRAIN_RECEIPT.json';register(trainreceipt)

# UCAD: four existing formal results, without adding a new classification.
for setting in ['ACDC5','ACDC10','COVID5','COVID10']:
    base=f'reproductions/UCAD_2026/results/{setting}/seed1337';source=base+'/TEST_RESULT_FULL_PRECISION.json';d=readjson(source);c=readjson(base+'/RESOLVED_CONFIG.json')
    add_run('UCAD','ACDC' if setting.startswith('ACDC') else 'COVID-19',d['setting'],d['run_id'],d['checkpoint'],sha(d['checkpoint']),
            d['evaluator']['source'],d['evaluator']['sha256'],d['test_manifest'],sha(d['test_manifest']),source,d,
            config=base+'/RESOLVED_CONFIG.json',train=c['labeled_manifest'],val=c['val_manifest'],
            raw=base+('/case_class_metrics.tsv' if setting.startswith('ACDC') else '/patient_metrics.tsv'),
            protocol='UCAD_frozen_'+d['evaluator']['name'],notes='Receipt omits checkpoint SHA and selected step; SHA freshly computed, not a historical receipt claim.')
    runrows[-1]['checkpoint_sha_origin']='CURRENT_FILE_HASH; historical receipt omits SHA'
    register(base+'/TRAIN_RECEIPT.json')

# Supplementary rows preserve paired controls. E5 is the whole-method comparison.
prev=readcsv('experiment_status_handoff_20261009_171518_BJT/RUN_INDEX.csv')
for r in prev:
    if r['historical_run_id'] not in ['R1_v2_B','R2_fixed92','C1_CTW_mean_uniform_v2B','E4_plain_residual_v2B','E5_AdaMix_CT_common_protocol']:continue
    directory=path(r['evaluation_source_path']).parent
    receipt=directory/'EVALUATION_RECEIPT.json';d=readjson(receipt)
    add_run(r['paper_name'],'ACDC','10%',r['historical_run_id'],d['checkpoint'],d['checkpoint_sha256'],
            'tools/v22_0_acdc_test_frozen_checkpoint.py',sha('tools/v22_0_acdc_test_frozen_checkpoint.py'),d['test_manifest'],d['test_manifest_sha256'],
            receipt,d['metrics'],selected=d['selected_step'],config=r['config_path'],train=acdc_train,val=acdc_val,
            raw=directory/'case_class_metrics.csv',protocol=d['protocol'],notes='Paired reference: '+r['paired_reference'],table_scope='SUPPLEMENTARY_MODULE_OR_CONTROL')
    # Formal evaluation actually uses the locked compatible metrics runner.
    if r['historical_run_id'].startswith('E'):
        driver=path('formal_E4_E5_20261008/scripts/evaluate.py')
    else:
        driver=directory.parent/'scripts'/({'R1_v2_B':'evaluate_R1.py','R2_fixed92':'evaluate_R2.py','C1_CTW_mean_uniform_v2B':'evaluate_C1.py'}[r['historical_run_id']])
    helper=driver.parent/'unified_evaluation.py'
    runrows[-1]['actual_evaluation_driver']=str(driver)
    runrows[-1]['actual_metric_implementation']=str(helper)
    register(driver);register(helper)

writecsv('TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv',runrows)
writecsv('METRICS_SOURCE_VALUES.csv',[{k:v for k,v in r.items() if k in ['row_id','paper_method','dataset','label_budget','historical_run','raw_precision_available','result_source','source_sha256','protocol'] or k.endswith(('_raw','_display','_recorded_text'))} for r in runrows])
print('Mapped rows',len(runrows),'including supplemental; source count',len(INDEX),flush=True)

# Actual protocols are preserved as structured per-run fields, never harmonized silently.
protocolrows=[]
for r in runrows:
    if r['config_path'] and path(r['config_path']).suffix=='.json':
        c=readjson(r['config_path'])
        protocolrows.append({'row_id':r['row_id'],'run':r['historical_run'],'config_sha256':sha(r['config_path']),
                            'actual_config_or_receipt':r['config_path'],'config_json':json.dumps(c,ensure_ascii=False,sort_keys=True),
                            'status':'PARTIAL' if 'derived' in c.get('schema','').lower() else 'VERIFIED_FILE_BINDING',
                            'note':'Derived statements remain intentions unless independently supported by log/checkpoint; full JSON retained.'})
for p in ['formal_E4_E5_20261008/tasks/E5/protocol.json','common_protocols/promise12/formal_protocol.yaml']:
    register(p)
writecsv('PER_RUN_PROTOCOLS.csv',protocolrows)

# Per-row CPU aggregate recomputation and explicit non-finite inventory.
aggregate_checks=[];nonfinite=[];zerooverlap=[]
def metric_rows(p):
    if Path(p).suffix=='.json':
        d=readjson(p);return [{'case': k, **metricdict(v)} for k,v in d.get('per_patient',{}).items()]
    return readcsv(p)

for rowid,p,expected in rawsets:
    a=metric_rows(p)
    if not a:continue
    for key in METRICS:
        vals=[float(r.get(key,r.get(key+'_mm',float('nan')))) for r in a]
        finite=all(math.isfinite(v) for v in vals)
        mean=statistics.fmean(vals)
        aggregate_checks.append({'row_id':rowid,'source':str(path(p)),'rows':len(a),'metric':key,'mean_recomputed':repr(mean),
                                 'receipt_mean':repr(expected[key]),'absolute_error':repr(abs(mean-expected[key])),
                                 'finite':finite,'within_1e_12':finite and abs(mean-expected[key])<1e-12})
        for i,v in enumerate(vals):
            if not math.isfinite(v):nonfinite.append({'row_id':rowid,'source':str(path(p)),'source_row_1based':i+2,'case':a[i].get('case',a[i].get('patient','')),'metric':key,'value':repr(v),'handling':'ordinary macro; no removal or imputation'})
    for i,r in enumerate(a):
        if float(r.get('dice',-1))==0:
            zerooverlap.append({'row_id':rowid,'source':str(path(p)),'source_row_1based':i+2,'case':r.get('case',r.get('case_id',r.get('patient',''))),
                                'class':r.get('class',r.get('class_name','')),'dice':r['dice'],'hd95':r.get('hd95',''),'asd':r.get('asd',''),
                                'empty_status':'UNRESOLVED_FROM_METRICS_ONLY; zero overlap is not proof of empty prediction'})
writecsv('AGGREGATE_RECOMPUTATION.csv',aggregate_checks)
writecsv('NONFINITE_RECORDS.csv',nonfinite,['row_id','source','source_row_1based','case','metric','value','handling'])
writecsv('ZERO_OVERLAP_CASES.csv',zerooverlap,['row_id','source','source_row_1based','case','class','dice','hd95','asd','empty_status'])

# D: compare all 120 class rows and all 20 patient rows, including per-patient identities.
r2raw=r2root+'/results/case_class_metrics.csv';r2patient=r2root+'/results/R2_patient_metrics.csv';a=readcsv(r2raw);b=readcsv(r2patient)
grouped=defaultdict(list)
for r in a:grouped[r['patient']].append(r)
patientchecks=[]
for r in b:
    entries=grouped[r['patient']]
    check={'patient':r['patient'],'class_rows':len(entries),'frames':len({q['case'] for q in entries}),'source':str(path(r2patient))}
    for k in METRICS:
        value=statistics.fmean(float(q[k]) for q in entries)
        check[k+'_recomputed']=repr(value);check[k+'_original']=r[k];check[k+'_error']=repr(abs(value-float(r[k])))
    patientchecks.append(check)
writecsv('R2_PATIENT_RECOMPUTATION.csv',patientchecks)
r2metrics=metricdict(readjson(r2root+'/results/EVALUATION_RECEIPT.json')['metrics'])
control=readjson('tcsmix_control_revision_20261006/formal_R1_20261007/results/EVALUATION_RECEIPT.json')['metrics']
writejson('R2_RECORD_VERIFICATION.json',{'class_rows':len(a),'unique_case_class_keys':len({(r['case'],r['class_id']) for r in a}),
    'frames':len({r['case'] for r in a}),'patients':len(grouped),'patient_table_rows':len(b),
    'class_ids':dict(Counter(r['class_id'] for r in a)),
    'each_patient_2_frames_6_class_rows':all(len(v)==6 and len({r['case'] for r in v})==2 for v in grouped.values()),
    'raw_macro':{k:statistics.fmean(float(r[k]) for r in a) for k in METRICS},
    'patient_macro':{k:statistics.fmean(float(r[k]) for r in b) for k in METRICS},
    'patient_table_max_error':max(float(r[k+'_error']) for r in patientchecks for k in METRICS),
    'expected':r2metrics,'all_finite':all(math.isfinite(float(r[k])) for r in a for k in METRICS),
    'paired_control_run':'R1_v2_B / 更换混合控制信号','delta_Dice_pp':100*(r2metrics['dice']-float(control['dice']))})
register(r2patient);register(r2root+'/results/R2_class_metrics.csv');register(r2root+'/results/R2_results.csv')

# B: actual training traces; the generation of label3/label7 remains a separate question.
traces=[]
for rel in [fullroot,fiveroot]:
    p=rel+'/audit/training_dynamics.csv';rows=readcsv(p);steps=[int(r['global_step']) for r in rows]
    register(p,copy=False);excerpts(p,[1,2,3,len(rows),len(rows)+1])
    source=rel+'/effective_source.py';register(source)
    log=rel+'/train.log';lines=path(log).read_text().splitlines()
    selected=spans(log,'pretrained_model=','resume_check_ok','iteration 30000','global_step=30000')
    excerpts(log,[1,2,3]+selected)
    duplicate_steps={str(k):v for k,v in Counter(steps).items() if v>1}
    sentinel=[r for r in rows if float(r['total_loss'])==0 and float(r['loss_1'])==0 and float(r['loss_2'])==0 and r['checkpoint_saved']=='1']
    traces.append({'run':rel,'trace_rows':len(rows),'unique_steps':len(set(steps)),'first_step':min(steps),'last_step':max(steps),
                   'unique_steps_exactly_1_through_30000':sorted(set(steps))==list(range(1,30001)),
                   'duplicate_step_counts':duplicate_steps,'terminal_sentinel_rows':len(sentinel),
                   'sentinel_source_line':'effective_source.py:549 appends checkpoint-only zero-loss terminal row; it is not another update',
                   'loss_finite_all':all(r['loss_finite']=='1' for r in rows),'grad_finite_all':all(r['grad_finite']=='1' for r in rows),
                   'student1_updates_supported':len(set(steps)),'student2_updates_supported':len(set(steps)),
                   'update_count_evidence':'One optimizer1.step and optimizer2.step per recorded loop; not independently instrumented optimizer call counters.',
                   'lr_assignment_in_entry':[f'{n}:{s.strip()}' for n,s in enumerate(path(source).read_text().splitlines(),1) if re.search(r'param_group.*\[.*lr|lr_\s*=',s)],
                   'source':str(path(source)),'source_sha256':sha(source),'trace_sha256':sha(p)})
writejson('ACDC_TRAINING_BUDGET_RECOMPUTATION.json',traces)

# Reconstruct current SSL prefix lists only; this does not reconstruct pretraining provenance.
register(acdc_train);register(acdc_val);register(acdc_test)
entries=[s.strip() for s in path(acdc_train).read_text().splitlines() if s.strip()]
subsets=[]
for ratio,n in [('5%',68),('10%',136)]:
    for i,s in enumerate(entries[:n]):subsets.append({'label_budget':ratio,'index_zero_based':i,'slice_id':s,
        'patient':s.split('_')[0],'manifest':acdc_train,'manifest_sha256_current':sha(acdc_train),
        'scope':'CURRENT_SSL_PREFIX_RECONSTRUCTION; historical pretrain labeled subset NOT_VERIFIED'})
writecsv('ACDC_SSL_LABEL_PREFIX_RECONSTRUCTION.csv',subsets)
writejson('ACDC_SSL_SPLIT_RECONSTRUCTION.json',{'train_slices':len(entries),'train_patients':len({s.split('_')[0] for s in entries}),
    'val_frames':len(path(acdc_val).read_text().splitlines()),'test_frames':len(path(acdc_test).read_text().splitlines()),
    'labeled_prefixes':{ratio:{'slices':len([r for r in subsets if r['label_budget']==ratio]),'patients':sorted({r['patient'] for r in subsets if r['label_budget']==ratio})} for ratio in ['5%','10%']},
    'limitation':'Existing SSL entry defines first68/136 slicing. No pretraining generation log or historical label-subset manifest was located.'})

# C: source layers and concrete historical hashes.
sourcebind=[]
def binding(run, role, p, hist_sha='', evidence='', strength='PARTIAL', comment=''):
    register(p,hist_sha)
    if evidence:register(evidence,copy=False)
    sourcebind.append({'run':run,'role':role,'file':str(path(p)),'current_sha256':sha(p),'historical_sha256':hist_sha,
                       'historical_match':sha(p)==hist_sha if hist_sha else '', 'historical_binding_evidence':str(path(evidence)) if evidence else '',
                       'status':strength,'limitation':comment})

binding('主 Full','generated entry',fullroot+'/effective_source.py',strength='VERIFIED',comment='Run-directory archived entry plus namespace, full trace and checkpoint identify execution; no complete imported-module content archive.')
binding('主 Full','TRC/TCSMix/optimizer helper','train_ACDC/v14_mask_adamix_components.py','75754fa75e6fd6fc01c97682262895998799c7ce3c0bc2dcbe058ef4c63dcaee',
        'audits/acdc10_third_innovation_codex_prep_top6_v21_3/ACDC10_OURS_CANONICAL_PROTOCOL_LOCK.json',strength='VERIFIED',comment='Contemporaneous canonical-source lock; does not certify all Python imports.')
binding('主 Full','CTW implementation','train_ACDC/v21_3_acdc10_third_innovation_components.py','f28abc48fedff330f9aa7470072ebc16a607c6c59fd14942c2adb111b8ce748a',
        'audits/acdc10_third_innovation_codex_prep_top6_v21_3/ROUND1_TOP6_CONTROLLER_DRY_CONSUMPTION.json',strength='VERIFIED',comment='Same SHA frozen before launch, source import in entry and actual X03 loss endpoint in training log.')
for role,p in [('paired augmentation','train_ACDC/dataloaders/dataset.py'),('DiceLoss dependency','train_ACDC/utils/losses.py')]:
    binding('主 Full',role,p,strength='PARTIAL',comment='Entry imports the module, current bytes match 20260907 evidence snapshot, but no prelaunch/run-time SHA receipt for this specific dependency located.')
    snapshot='final_report/paper_evidence_for_mep_draft8_20260907T100152Z/sources/'+p
    binding('主 Full',role+' later archive',snapshot,sha(p),strength='VERIFIED',comment='Byte equality with current snapshot, dated later than the historical training; equality alone does not prove runtime import identity.')
for ratio in ['5','10']:
    binding('COVID'+ratio+'_FINAL3','archived actual entry',f'experiments/three_innovation_ablation_transfer_v22_0/COVID{ratio}_FINAL3/self_train/train_promise12_bcp_fft.py',
            'e1006ebe9731d7665a886dc5f897f130c5818c132fa575cc16c9ea845cb27386',
            'audits/three_innovation_ablation_transfer_v22_0/V22_0_FORMAL_START_MANIFEST.json',strength='VERIFIED',comment='Actual common COVID5/10 archived training entry; not missing.')
for role,p,h in [('COVID adapter','train_COVID19/covid19_dataset_adapter.py','a6031696e9a63888c385cfe735bc010e73f5bad5cdf80e1cb6ed264e812d4ba1'),
                 ('paired FFT helper','train_promise12/utils/displacement.py','7e807dfc8944ed0d683110e4077bb3d355021b5f7b77ab08c9b989176836ff96')]:
    binding('COVID earlier seed42 family',role,p,h,'audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json',strength='VERIFIED',comment='Actual early family identity binds this dependency SHA; this family is not COVID5/10_FINAL3.')
    binding('COVID5/10_FINAL3',role,p,h,'audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json',strength='PARTIAL',comment='Matches earlier historical freeze/current source; FINAL3 entry imports it, but FINAL3-specific imported-module receipt/archive was not located. Do not transfer early-run binding silently.')

freeze=external+'/manifests/FINAL_SOURCE_FREEZE.sha256';register(freeze)
for s in path(freeze).read_text().splitlines():
    pieces=s.split(None,1)
    if len(pieces)==2 and pieces[1] in [external+'/adapters/common/data.py',external+'/adapters/common/metrics.py',external+'/adapters/evaluate.py']:
        binding('HiEn/PMEUL actual evaluator','frozen evaluation adapter',pieces[1],pieces[0],freeze,strength='VERIFIED')
writecsv('SOURCE_VERSION_BINDING.csv',sourcebind)

# Small required original sources, receipts, and line extracts.
small_sources=[
    'train_ACDC/beta_fft_label3.py','train_ACDC/beta_fft_label7.py','train_COVID19/covid19_official20_v2.py',
    'tools/v21_5_acdc10_test_frozen_checkpoint.py','tools/v22_1_postfull_test_launcher.py',
    'audit/beta_fft_clean_pretrained_label7_audit.md','audit/beta_fft_clean_pretrained_label7_evidence.csv',
    'audit/beta_fft_acdc_official_supervised_pretrain_recovery.md','audit/beta_fft_clean_supervised_pretrain_final_config.json',
    'audits/PROMISE10_COVID1337_CRITICAL_SOURCE_SHA256.tsv','audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json',
    'revision_runs_20261006/reports/covid_spacing_geometry.csv','revision_runs_20261006/covid_metrics_before_after_macro.csv',
    'revision_runs_20261006/covid_metrics_before_after.csv','revision_runs_20261006/COVID_SPACING_AUDIT.md',
    'revision_runs_20261006/reports/covid_reevaluation_receipt.json',
    external+'/manifests/RUN_MATRIX.tsv',external+'/manifests/DATASET_CONTRACT.json',
    'final_report/paper_internal_evidence_v23_0/tables/TABLE1_ACDC_5P_10P_FULL_PRECISION.tsv',
    'final_report/paper_internal_evidence_v23_0/tables/TABLE2_PROMISE20_FULL_PRECISION.tsv',
    'final_report/paper_internal_evidence_v23_0/tables/TABLE3_COVID_5P_10P_FULL_PRECISION.tsv',
    'final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv',
    'final_report/ACDC_5P_10P_TEST_RESULTS.tsv','final_report/PROMISE20_FINAL_TEST_RESULTS.tsv',
    'final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv',
    fullroot+'/full_completion_report.json',
    'audits/acdc10_third_innovation_codex_prep_top6_v21_3/candidate_V21_3_X03_CONTOUR_RELIABILITY_WEIGHT_FORMAL_LAUNCH_READINESS.json',
]
for p in small_sources:register(p)
for p,needles in [
    (fullroot+'/effective_source.py',['from dataloaders','from utils','innovation','WeakStrongAugment','load_parent_pretrain','transplant_parent_optimizer_state','optimizer1.step','optimizer2.step','update_model_ema','split="val"']),
    ('train_ACDC/dataloaders/dataset.py',['class WeakStrongAugment','random_rot_flip','random_rotate','ColorJitter','cutout_gray','mask[y:','image_strong']),
    ('train_ACDC/utils/losses.py',['class DiceLoss','smooth =','z_sum =','y_sum =','self.n_classes']),
    ('train_ACDC/v21_3_acdc10_third_innovation_components.py',['_norm01','_spatial_gradient','CONTOUR_RELIABILITY','boundary_weight','p2.detach']),
    ('train_COVID19/covid19_dataset_adapter.py',['image_batch.clone','np.stack(images','spacing =']),
    ('train_promise12/utils/displacement.py',['for i in range','img_with_labels =','img_without_labels =','swapped_img']),
    ('tools/v21_5_acdc10_test_frozen_checkpoint.py',['TEST_SELECTION_EXPOSURE','SelectionEndpoint']),
]:
    nums=set()
    for n in spans(p,*needles):nums.update(range(max(1,n-2),n+3))
    excerpts(p,nums)

# Preserve contradictory historical annotations and their generating code; do not derive selection causality.
conflicts=[{'identity':'X03 formal TEST evaluation receipt','time':'2026-08-19T12:55:05.155768+00:00','source':str(path(f10source)),
             'original_quote':'"TEST_SELECTION_EXPOSURE": true','writer':'tools/v21_5_acdc10_test_frozen_checkpoint.py:203 (hard-coded True)',
             'assessment':'Literal conflicts with author correction as a label; not a candidate-choice event. Validation-best selection independently recorded; do not infer selection causality.'},
            {'identity':'20260904 derived comparator remediation summary','time':'2026-09-04 (directory timestamp; JSON lacks own timestamp)',
             'source':str(path(external+'/paper_blocker_remediation_20260904/FINAL_COMPARISON.json')),
             'original_quote':'"reason": "CLASS_B material TEST contamination and evaluator mismatch"','writer':'derived assistant audit summary, not original method-selection trace',
             'assessment':'Retained legacy assertion; no new causal conclusion. Author correction governs manuscript naming and candidate-selection description.'}]
writecsv('HISTORICAL_ANNOTATION_CONFLICTS.csv',conflicts)
excerpts(external+'/paper_blocker_remediation_20260904/FINAL_COMPARISON.json',spans(external+'/paper_blocker_remediation_20260904/FINAL_COMPARISON.json','ACDC_10P'))

writecsv('SOURCE_INDEX.csv',list(INDEX.values()))
writejson('COLLECTION_SUMMARY.json',{'row_count':len(runrows),'table_candidate_rows':sum(r['table_scope']=='TABLE2_4_CANDIDATE' for r in runrows),
    'supplemental_rows':sum(r['table_scope']!='TABLE2_4_CANDIDATE' for r in runrows),'aggregate_tests':len(aggregate_checks),
    'aggregate_failures':[r for r in aggregate_checks if not r['within_1e_12']],
    'nonfinite_metric_count':len(nonfinite),'zero_overlap_class_rows':len(zerooverlap),'source_count':len(INDEX),
    'hash_mismatch_sources':[r for r in INDEX.values() if r['expected_sha_matches'] is False],
    'mapping_limitation':'Current manuscript Table2-4 row roster is not present in located server tables. The 24 located candidate rows are fully enumerated, not asserted as the final paper roster.',
    'training_started':False,'gpu_forward':False,'checkpoint_reselected':False,'original_files_written':False})
print('Done',json.dumps(readjson(OUT/'COLLECTION_SUMMARY.json'),ensure_ascii=False),flush=True)
