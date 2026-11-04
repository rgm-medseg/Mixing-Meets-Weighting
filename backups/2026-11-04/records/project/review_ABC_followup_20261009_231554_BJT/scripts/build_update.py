#!/usr/bin/env python3
"""Read-only source audit; write only into this new evidence directory.

Standard library only. No models, arrays, distance metrics, checkpoints,
training scripts, or evaluator entry points are executed or loaded.
"""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import socket
import subprocess

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
PRIOR = ROOT / 'review_closure_evidence_20261009_192409_BJT'
E = OUT / 'evidence'
sources = []
source_cache = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csvwrite(name, rows, keys=None):
    rows = list(rows)
    if keys is None:
        keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def readcsv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def source(rel, label, ranges=None, copy=True, purpose=''):
    p = ROOT / rel
    assert p.is_file(), rel
    original_sha = sha(p)
    if ranges is None and copy:
        dest = E / label / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        lines = 'ALL'
        kind = 'BYTE_COPY'
    elif ranges:
        dest = E / label / (p.name + '.lines.txt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        all_lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        text = [f'ORIGINAL_PATH: {p}', f'ORIGINAL_SHA256: {original_sha}',
                'Line numbers below are original source line numbers; this is a text excerpt.']
        for a, b in ranges:
            text.append(f'--- ORIGINAL LINES {a}-{min(b,len(all_lines))} ---')
            text.extend(f'{i}: {all_lines[i-1]}' for i in range(a, min(b,len(all_lines))+1))
        dest.write_text('\n'.join(text)+'\n', encoding='utf-8')
        lines = ';'.join(f'{a}-{b}' for a,b in ranges)
        kind = 'LINE_NUMBERED_EXCERPT'
    else:
        dest = None
        lines = 'REFERENCE_ONLY'
        kind = 'HASHED_REFERENCE'
    rec = dict(source_id=f'S{len(sources)+1:03d}', original_path=str(p), original_sha256=original_sha,
               original_bytes=p.stat().st_size, original_lines=lines, export_kind=kind,
               exported_path=str(dest.relative_to(OUT)) if dest else '',
               exported_sha256=sha(dest) if dest else '', purpose=purpose)
    sources.append(rec)
    source_cache[rel] = rec
    return rec

def line(rel, needle):
    for i,s in enumerate((ROOT/rel).read_text(encoding='utf-8').splitlines(),1):
        if needle in s:
            return i
    raise ValueError((rel, needle))

def ref(rel, needle=None):
    s = source_cache[rel]
    n = line(rel,needle) if needle else s['original_lines']
    return f"{rel}:L{n}; SHA256={s['original_sha256']}"

for p in ['AGENTS.md','PAPER_AUTHOR_CONFIRMED_FACTS.md']:
    source(p, 'author_records', purpose='Author-adopted names and COVID values; no replacement recommendation')
check_rel = 'Table_2_3_4_Row_Checklist_20261009.txt'
check_src = source(check_rel, 'paper_roster', purpose='Actual checklist supplied by author; declared PDF SHA is not independently inspected here')
prior_rel = str(PRIOR.relative_to(ROOT)/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv')
source(prior_rel, 'prior_mapping', purpose='Carry previous mappings, precision and limitations; not a new 29-run recomputation')
prior_rows = {r['row_id']:r for r in readcsv(ROOT/prior_rel)}
source(str(PRIOR.relative_to(ROOT)/'SOURCE_VERSION_BINDING.csv'), 'prior_mapping', purpose='Reuse prior component SHA evidence; early family is not FINAL3')

# A: exact paper roster and per-field status. No matching by score.
roster = []
for n,s in enumerate((ROOT/check_rel).read_text().splitlines(),1):
    if s.startswith(('Table 2 |','Table 3 |','Table 4 |')):
        a = [x.strip() for x in s.split('|')]
        assert len(a)==12
        roster.append(dict(paper_position=f'P{len(roster)+1:02d}', table=a[0],dataset=a[1],budget=a[2],
                           paper_method=a[3],dice_display=a[4],iou_display=a[5],hd95_display=a[6],
                           asd_display=a[7],paper_distance_statement=a[8],checklist_state=a[9],
                           prior_row_id='' if a[10]=='NA' else a[10], protocol_note=a[11],
                           checklist_original_line=n,checklist_sha256=check_src['original_sha256'],
                           identity='REPRODUCED_AUTHOR_CONFIRMED'))
assert len(roster)==73
assert sum(bool(r['prior_row_id']) for r in roster)==20
assert {r['table']:sum(x['table']==r['table'] for x in roster) for r in roster} == {'Table 2':32,'Table 3':15,'Table 4':26}
assert {r['table']:sum(x['table']==r['table'] and not x['prior_row_id'] for x in roster) for r in roster} == {'Table 2':22,'Table 3':11,'Table 4':20}

old_rel = 'final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json'
new_base = 'external_comparators/2026_reproduction/paper_blocker_remediation_20260904/'
new_rel = new_base+'outputs/promise20_final3_canonical_reeval/EVALUATION_RECEIPT.json'
archive_rel = 'final_report/paper_evidence_for_mep_draft8_20260907T100152Z/sources/'+new_rel
complete_rel = new_base+'outputs/promise20_final3_canonical_reeval/TEST_COMPLETION_RECEIPT.json'
config_rel = new_base+'configs/PROMISE20_CANONICAL_REEVAL.json'
for rel,label in [(old_rel,'promise_old'),(new_rel,'promise_new'),(archive_rel,'promise_20260907_archive'),
                  (complete_rel,'promise_new'),(config_rel,'promise_config')]:
    source(rel,label,purpose='PROMISE ROW06 existing evaluator-version chain; no new evaluation')
source(new_base+'scripts/canonical_promise20_reeval.py','promise_driver',[(30,45),(68,105)],purpose='Historical wrapper delegates to canonical launcher; read only')
source('tools/v22_1_postfull_test_launcher.py','promise_metric',[(172,225)],purpose='PROMISE-only metric and reconstruction definitions; other datasets excluded')
source('external_comparators/2026_reproduction/manifests/generated/PROMISE_test.list','promise_split',purpose='Tiny existing test manifest for new receipt hash')
old = json.loads((ROOT/old_rel).read_text())
new = json.loads((ROOT/new_rel).read_text())
comp = json.loads((ROOT/complete_rel).read_text())
cfg = json.loads((ROOT/config_rel).read_text())
assert sha(ROOT/new_rel)=='42c06caaa2187841960ca1e26b8655b5c35598cdd093c5444e17fe87cfaa802d'
assert sha(ROOT/new_rel)==sha(ROOT/archive_rel)
assert old['CheckpointSHA']==new['checkpoint_sha256']==comp['CheckpointSHA']
assert old['TestManifestSHA']==new['test_manifest_sha256']==comp['TestManifestSHA']
assert sha(ROOT/'external_comparators/2026_reproduction/manifests/generated/PROMISE_test.list')==new['test_manifest_sha256']
assert sha(ROOT/config_rel)==new['config_sha256']
assert old['EvaluatorSHA']==new['historical_evaluator_sha256']
assert sha(ROOT/'tools/v22_1_postfull_test_launcher.py')==new['evaluator_sha256']
assert old['BestValIter']==comp['BestValIter']==cfg['best_val_iteration']==24400
assert old['BestValDice']==comp['BestValDice']==cfg['best_val_dice']==0.8912
metrics = []
for oldkey,key in [('Dice','dice'),('IoU','iou'),('HD95','hd95'),('ASD','asd')]:
    a,b = old[oldkey], new['metrics'][key]
    assert a==b==comp[oldkey]
    metrics.append(dict(metric=key,old_raw=repr(a),new_raw=repr(b),delta=repr(b-a),
                        display_percent_or_grid=f'{100*a:.2f}' if key in ('dice','iou') else f'{a:.3f}',
                        unit='fraction' if key in ('dice','iou') else 'grid',status='VERIFIED',
                        old_source=ref(old_rel,'"'+oldkey+'"'),new_source=ref(new_rel,'"'+key+'"')))
csvwrite('PROMISE_RECEIPT_COMPARISON.csv',metrics)
csvwrite('PROMISE_FULL_EVALUATOR_VERSION_CHAIN.csv',[
    dict(version='ORIGINAL_20260822',evaluator_path=old['EvaluatorPath'],evaluator_sha256=old['EvaluatorSHA'],
         checkpoint_sha256=old['CheckpointSHA'],test_manifest_sha256=old['TestManifestSHA'],
         validation_step=24400,validation_dice=0.8912,timestamp=old['TestTimestamp'],
         receipt=str(ROOT/old_rel),receipt_sha256=sha(ROOT/old_rel),status='VERIFIED',scope='Preserve original receipt SHA; no substitution'),
    dict(version='EXISTING_CANONICAL_REEVAL_20260904',evaluator_path=new['evaluator'],evaluator_sha256=new['evaluator_sha256'],
         checkpoint_sha256=new['checkpoint_sha256'],test_manifest_sha256=new['test_manifest_sha256'],
         validation_step=24400,validation_dice=0.8912,timestamp=new['completed_at'],
         receipt=str(ROOT/new_rel),receipt_sha256=sha(ROOT/new_rel),status='VERIFIED',
         scope='Same checkpoint/test manifest; four metrics identical; existing receipt, not evaluated this turn')])

field_rows=[]
for r in roster:
    p=prior_rows.get(r['prior_row_id'])
    if p:
        assert p['dataset']==r['dataset'] and p['label_budget']==r['budget']
        r.update(binding_status='VERIFIED',binding_scope='Existing row mapping carried; individual protocol gaps remain',
                 historical_run=p['historical_run'],prior_mapping_status=p['mapping_status'],
                 original_result=p['result_source'],original_result_sha256=p['source_sha256'],
                 checkpoint_sha256=p['checkpoint_sha256_recorded'],test_manifest_sha256=p['test_manifest_sha256_recorded'],
                 validation_selected_step=p['checkpoint_selected_step'],
                 evaluator_chain_status='VERIFIED' if r['prior_row_id']=='ROW06' else p['mapping_status'],
                 evaluator_chain_update=str(ROOT/new_rel) if r['prior_row_id']=='ROW06' else '')
    else:
        r.update(binding_status='NOT_FOUND',binding_scope='No row-specific binding located in scoped existing records; does not imply not reproduced',
                 historical_run='',prior_mapping_status='',original_result='',original_result_sha256='',
                 checkpoint_sha256='',test_manifest_sha256='',validation_selected_step='',evaluator_chain_status='NOT_FOUND',evaluator_chain_update='')
    fields = ['run','checkpoint_sha','test_split_sha','train_split_binding','val_split_binding','validation_selected_step',
              'validation_selection_rule','evaluator_version','actual_driver','actual_metric','raw_metrics',
              'distance_unit','spacing_axis_grid','aggregation','empty_nonfinite_rule','postprocessing']
    vals={} if not p else dict(run=p['historical_run'],checkpoint_sha=p['checkpoint_sha256_recorded'],
        test_split_sha=p['test_manifest_sha256_recorded'],train_split_binding=p['train_manifest'],val_split_binding=p['validation_manifest'],
        validation_selected_step=p['checkpoint_selected_step'],validation_selection_rule=p['checkpoint_selection'],
        evaluator_version=p['evaluator_sha256_recorded'],actual_driver=p['actual_evaluation_driver'],
        actual_metric=p.get('actual_metric_implementation') or p['evaluator_reference_path'],
        raw_metrics='; '.join(f'{k}={p[k+"_raw"]}' for k in ['dice','iou','hd95','asd']),
        distance_unit=p['distance_unit'],spacing_axis_grid=p['protocol'],aggregation=p['aggregation'],
        empty_nonfinite_rule='',postprocessing=p['postprocessing'])
    for f in fields:
        value=vals.get(f,'')
        status='NOT_FOUND' if not value else 'VERIFIED'
        notes='No new protocol assumption. Paper display is not raw precision.'
        evidence=ref(prior_rel,p['row_id']+',') if p else ref(check_rel,f"{r['table']} | {r['dataset']} | {r['budget']} | {r['paper_method']} |")
        if p and f in ['train_split_binding','val_split_binding']:
            status='PARTIAL' if value else 'NOT_FOUND'; notes='Path carried; full historical manifest SHA binding not supplied by this mapping column.'
        if p and f in ['evaluator_version','actual_driver','actual_metric','spacing_axis_grid','postprocessing']:
            status='PARTIAL' if p['mapping_status']=='PARTIAL' else ('VERIFIED' if value else 'NOT_FOUND')
            notes=p['notes'] or 'Carried prior code/receipt verification; not re-executed.'
        if p and f=='empty_nonfinite_rule':
            status='PARTIAL'; notes='Previously audited definitions are carried in prior report; this mapping CSV is not an independent empty-case receipt.'
            value='See previous evaluator-definition audit; no new per-row rule inferred.'
        if p and r['prior_row_id']=='ROW06' and f in ['evaluator_version','actual_driver','actual_metric','spacing_axis_grid','postprocessing']:
            status='VERIFIED'; evidence=ref(new_rel)
            if f=='evaluator_version': value=old['EvaluatorSHA']+' -> '+new['evaluator_sha256']
            if f=='actual_driver': value=new_base+'scripts/canonical_promise20_reeval.py -> tools/v22_1_postfull_test_launcher.py::promise_run'
            if f=='actual_metric': value='tools/v22_1_postfull_test_launcher.py::promise_binary_metrics'
            if f=='postprocessing':value='none'
            notes='Existing Sept4 receipt closes version difference; same checkpoint, split and all four metrics.'
        field_rows.append(dict(paper_position=r['paper_position'],table=r['table'],dataset=r['dataset'],budget=r['budget'],
                               method=r['paper_method'],prior_row_id=r['prior_row_id'],field=f,status=status,value=value,evidence=evidence,limitation=notes))
csvwrite('TABLE2_3_4_73_ROW_INDEX.csv',roster)
csvwrite('ROW_FIELD_STATUS.csv',field_rows)
csvwrite('MISSING_53_ROW_BINDINGS.csv',[r for r in roster if not r['prior_row_id']])

# COVID: retain result versions; never compute distances or infer spacing fixes.
macro_rel='revision_runs_20261006/covid_metrics_before_after_macro.csv'
source(macro_rel,'covid_spacing_existing',[(1,1),(6,6),(9,9),(16,16),(18,18)],purpose='Only four requested existing spacing-only rows; original source labels retained, no paper replacement')
source('revision_runs_20261006/reports/covid_spacing_geometry.csv','covid_geometry_prior',
       purpose='Carry already-verified four-patient XY spacing/axis metadata; no image or array included')
source('revision_runs_20261006/reports/covid_reevaluation_receipt.json','covid_geometry_prior',
       purpose='Existing Oct6 receipt covers its stated 48 case records; not a new receipt for every later macro row')
source(str(PRIOR.relative_to(ROOT)/'NONFINITE_RECORDS.csv'),'prior_nonfinite_scope',
       purpose='Earlier scoped nonfinite inventory; does not cover 53 missing rows or prove mask emptiness')
macro=readcsv(ROOT/macro_rel)
keys={'ROW07':'COVID19_5P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE','ROW09':'COVID19_10P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE',
      'ROW08':'COVID5_FINAL3','ROW10':'COVID10_FINAL3'}
covrows=[]
existing=[]
for r in roster:
    if r['dataset']!='COVID-19':continue
    rid=r['prior_row_id']; oldrow=prior_rows.get(rid)
    matches=[x for x in macro if x['run']==keys.get(rid) and x['policy']=='spacing_only_derived_center_grid']
    assert len(matches)<=1
    m=matches[0] if matches else None
    covrows.append(dict(paper_position=r['paper_position'],method=r['paper_method'],budget=r['budget'],prior_row_id=rid,
      raw_result_binding_status='VERIFIED' if oldrow else 'NOT_FOUND',
      existing_spacing_only_result_status='VERIFIED' if m else 'NOT_FOUND',
      source_run_label=m['run'] if m else (oldrow['historical_run'] if oldrow else ''),
      after_hd95_mm=m['after_hd95_mm'] if m else '',after_asd_mm=m['after_asd_mm'] if m else '',
      paper_adopted_hd95_mm=r['hd95_display'] if rid in ['ROW08','ROW10'] else '',
      source=ref(macro_rel,m['run']+',') if m else (oldrow['result_source'] if oldrow else ''),
      limitation='Separate existing spacing-only version; author-adopted Full 22.79/11.88 remain; withdrawn mandatory replacement stays withdrawn' if m else 'No existing correction receipt/result located. No metric guessed or re-evaluated.'))
    if m:
        assert m['before_dice']==m['after_dice'] and m['before_iou']==m['after_iou']
        existing.append({**m,'source_path':str(ROOT/macro_rel),'source_sha256':sha(ROOT/macro_rel),
                         'source_line':line(macro_rel,m['run']+','),'result_version':'EXISTING_SPACING_ONLY_NOT_PAPER_ADOPTED_FULL',
                         'paper_replacement_authorized':'false','native_grid_result':'false'})
assert len(covrows)==26 and len(existing)==4
csvwrite('COVID_MAIN_TABLE_CORRECTION_COVERAGE.csv',covrows)
csvwrite('COVID_EXISTING_SPACING_ONLY_FOUR_ROWS.csv',existing)

source('train_COVID19/covid19_volume_evaluator.py','covid_evaluator',[(14,108)],purpose='UCAD referenced actual metric definitions: stack axis2, both/one empty, nonfinite mean')
source('reproductions/UCAD_2026/scripts/eval_ucad_with_frozen_metrics.py','ucad_driver',[(21,26),(113,131)],purpose='Actual UCAD driver passes derived spacing unchanged; do not execute')
source('external_comparators/2026_reproduction/adapters/evaluate.py','external_driver',[(13,19),(58,77)],purpose='Executed adapter vs canonical reference field distinction')
source('external_comparators/2026_reproduction/adapters/common/metrics.py','external_metric',[(50,74),(99,122)],purpose='COVID pooled symmetric surface distances and actual aggregation')
source('external_comparators/2026_reproduction/adapters/common/data.py','external_data',[(194,218)],purpose='Derived slices stack Z first, unchanged spacing, later moveaxis before distances')
source('external_comparators/2026_reproduction/manifests/FINAL_SOURCE_FREEZE.sha256','external_freeze',
       [(line('external_comparators/2026_reproduction/manifests/FINAL_SOURCE_FREEZE.sha256',x),line('external_comparators/2026_reproduction/manifests/FINAL_SOURCE_FREEZE.sha256',x)) for x in ['adapters/evaluate.py','adapters/common/metrics.py','adapters/common/data.py']],
       purpose='Previously verified source freeze for actual adapter imports')
covexternal=[]
inclusion=[]
for budget in ['5','10']:
    ucad_rel=f'reproductions/UCAD_2026/results/COVID{budget}/seed1337/TEST_RESULT_FULL_PRECISION.json'
    source(ucad_rel,f'ucad_covid{budget}',purpose='Existing original UCAD receipt, not corrected result')
    u=json.loads((ROOT/ucad_rel).read_text())
    covexternal.append(dict(run=u['run_id'],budget=budget+'%',receipt=str(ROOT/ucad_rel),receipt_sha256=sha(ROOT/ucad_rel),
         actual_driver='reproductions/UCAD_2026/scripts/eval_ucad_with_frozen_metrics.py',metric='train_COVID19/covid19_volume_evaluator.py',
         evaluator_reference=u['evaluator']['source'],spacing='Derived NPZ spacing passed unchanged; no XY repair in driver',
         arrays='2D XY slices -> stack axis=2 -> XY,Z volume',aggregation='4-patient macro (ordinary mean)',
         hd95='95th percentile of pooled bidirectional surface distances',asd='Mean of same pooled bidirectional distances',
         empty_rule='both empty: Dice/IoU=1, distances=0; one empty: Dice/IoU=0, distances=[inf]; percentile can be nonfinite',
         nonfinite_rule='Ordinary mean; no nanmean or exclusion',postprocessing='none',existing_correction_status='NOT_FOUND',
         dice_raw=u['Dice'],iou_raw=u['IoU'],hd95_raw=u['HD95'],asd_raw=u['ASD'],
         distance_version='legacy_nominal_mm_on_derived_256_grid_UNSCALED_XY',protocol_status='PARTIAL',
         scope='Old receipt metric reference plus current driver static parsing; no new correction or run-time wrapper SHA claim'))
    for method in ['hien','pmeul']:
        job=f'{method}_covid19_{budget}p_seed1337'
        rel=f'external_comparators/2026_reproduction/results/{job}/EVALUATION_RECEIPT.json'
        source(rel,job,purpose='Excluded from current Table4; actual completed evaluation receipt, no spacing correction')
        d=json.loads((ROOT/rel).read_text())
        covexternal.append(dict(run=job,budget=budget+'%',receipt=str(ROOT/rel),receipt_sha256=sha(ROOT/rel),
             actual_driver='external_comparators/2026_reproduction/adapters/evaluate.py',
             metric='external_comparators/2026_reproduction/adapters/common/metrics.py',
             evaluator_reference=d['evaluator'],spacing='Derived NPZ/manifest spacing unchanged',
             arrays='Z,XY stacked -> moveaxis(0,-1) -> XY,Z before physical distances',aggregation='4-patient case macro (ordinary mean)',
             hd95='95th percentile of pooled bidirectional surface distances',asd='Mean of same pooled bidirectional distances',
             empty_rule='both empty: Dice/IoU=1, distances=0; one empty: Dice/IoU=0, distances=[inf]; percentile can be nonfinite',
             nonfinite_rule='Ordinary mean; no nanmean or exclusion',postprocessing='none',existing_correction_status='NOT_FOUND',
             dice_raw=d['metrics']['dice'],iou_raw=d['metrics']['iou'],hd95_raw=d['metrics']['hd95'],asd_raw=d['metrics']['asd'],
             distance_version='legacy_nominal_mm_on_derived_256_grid_UNSCALED_XY',protocol_status='VERIFIED',
             scope='Canonical evaluator in receipt is reference; executed adapter bound by previous source freeze. Not in 73-row numerator.'))
        inclusion.append(dict(run=job,method=d['method'],budget=budget+'%',current_table_inclusion='EXCLUDED',
             roster_status='VERIFIED',predeclared_inclusion_reason_status='NOT_FOUND',existing_correction_status='NOT_FOUND',
             author_scope_source=str(ROOT/check_rel),checklist_sha256=check_src['original_sha256'],
             original_evaluation_receipt=str(ROOT/rel),receipt_sha256=sha(ROOT/rel),
             closure_condition='Dated author-defined table scope/inclusion rule plus any existing correction receipt; do not infer rationale from scores'))
csvwrite('COVID_EXTERNAL_ACTUAL_PROTOCOL.csv',covexternal)
csvwrite('COVID_HIEN_PMEUL_INCLUSION_AUDIT.csv',inclusion)
for name in ['HIEN_PROTOCOL_AUDIT.md','PMEUL_PROTOCOL_AUDIT.md']:
    source('external_comparators/2026_reproduction/audits/'+name,'external_adaptation',[(82,91)],
           purpose='Existing dataset adaptation definitions; not evidence of current Table4 inclusion rationale')

# B: reuse known metadata; never load weights or rerun confirmed SSL budget.
meta_rel=str(PRIOR.relative_to(ROOT)/'CHECKPOINT_CPU_METADATA.json')
meta=json.loads((ROOT/meta_rel).read_text())
source(meta_rel,'prior_pretrain_metadata',copy=False,purpose='Previously CPU-parsed metadata only; no weight read this turn')
pretrain=[x for x in meta if '/Pre_model/label' in x['path']]
assert len(pretrain)==2
(E/'pretrain').mkdir(exist_ok=True)
(E/'pretrain'/'PRETRAIN_METADATA_PRIOR_EXTRACT.json').write_text(json.dumps(pretrain,ensure_ascii=False,indent=2)+'\n')
source('audit/beta_fft_clean_pretrained_label7_audit.md','pretrain_history',[(1,11)],purpose='June18 metadata absence; historical clean-split decision is not a new launch decision')
source('audit/beta_fft_acdc_official_supervised_pretrain_recovery.md','pretrain_history',[(1,23)],purpose='Definition/default and recovered entry, not executed generation command')
reuse_rel='audit/handoff_packages/beta_fft_acdc10_audit_handoff_20260624_125849/train_logs/beta_fft_acdc_official_full_seed42/train.log'
source(reuse_rel,'pretrain_reuse_only',[(1,4)],purpose='SSL loads label7; namespace pre_iterations=10000 is not generation proof')
source('audit/v14_1_acdc_5pct_seed_provenance_audit.md','pretrain_reuse_only',[(16,23),(32,48)],
       purpose='label3 downstream reuse and downstream seed, not pretraining seed; unrelated later audit opinions excluded')
b=[]
for m in pretrain:
    label=Path(m['path']).parent.name
    for f in ['executed_generation_command','generation_log','pretraining_seed','frozen_label_manifest',
              'actual_pretraining_updates','selected_pretraining_step','selected_pretraining_val_score']:
        b.append(dict(pretrain=label,checkpoint_path=m['path'],checkpoint_sha256_prior=m['sha256'],field=f,status='NOT_FOUND',value='',
           metadata_source=str(ROOT/meta_rel),metadata_source_sha256=sha(ROOT/meta_rel),
           known_metadata='keys=net,opt; SGD metadata has no explicit step counter',
           closure_condition='Contemporaneous generation command/log/frozen labeled cases/validation record bound to this exact checkpoint SHA; defaults and later config are insufficient'))
csvwrite('B_PRETRAIN_GENERATION_FIELDS.csv',b)
budget_rel=str(PRIOR.relative_to(ROOT)/'ACDC_TRAINING_BUDGET_RECOMPUTATION.json')
source(budget_rel,'prior_ssl_budget',purpose='Previously closed 30000 updates per student; carried without recalculation')
csvwrite('B_BUDGET_BOUNDARY.csv',[
 dict(setting='ACDC5%',ssl_per_student_updates=30000,ssl_status='VERIFIED',pretrain_updates='',pretrain_status='NOT_FOUND',
      total_pretrain_plus_ssl_updates='',total_status='PARTIAL',lr=0.01,lr_status='VERIFIED',source=budget_rel,
      note='Unknown pretrain count prevents total budget; confirmed SSL 30000 is not pending.'),
 dict(setting='ACDC10%',ssl_per_student_updates=30000,ssl_status='VERIFIED',pretrain_updates='',pretrain_status='NOT_FOUND',
      total_pretrain_plus_ssl_updates='',total_status='PARTIAL',lr=0.01,lr_status='VERIFIED',source=budget_rel,
      note='Only pretrain generation missing; initialization/SGD inheritance and SSL budget are carried as closed.')])

# C: import declarations and entry snapshots do not prove imported bytes at run time.
full_rel='runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/effective_source.py'
source(full_rel,'main_full_entry',[(18,27),(70,75),(167,178),(273,277)],purpose='Import/use and entry-only copy scope; archived entry already verified')
coventry='experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/train_promise12_bcp_fft.py'
source(coventry,'covid_final3_entry',[(24,31),(2311,2329),(2435,2443),(2456,2464)],purpose='FINAL3 import/use and lineage keys; no historical adapter/FFT SHA in these fields')
latermap='mep_promise_covid_training_supplement_20260907T104328Z/RUN_SOURCE_MAP.tsv'
source(latermap,'later_dependency_boundary',[(1,1),(3,4)],purpose='Sept7 supplied current dependency copies; explicitly no separate historical import snapshot')
sv=readcsv(PRIOR/'SOURCE_VERSION_BINDING.csv')
components=[('WeakStrongAugment','主 Full','paired augmentation',full_rel,21),
            ('DiceLoss','主 Full','DiceLoss dependency',full_rel,24),
            ('COVID adapter','COVID5/10_FINAL3','COVID adapter',coventry,2438),
            ('FFT helper','COVID5/10_FINAL3','paired FFT helper',coventry,29)]
c=[]
for component,run,role,entry,ln in components:
    s=next(x for x in sv if x['run']==run and x['role']==role)
    targets=['X03 / Full_main'] if run=='主 Full' else ['COVID5_FINAL3','COVID10_FINAL3']
    for target in targets:
        c.append(dict(target_run=target,component=component,import_declaration_status='VERIFIED',
              import_reference=f'{entry}:L{ln}',entry_sha256=sha(ROOT/entry),
              current_or_later_dependency_path=s['file'],current_or_later_sha256=s['current_sha256'],
              snapshot_sha_evidence_status='VERIFIED',historical_run_import_binding_status='NOT_FOUND',
              historical_run_dependency_sha256='',historical_binding_source='',overall_status='PARTIAL',
              snapshot_sha_scope='Carried current/Sept7 or earlier-family byte identity; not the target-run historical binding',
              evidence=str(PRIOR/'SOURCE_VERSION_BINDING.csv'),evidence_sha256=sha(PRIOR/'SOURCE_VERSION_BINDING.csv'),
              limitation=s['limitation'],closure_condition='Target-run dated dependency copy or loaded __file__+SHA/commit linked to entry/run/checkpoint'))
csvwrite('C_HISTORICAL_IMPORT_BINDINGS.csv',c)

# A small historical display conflict is preserved, with no run inferred by score.
legacy='final_report/PAPER_COMPARISON_TABLES_DRAFT_V19_4.md'
source(legacy,'legacy_numeric_reference',copy=False,purpose='Only legacy DC-Net10 numeric conflict referenced; outdated provenance labels not propagated')
csvwrite('A_EXISTING_RECORD_CONFLICT.csv',[
 dict(table='Table 2',dataset='ACDC',budget='10%',method='DC-Net',field='HD95',current_checklist_display='2.28',
      legacy_display='1.28',current_source=ref(check_rel,'ACDC | 10% | DC-Net |'),legacy_source=ref(legacy,'| DC-Net |'),
      status='PARTIAL',decision='Use author-supplied current roster for indexing; no original run binding found, no historical overwrite'),
 dict(table='Table 2',dataset='ACDC',budget='10%',method='DC-Net',field='ASD',current_checklist_display='0.68',
      legacy_display='0.38',current_source=ref(check_rel,'ACDC | 10% | DC-Net |'),legacy_source=ref(legacy,'| DC-Net |'),
      status='PARTIAL',decision='Keep conflict; only original run/receipt can close, not matching rounded scores')])

# Reproducible bounded read-only search ledger for remaining actual run bindings.
ledger=[]
queries=[
 ('A53_actual_run_metadata',['runs','formal_test','configs','external_comparators/2026_reproduction/results','reproductions'],
  r'SAMT[-_ ]?PCL|HCCP[-_ ]?SAM2|UniMatch[ _-]?(V2|v2)|CorrMatch|AIFP|AD[-_ ]MT|UA[-_ ]MT|MC[-_ ]Net|DC[-_ ]Net|SS[-_ ]Net|"method"\s*:\s*"(MT|DTC|BCP)"',
  ['*.json','*.tsv','*.yaml','*.yml','*.md','*.csv']),
 ('A_external_COVID_existing_spacing_corrections',
  [f'reproductions/UCAD_2026/results/COVID{x}/seed1337' for x in [5,10]]+
  [f'external_comparators/2026_reproduction/results/{m}_covid19_{x}p_seed1337' for m in ['hien','pmeul'] for x in [5,10]],
  r'spacing_only|corrected_spacing|spacing.{0,35}correct|XY.{0,35}spacing',['*.json','*.tsv','*.csv','*.md']),
 ('C_run_specific_import_receipts',
  ['audits/acdc10_third_innovation_codex_prep_top6_v21_3','audits/three_innovation_ablation_transfer_v22_0',
   'runtime/acdc10_third_innovation_codex_prep_top6_v21_3','runtime/three_innovation_ablation_transfer_v22_0'],
  r'dataloaders/dataset\.py|utils/losses\.py|covid19_dataset_adapter\.py|utils/displacement\.py|loaded.__file__',
  ['*.json','*.tsv','*.csv','*.md','*.sha256'])]
for q,paths,pattern,globs in queries:
    paths=[p for p in paths if (ROOT/p).exists()]
    cmd=['rg','-l','-i',pattern]
    for g in globs:cmd += ['-g',g]
    cmd += paths
    run=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    assert run.returncode in [0,1],run.stderr
    matches=run.stdout.splitlines()
    (E/(q+'_matches.txt')).write_text('\n'.join(matches)+'\n',encoding='utf-8')
    ledger.append(dict(query_id=q,status='VERIFIED',scopes=';'.join(paths),pattern=pattern,globs=';'.join(globs),
           matching_files=len(matches),matching_file_list='evidence/'+q+'_matches.txt',
           conclusion=('Single hit is ACDC10_NEW_ARM_L3_L5_IDENTITY_AUDIT.tsv: other ablation arms, import paths only, no target X03/COVID dependency SHA; qualified bindings NOT_FOUND.' if q=='C_run_specific_import_receipts' else 'NOT_FOUND qualified target bindings in reviewed metadata; keyword hits are leads, not proof.'),
           limit='Project-scoped existing metadata only; absence is not a claim that reproduction never occurred.'))
for label in ['label3','label7']:
    p=ROOT/'train_ACDC/Pre_model'/label
    names=sorted(x.name for x in p.iterdir())
    ledger.append(dict(query_id='B_'+label+'_directory_inventory',status='VERIFIED',scopes=str(p),pattern='directory names only',globs='',
          matching_files=len(names),matching_file_list=';'.join(names),conclusion='No generation log/manifest in this checkpoint directory',limit='Weights not opened or copied'))
ledger += [
 dict(query_id='A53_extended_history_search',status='PARTIAL',scopes='final_report;reports;best_results;final_artifacts;configs;manifests;related_methods;external_sources;audit;audits',
      pattern='Target method names and referenced manifest/receipt chains',globs='small metadata/source documents',matching_files='',matching_file_list='',
      conclusion='Earlier bounded reads located legacy numeric tables and repositories, no 53 qualified row bindings',
      limit='No all-disk or unrelated-project search. Legacy display scores and repository presence are not a run receipt.'),
 dict(query_id='B_pretrain_generation_history',status='PARTIAL',scopes='audit;audit/handoff_packages;audits;configs;manifests;train_ACDC/Pre_model/label3;label7',
      pattern='Pretrain recovery/generation/label3/label7 references',globs='small text metadata',matching_files='',matching_file_list='',
      conclusion='Definition/default and downstream reuse found; generation command/log/seed/labels/updates/selection not found',
      limit='Do not substitute pre_iterations=10000, downstream seed1337, or training_started:false for execution evidence.')]
csvwrite('SEARCH_LEDGER.csv',ledger)

now=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
report=f'''# A/B/C 收尾证据增补

生成：{now}；实际主机：{socket.gethostname()}。题目：Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation。模块 TRC、TCSMix、CTW；ACDC、PROMISE12、COVID-19。

本轮仅解析既有文本、源码和 receipt，进行四个标量差值复算；没有加载权重、影像或预测数组，没有运行模型或评价器，没有启动/恢复训练、重选 checkpoint、增加 seed、改动原产物或干预进程。E3 保持取消。D、237 文件校验、29 运行/116 均值及已关闭预算/CTW/入口事项没有重复执行。

**本轮新关闭：A 中 ROW06 PROMISE Full 的评价器版本差异。A 整体 PARTIAL；B 生成记录与 C 历史 import 绑定仍未闭合。** 缺失字段逐个列在 CSV；NOT_FOUND 仅表示在本轮限定资料中未取得证据，不表示方法未复现。

| 项目 | 状态 | 本轮结论 | 关闭条件 |
| --- | --- | --- | --- |
| A：当前表 2–4 范围 | VERIFIED | 清单实读 73 个表位置；20 个有既有映射，53 个尚缺；不是 73 个不同训练 | 已关闭范围计数 |
| A：ROW06 评价版本链 | VERIFIED | 原 e338… 与 Sept4 的 2a20… 均保留；同 checkpoint、同测试 manifest、四指标差值均 0 | 该差异缺口关闭 |
| A：其余 53 行绑定 | NOT_FOUND | ACDC22、PROMISE11、COVID20，原精度、协议等留空 | 提供对应 run/receipt、checkpoint与 split SHA、验证选点、实际 driver/metric 和距离规则 |
| A：UCAD 与其他主表行已有 COVID 修正 | NOT_FOUND | UCAD 两条有旧 receipt，未找到既有修正；其余20条先缺行绑定；未新做评价 | 既有修正 receipt/结果及版本身份；本轮不授权新增重评价 |
| A：未入表的 HiEn/PMEUL COVID | PARTIAL | 四条真实评价完成，当前清单排除；既定纳入原因及已有修正均 NOT_FOUND | 既定范围/纳入决策原记录，及已有修正结果；不按成绩改范围 |
| B：label3/label7 实际生成 | NOT_FOUND | 两检查点共14生成字段为空；5%总预算未知 | 精确绑定 checkpoint SHA 的生成命令、日志、seed、冻结标签、更新数、选模步数/分数 |
| C：四项历史 import 绑定 | PARTIAL | 导入声明及现有/后期副本 SHA 已知；目标运行历史绑定6条均 NOT_FOUND | 目标运行当时的依赖副本或 loaded __file__+SHA/commit |

## A：范围、版本和实际评价链

清单原路径 `{check_rel}`，SHA256 `{check_src['original_sha256']}`。头部所列 PDF SHA 是清单作者声明；本轮未取得/重新解析该 PDF，不将其写成独立 PDF 校验。

`TABLE2_3_4_73_ROW_INDEX.csv` 保留清单显示值与行号；`ROW_FIELD_STATUS.csv` 分 run、checkpoint SHA、test/train/val split、验证选点、评价版本/实际 driver/metric、原指标、单位、spacing/轴/网格、聚合、空例/非有限值及后处理登记。20/73 指已有**行映射**，并不意味着20行所有历史协议字段均已闭合。53缺口为22+11+20；4条未入表 COVID HiEn/PMEUL、5条补实验均不计入73的分子。全部比较沿用作者确认“复现”，不创建 local/published 分类，不由取整分数倒推出 run。

ROW06 原 receipt：`{old_rel}`，SHA `{sha(ROOT/old_rel)}`；原 evaluator `{old['EvaluatorSHA']}`。Sept4 receipt：`{new_rel}`，SHA `{sha(ROOT/new_rel)}`；Sept7 归档副本字节一致。新 evaluator `{new['evaluator_sha256']}`；保留旧 SHA，不覆盖成新 SHA。版本链与数值复算见 `PROMISE_FULL_EVALUATOR_VERSION_CHAIN.csv`、`PROMISE_RECEIPT_COMPARISON.csv`，各 receipt 的确切引用行见 CSV 和 `SOURCE_INDEX.csv`。

两版本 checkpoint SHA `{new['checkpoint_sha256']}`，test manifest SHA `{new['test_manifest_sha256']}` 均一致；验证选点仍是24400步、Dice0.8912。原评价时间 `{old['TestTimestamp']}`；既有新评价完成 `{new['completed_at']}`。四原精度：Dice0.8386401640018327、IoU0.7237411265148253、HD95 2.9863295043764113 grid、ASD1.4106984188286347 grid；逐项新−旧＝0.0。receipt 的 training_started、optimizer_created、backward_called、checkpoint_reselection_after_test 均 false；completion 的 optimizer_step_called 亦 false。**这些是 Sept4 已有记录，本轮没有重新评价。**

实际新 driver 为 `canonical_promise20_reeval.py`，调用 `tools/v22_1_postfull_test_launcher.py::promise_run`。源码172–225行：二值前景、原 staged 网格病例重建，无物理 spacing、10病例普通均值；HD95用 MedPy binary.hd95，ASD用预测到GT单向 binary.asd；双空Dice/IoU为0、距离0，单空为网格对角线距离；无后处理。空例定义来自代码，不能据此声称历史患者出现了空例。

COVID 26个主表位置的修正覆盖见 `COVID_MAIN_TABLE_CORRECTION_COVERAGE.csv`。本轮依据已有 `{macro_rel}` 仅摘录4个 `spacing_only_derived_center_grid` 原记录（源行6、9、16、18，原标签保留），没有调用距离函数：β-FFT5=35.761468219124396，β-FFT10=23.022861570142222；原标签COVID5_FINAL3=33.56438112527303、COVID10_FINAL3=18.957128805392596 mm。它们是本轮用户指定登记的**另一个既有 spacing-only 结果版本**，固定预测/GT、只修正XY、Z不变；不是 native-grid 重评价，Dice/IoU原记录逐字一致。

**论文采用的主 Full HD95 仍为COVID5% 22.79 mm、10% 11.88 mm。将论文主 Full 改为33.56/18.96的建议仍标记“结果对应关系混淆，已按作者确认撤回”，不再列作必须修改项；上述另版数据不替换或重绑定论文主结果。** 原材料和全部旧版本保留。

UCAD 两条实际 wrapper `reproductions/UCAD_2026/scripts/eval_ucad_with_frozen_metrics.py:113–130`，直接把 derived NPZ spacing传给 `covid19_volume_evaluator.py`；未修正XY。HiEn/PMEUL 的实际 driver 是 `adapters/evaluate.py`，经 `adapters/common/metrics.py` 和 `adapters/common/data.py`；receipt中canonical evaluator是**参考字段**，不能说实际运行了该参考入口。源 SHA、出处/行号与六条旧结果见 `COVID_EXTERNAL_ACTUAL_PROTOCOL.csv`、`SOURCE_INDEX.csv`。

UCAD 用XY切片沿axis2堆叠；HiEn/PMEUL先构成Z,XY，评价前moveaxis(0,-1)转XY,Z，spacing对应三轴。二者旧流程都直接沿用 derived spacing，没有XY resize-spacing修正。COVID前景二值，HD95是拼接双向表面距离的95百分位；ASD是同一拼接距离的均值，四病例等权普通均值，无后处理。双空Dice/IoU=1、距离0；单空Dice/IoU=0、距离数组[inf]，percentile可能非有限；普通mean不剔除非有限值、不改成nanmean。沿用既有 `NONFINITE_RECORDS.csv` 的已审查范围0条非有限记录；它不覆盖53缺失行，也不能由有限距离反推所有掩码非空。仅解析这些定义，本轮不生成新体距离。UCAD及四条外部方法没有找到既有 spacing-only结果，保持空值而不比例推算。既有四患者几何表与Oct6 receipt已随包提供；receipt自身只声明48条病例记录，不强称覆盖后来所有macro行。

当前清单未纳入的四条 COVID HiEn/PMEUL，旧评价receipt已完成；既有协议规定冻结划分和医学适配，但没有找到作者对当前主表的既定纳入/排除原因。`COVID_HIEN_PMEUL_INCLUSION_AUDIT.csv`将“当前排除”VERIFIED、“预定纳入原因”NOT_FOUND分别记录，旧助手判断和分数不用于推出作者选表原因。

另保留一个具体待来源绑定差异：旧V19_4表的DC-Net ACDC10 HD95/ASD为1.28/0.38，当前清单为2.28/0.68。见 `A_EXISTING_RECORD_CONFLICT.csv` 的双方SHA与行号。本轮按当前清单索引，未改论文或旧表，也未由其中任一组分数认定运行。

## B：只缺预训练生成证据

label3 SHA `3486fd3aecf5f30e77e7e15bec0841f56c87db6246a6dc50cc4dc6f2ee2087e0`；label7 SHA `477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3`。沿用既有CPU解析元数据，两者仅net/opt，SGD无step；本轮未加载/复制权重。`B_PRETRAIN_GENERATION_FIELDS.csv` 的14个字段NOT_FOUND：实际生成命令/日志/seed/冻结标注清单/真实更新数/验证选点步数/分数。

June18恢复报告只证明README入口及源码默认10000步，且本地主入口预训练调用曾注释；June24日志只证明后续SSL加载label7及Namespace默认pre_iterations10000。这些不等于这两个检查点真的训练10000步，也不能转移后续SSL seed到预训练。label3后续seed审计同样不是生成证明。两个目录当前只有权重，无生成日志。原始引用行和SHA见索引及小型摘录。

ACDC5/10每学生30000次半监督更新、两学生/教师初始化、SGD继承、恒定LR0.01为**已确认事项**，本轮仅沿用。`B_BUDGET_BOUNDARY.csv` 将未知预训练和已知SSL分开；5%总预算仍空/未知，不用derived config补出总数，不再把30000半监督预算列待办。

## C：声明、入口副本与历史依赖绑定分开

`C_HISTORICAL_IMPORT_BINDINGS.csv`有6条目标运行×依赖记录，归属4个组件：主Full WeakStrongAugment、DiceLoss；COVID_FINAL3 adapter、FFT helper（5%和10%分别记录）。入口导入与调用VERIFIED，现有/Sept7副本或早期家族SHA可读；但本轮没有找到这些依赖针对目标历史运行的副本或loaded __file__+SHA/commit，历史绑定字段NOT_FOUND，整体PARTIAL。

主Full effective_source.py:21/24/74/173–174/276，证明导入、调用及仅保存入口自身；不证明所导入模块当时的字节。COVID归档入口:29/2438/2459，lineage2311–2329只保存训练核心、预训练权重和manifest身份，未覆盖adapter/FFT导入SHA。Sept7 `RUN_SOURCE_MAP.tsv:3–4`明确区分入口已归档与当前依赖副本，不等于FINAL3当时依赖存档。早期COVID家族身份SHA不可自动转移到FINAL3。

CTW启动前SHA链与COVID5/10共同归档实际入口已核验，仍是关闭项；不再报告“找不到COVID训练入口”。闭合C只需目标历史运行对应依赖副本或加载身份记录，不需要运行模型或重训。

## 可采用的论文/作者说明句子

- “论文主Full（论文R1）对应原始X03/Full_main，在ACDC10%取得Dice90.35%；89.98%的独立补实验命名为‘更换混合控制信号’，为E4、C1、R2固定92×92的配对参照。”
- “PROMISE12 Full的既有canonical复核使用相同验证选定检查点及测试清单，Dice、IoU、HD95和ASD与原receipt完全一致；两版评价器SHA分别保留。”
- “ACDC5%和10%半监督阶段每个学生均完成30000次更新，学习率为0.01；现存记录不足以确定label3/label7预训练的实际更新数及5%预训练加半监督总预算。”
- “COVID论文主Full的HD95沿用作者确认的5%22.79mm和10%11.88mm；其他既有spacing-only版本分别登记，不代替论文采用结果。”
- 内部来源说明可写：“当前主表73个位置中20个已绑定既有运行资料，其余53个尚需补来源链；这不改变作者已确认的复现身份。” 此句适合交接/审查记录，未经作者决定不当作论文实验结论。

ACDC测试未参与候选方法选择，沿用作者更正；旧硬编码字段、助手总结和rank标签均不构成新的选择事件。本轮未发现新的原始选择事件记录，也未再次由这些字段推出选择因果。

## 交付与未闭合项

下一步只需要现有历史材料：53行对应receipt/冻结协议；label3/label7生成日志与选模链；四项目标import版本；既有外部COVID修正和预先表格纳入规则（如存在）。未找到的修正保持NOT_FOUND，本轮不授权体距离重评价。具体状态与关闭条件逐字段见CSV；所有来源的完整SHA与引用行见 `SOURCE_INDEX.csv`。小包不含影像、预测数组、权重、完整训练日志、凭据或无关项目资料。
'''
(OUT/'REVIEW_CLOSURE_EVIDENCE.md').write_text(report,encoding='utf-8')
csvwrite('SOURCE_INDEX.csv',sources)
validation=dict(generated_at=now,host=socket.gethostname(),paper_positions=73,carried_mapped_positions=20,
    missing_positions=53,table_counts={'Table2':32,'Table3':15,'Table4':26},missing_counts={'ACDC':22,'PROMISE12':11,'COVID-19':20},
    excluded_covid_external_runs=4,excluded_supplementary_runs=5,promise_four_metrics_exact_equal=True,
    promise_checkpoint_split_exact_equal=True,promise_receipt_expected_sha_match=True,promise_archive_byte_identical=True,
    covid_existing_spacing_only_records=4,covid_main_table_correction_NOT_FOUND=22,
    b_generation_fields_NOT_FOUND=14,c_import_target_rows=6,c_historical_bindings_NOT_FOUND=6,
    models_loaded=False,weights_read=False,prediction_arrays_read=False,distance_reevaluation=False,
    training_commands_executed=False,process_control_commands_executed=False,original_artifacts_modified=False,
    execution_scope='stdlib text parsing + four receipt scalar differences + project-scoped rg metadata search',
    checked_at_build='CSV cardinality, exact existing receipt SHA, same checkpoint/split, 4 exact scalar equalities; archive checks follow separately')
(OUT/'BUILD_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'directory':str(OUT),'source_count':len(sources),'query_matches':{x['query_id']:x['matching_files'] for x in ledger[:3]},'validation':validation},ensure_ascii=False))
