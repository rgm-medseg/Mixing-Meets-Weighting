#!/usr/bin/env python3
"""Verify only this newly generated text evidence package and create its archive."""
from pathlib import Path, PurePosixPath
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
import tarfile

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
def digest(b):return hashlib.sha256(b).hexdigest()
def read(name):
    with (OUT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
sources=read('SOURCE_INDEX.csv')
for s in sources:
    p=Path(s['original_path'])
    assert p.is_relative_to(ROOT)
    assert digest(p.read_bytes())==s['original_sha256'],p
    if s['exported_path']:
        e=OUT/s['exported_path']
        assert digest(e.read_bytes())==s['exported_sha256'],e
        if s['export_kind']=='BYTE_COPY':assert e.read_bytes()==p.read_bytes()
roster=read('TABLE2_3_4_73_ROW_INDEX.csv')
assert len(roster)==73
assert len({r['paper_position'] for r in roster})==73
assert Counter(r['table'] for r in roster)=={'Table 2':32,'Table 3':15,'Table 4':26}
missing=[r for r in roster if r['binding_status']=='NOT_FOUND']
assert len(missing)==53
assert Counter(r['dataset'] for r in missing)=={'ACDC':22,'PROMISE12':11,'COVID-19':20}
assert len(read('MISSING_53_ROW_BINDINGS.csv'))==53
assert all(r['identity']=='REPRODUCED_AUTHOR_CONFIRMED' for r in roster)
assert all(r['prior_row_id'] not in {'ROW13','ROW14','ROW18','ROW19','ROW25','ROW26','ROW27','ROW28','ROW29'} for r in roster)
fields=read('ROW_FIELD_STATUS.csv')
assert len(fields)==73*16
assert set(r['status'] for r in fields)<= {'VERIFIED','PARTIAL','NOT_FOUND'}
positions={r['paper_position'] for r in missing}
assert all(r['status']=='NOT_FOUND' and r['value']=='' for r in fields if r['paper_position'] in positions)
promise=read('PROMISE_RECEIPT_COMPARISON.csv')
assert len(promise)==4 and all(float(r['delta'])==0 for r in promise)
b=read('B_PRETRAIN_GENERATION_FIELDS.csv')
assert len(b)==14 and all(r['status']=='NOT_FOUND' and not r['value'] for r in b)
c=read('C_HISTORICAL_IMPORT_BINDINGS.csv')
assert len(c)==6 and len({r['component'] for r in c})==4
assert all(r['historical_run_import_binding_status']=='NOT_FOUND' and not r['historical_run_dependency_sha256'] for r in c)
cov=read('COVID_MAIN_TABLE_CORRECTION_COVERAGE.csv')
assert len(cov)==26
assert Counter(r['existing_spacing_only_result_status'] for r in cov)=={'NOT_FOUND':22,'VERIFIED':4}
adopted={r['budget']:r['paper_adopted_hd95_mm'] for r in cov if r['method']=='Full model'}
assert adopted=={'5%':'22.79','10%':'11.88'}
spacing=read('COVID_EXISTING_SPACING_ONLY_FOUR_ROWS.csv')
assert len(spacing)==4 and all(r['paper_replacement_authorized']=='false' for r in spacing)
assert all(r['before_dice']==r['after_dice'] and r['before_iou']==r['after_iou'] for r in spacing)
assert len(read('COVID_HIEN_PMEUL_INCLUSION_AUDIT.csv'))==4
for p in OUT.rglob('*'):
    assert not p.is_symlink(),p
    if p.is_file():
        assert p.name=='SHA256SUMS' or p.suffix in {'.md','.csv','.json','.txt','.py','.list'},p
        assert p.stat().st_size<2*1024*1024,p
validation=dict(time_bjt=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    source_records_checked=len(sources),source_sha_all_match=True,export_sha_all_match=True,
    paper_rows=73,mapped_rows=20,missing_rows=53,field_records=73*16,
    promise_four_scalar_differences_zero=True,b_generation_missing_fields=14,c_target_binding_missing_rows=6,
    paper_covid_full_hd95_mm=adopted,only_small_text_files=True,original_files_modified=False,
    validation_scope='This new package only; previous 237-file/116-mean audit not repeated',status='PASS')
(OUT/'PACKAGE_VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
manifest=''.join(f'{digest(p.read_bytes())}  {p.relative_to(OUT).as_posix()}\n' for p in files)
(OUT/'SHA256SUMS').write_text(manifest)
archive=ROOT/(OUT.name+'.tar.gz')
with tarfile.open(archive,'w:gz') as t:
    for p in sorted(p for p in OUT.rglob('*') if p.is_file()):
        t.add(p,arcname=(Path(OUT.name)/p.relative_to(OUT)).as_posix(),recursive=False)
assert archive.stat().st_size<30*1024*1024
with tarfile.open(archive,'r:gz') as t:
    members=t.getmembers()
    contents={}
    for m in members:
        rel=PurePosixPath(m.name)
        assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
        assert rel.parts[0]==OUT.name
        contents[str(PurePosixPath(*rel.parts[1:]))]=t.extractfile(m).read()
    archived_manifest=contents.pop('SHA256SUMS').decode('utf-8')
    assert archived_manifest==manifest
    for s in archived_manifest.splitlines():
        h,n=s.split('  ',1)
        assert digest(contents[n])==h
    assert len(contents)==len(files)
archive_hash=digest(archive.read_bytes())
(ROOT/(archive.name+'.sha256')).write_text(f'{archive_hash}  {archive.name}\n')
archive_validation=dict(archive=str(archive),archive_bytes=archive.stat().st_size,sha256=archive_hash,
    archive_regular_files=len(members),internal_manifest_entries=len(files),internal_sha_all_match=True,
    safe_relative_paths=True,all_csv_utf8_readable=True,under_30_mib=True,status='PASS')
(ROOT/(archive.name+'.ARCHIVE_VALIDATION.json')).write_text(json.dumps(archive_validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(archive_validation,ensure_ascii=False))
