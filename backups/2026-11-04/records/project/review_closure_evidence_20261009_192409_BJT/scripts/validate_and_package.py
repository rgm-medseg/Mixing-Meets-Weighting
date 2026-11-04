"""Validate and archive this NEW audit directory. Never modify original sources."""
import ast
import csv
import hashlib
import json
import re
import socket
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f,delimiter='\t' if p.suffix=='.tsv' else ','))
def load(n):return json.loads((OUT/n).read_text())
def fail(message):raise AssertionError(message)

index=rows(OUT/'SOURCE_INDEX.csv')
indexed_exports={r['packed_path'] for r in index if r['packed_path']}
assert all(str(p.relative_to(OUT)) in indexed_exports for p in (OUT/'evidence').rglob('*') if p.is_file()),'Every evidence export must have an unambiguous SOURCE_INDEX row'
checked_sources=[];missing=[];copy_count=0;excerpt_count=0
for r in index:
    p=Path(r['original_path'])
    if r['exists']=='False':
        assert not p.exists(),r['original_path'];missing.append(r['source_id']);continue
    assert p.is_file(),r['original_path']
    current=sha(p)
    assert current==r['sha256'],'Source changed since collection: '+r['source_id']
    if r['expected_sha256']:assert r['expected_sha_matches']=='True',r['source_id']
    checked_sources.append({'source_id':r['source_id'],'sha256':current,'matches_collected_sha':True})
    if r['packed_path']:
        packed=OUT/r['packed_path'];assert packed.is_file(),str(packed)
        assert packed.resolve().is_relative_to(OUT),str(packed)
        if r['packed_kind'] in ['BYTE_IDENTICAL_COPY','NEW_CPU_AUDIT_ARTIFACT']:
            assert sha(packed)==current,r['source_id'];copy_count+=1
        elif r['packed_kind']=='LINE_NUMBERED_EXCERPT':
            source=p.read_text().splitlines();lines=packed.read_text().splitlines()
            assert lines[:2]==['ORIGINAL: '+str(p),'SHA256: '+current],r['source_id']
            for line in lines[2:]:
                m=re.fullmatch(r'(\d+): (.*)',line)
                assert m,r['source_id']+': excerpt line format'
                n=int(m.group(1));assert 1<=n<=len(source)
                assert source[n-1]==m.group(2),r['source_id']+': excerpt mismatch'
            excerpt_count+=1

mapping=rows(OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv')
assert len(mapping)==29 and len({r['row_id'] for r in mapping})==29
assert sum(r['table_scope']=='TABLE2_4_CANDIDATE' for r in mapping)==24
assert all(r['checkpoint_sha_match']=='True' for r in mapping)
assert all(r['paper_identity']=='REPRODUCED_AUTHOR_CONFIRMED' for r in mapping)
main=next(r for r in mapping if r['paper_method']=='完整方法 / 论文 R1')
control=next(r for r in mapping if r['paper_method']=='更换混合控制信号')
assert main['dice_display']=='90.35' and main['checkpoint_selected_step']=='29600'
assert control['dice_display']=='89.98' and control['historical_run']=='R1_v2_B'
assert all(r['checkpoint_selected_step'] for r in mapping)
external=load('EXTERNAL_SELECTED_CHECKPOINT_CPU_METADATA.json')
assert len(external)==14 and all(not r['forward_called'] and not r['checkpoint_selection_changed'] for r in external)
assert all(r['selected_model_role']=='teacher' for r in external if r['row_id'] in ['ROW13','ROW14'])
agg=rows(OUT/'AGGREGATE_RECOMPUTATION.csv')
assert len(agg)==116 and all(r['within_1e_12']=='True' and r['finite']=='True' for r in agg)
assert {r['row_id'] for r in agg}=={r['row_id'] for r in mapping}
assert not rows(OUT/'NONFINITE_RECORDS.csv')
r2=load('R2_RECORD_VERIFICATION.json')
assert r2['class_rows']==120 and r2['unique_case_class_keys']==120 and r2['patients']==20 and r2['patient_table_rows']==20
assert r2['patient_table_max_error']<1e-12 and r2['class_table_max_error']<1e-12
budget=load('ACDC_TRAINING_BUDGET_RECOMPUTATION.json')
assert len(budget)==2 and all(r['unique_steps_exactly_1_through_30000'] for r in budget)
assert all(r['student1_updates_supported']==30000 and r['student2_updates_supported']==30000 for r in budget)
coverage=rows(OUT/'COVID_RESULT_VERSION_COVERAGE.csv')
assert Counter(r['status'] for r in coverage)=={'VERIFIED_EXISTING_RESULT':4,'NOT_FOUND':6}
assert all(float(r['delta_dice'])==0 and float(r['delta_iou'])==0 for r in coverage if r['status']=='VERIFIED_EXISTING_RESULT')
geometry=rows(OUT/'COVID_GEOMETRY_RECOMPUTATION.csv')
assert len(geometry)==4 and all(float(r['spacing_max_error'])==0 and r['Z_spacing_unchanged']=='True' for r in geometry)

report=(OUT/'REVIEW_CLOSURE_EVIDENCE.md').read_text()
references=re.findall(r'`(/[^`]+)`，行 ([0-9,]+)；SHA256 `([a-f0-9]{64})`（(S\d+)）',report)
by_id={r['source_id']:r for r in index}
for path,numbers,digest,sid in references:
    r=by_id[sid];assert path==r['original_path'] and digest==r['sha256']
    assert r['packed_path'],sid+': cited textual evidence not exported'
    total=int(r['original_lines']);assert all(1<=int(n)<=total for n in numbers.split(',')),sid
    if r['packed_kind']=='LINE_NUMBERED_EXCERPT':
        exported={int(n) for n in re.findall(r'(?m)^(\d+): ',(OUT/r['packed_path']).read_text())}
        assert set(map(int,numbers.split(','))).issubset(exported),sid+': citation absent from excerpt'
assert 'VERIFIED_EXISTING_RESULT' not in report or coverage
assert len(references)>=20

allowed={'.json','.csv','.md','.list','.py','.tsv','.sha256','.yaml','.txt'}
for p in OUT.rglob('*'):
    assert not p.is_symlink(),'No symlinks allowed: '+str(p)
    if not p.is_file():continue
    assert p.suffix in allowed,'Disallowed artifact: '+str(p)
    assert p.stat().st_size<1200000,'Oversize artifact: '+str(p)
    text=p.read_text() # all package artifacts must be text, never arrays or weights
    assert not re.search(r'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----|AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{30,}',text),'Credential-shaped contents excluded'
    if p.suffix=='.json':json.loads(text)
    if p.suffix in ['.csv','.tsv']:rows(p)
    if p.parent==OUT/'scripts' and p.suffix=='.py':ast.parse(text,filename=str(p))

receipt={
 'validated_bjt':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'host':socket.gethostname(),
 'scope':'Existing records and CPU-only parse/recalculation; no original mutation, training, checkpoint reselection, GPU forward or process signals',
 'source_index_rows':len(index),'existing_sources_sha_rechecked':len(checked_sources),'missing_source_ids':missing,
 'byte_identical_copies_verified':copy_count,'line_numbered_excerpts_verified':excerpt_count,
 'source_citations_verified':len(references),'run_rows':29,'table_candidate_rows':24,'supplemental_rows':5,
 'all_mapped_checkpoints_have_selected_step':True,'aggregate_checks':len(agg),'aggregate_failures':0,
 'R2_case_class_records':120,'R2_patient_records':20,'R2_class_summary_records':3,
 'ACDC_Full5_Full10_SSL_unique_steps_each':30000,
 'COVID_corrected_existing_rows':4,'COVID_external_corrected_rows_not_found':6,
 'text_only_no_image_weight_array_or_credentials':True,'checked_sources':checked_sources,
 'closure_status':{'A':'PARTIAL','B':'PARTIAL','C':'PARTIAL','D':'VERIFIED'},
 'archive_validation_record':'External ARCHIVE_VALIDATION.json records final archive/member-hash verification to avoid a recursive archive digest inside itself.'
}
(OUT/'VALIDATION_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')

files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
manifest={str(p.relative_to(OUT)):sha(p) for p in files}
(OUT/'SHA256SUMS.txt').write_text(''.join(digest+'  '+name+'\n' for name,digest in manifest.items()))
bundle=ROOT/(OUT.name+'.tar.gz')
assert not bundle.exists(),'Refuse to overwrite an existing archive'
with tarfile.open(bundle,'w:gz') as tar:
    tar.add(OUT,arcname=OUT.name,recursive=True)
assert bundle.stat().st_size<30*1024*1024
with tarfile.open(bundle,'r:gz') as tar:
    members=tar.getmembers();packed_files=[m for m in members if m.isfile()]
    assert len(packed_files)==len(files)+1
    for m in members:
        assert not m.issym() and not m.islnk() and '..' not in Path(m.name).parts
        assert m.name==OUT.name or m.name.startswith(OUT.name+'/')
    packed_manifest=tar.extractfile(OUT.name+'/SHA256SUMS.txt').read().decode()
    assert packed_manifest==(OUT/'SHA256SUMS.txt').read_text()
    for name,digest in manifest.items():
        assert hashlib.sha256(tar.extractfile(OUT.name+'/'+name).read()).hexdigest()==digest,name
bundle_sha=sha(bundle)
checksum=Path(str(bundle)+'.sha256')
checksum.write_text(bundle_sha+'  '+bundle.name+'\n')
archive_validation={
 'verified_bjt':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'archive':str(bundle),
 'sha256':bundle_sha,'compressed_bytes':bundle.stat().st_size,'files_including_inner_manifest':len(files)+1,
 'all_archive_file_sha256_verified':True,'readable_tar_gzip':True,'under_30_MiB':True,
 'no_image_weight_array_or_credentials':True,'source_citations_verified':len(references),
 'directory_snapshot_manifest':str(OUT/'SHA256SUMS.txt')
}
Path(str(bundle)+'.ARCHIVE_VALIDATION.json').write_text(json.dumps(archive_validation,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(archive_validation,ensure_ascii=False,indent=2))
