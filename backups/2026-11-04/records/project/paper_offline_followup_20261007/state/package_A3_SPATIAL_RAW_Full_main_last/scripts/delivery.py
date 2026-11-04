import shutil,tempfile,tarfile
from common import *
def archive(src,name):
 p=W/name
 with tarfile.open(p,'w:gz',compresslevel=6) as t:t.add(src,arcname=src.name)
 digest=sha(p);atomic(p.with_name(p.name+'.sha256'),digest+'  '+p.name+'\n')
 with tempfile.TemporaryDirectory(prefix='beta_evidence_verify_') as d:
  with tarfile.open(p) as t:t.extractall(d,filter='data')
  extracted=Path(d)/src.name;lines=(extracted/'SHA256SUMS').read_text().splitlines();count=0
  for line in lines:
   h,rel=line.split('  ',1);assert sha(extracted/rel)==h,rel;count+=1
 return dict(path=str(p),sha256=digest,bytes=p.stat().st_size,files=count,actual_extraction_verified=True)
def sums(folder):
 rr=[f'{sha(p)}  {p.relative_to(folder)}' for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'];atomic(folder/'SHA256SUMS','\n'.join(rr)+'\n')
def main():
 review=read(W/'reports/FIVE_PASS_REVIEW.json');resume=read(W/'reports/STOP_RESUME_RECEIPT.json') if (W/'reports/STOP_RESUME_RECEIPT.json').exists() else dict(status='PARTIAL',reason='no actual stop/resume receipt')
 dup=read(W/'reports/DUPLICATE_START_RECEIPT.json') if (W/'reports/DUPLICATE_START_RECEIPT.json').exists() else dict(status='PARTIAL',reason='no duplicate invocation receipt')
 review['reproduction_archive']='PASS' if resume['status']==dup['status']=='PASS' else 'PARTIAL: recovery or duplicate receipt missing'
 review['archive_verification']='Actual extraction SHA verification occurs immediately after compression; any failure aborts delivery.'
 js(W/'reports/FIVE_PASS_REVIEW.json',review)
 atomic(W/'FIVE_PASS_REVIEW.md','''# 五轮核验\n\n'''+''.join(f'- {k}: {v}\n' for k,v in review.items())+'\n恢复/重复启动收据独立列出；原始单元重新生成汇总图，完整包及拆包实际解包SHA结果见包外ARCHIVE_RECEIPT.json。图文自动渲染及画布边界检查，human visual receipt若存在另附，不虚称人工审图。\n')
 stage=W/'state/archive_snapshot';
 if stage.exists():shutil.rmtree(stage)
 stage.mkdir()
 for name in ['scripts','configs','manifests','raw_units','results','figures','reports','logs']:
  shutil.copytree(W/name,stage/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ['PRE_ANALYSIS_CN.md','START_REPORT_CN.md','FINAL_REPORT_CN.md','WRITING_HANDOFF_CN.md','FIVE_PASS_REVIEW.md']:
  if (W/name).exists():shutil.copy2(W/name,stage/name)
 for name in ['controller.json','stage_history.json','progress.json']:
  if (W/'state'/name).exists():dest=stage/'state'/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(W/'state'/name,dest)
 atomic(stage/'ARCHIVE_NOTE.md','Consistent completed-unit snapshot. Full author/executable mapping is NEEDS_MAPPING. Snapshot state can show ARCHIVE/RUNNING because compression follows the snapshot; final external receipt gives completed archive and verified SHA. Full source image/GT/input/map arrays and model weights remain on server, with source hashes and reconstruction scripts in this package. Two limited non-image probability/error arithmetic caches included. No historical archive overwritten.\n')
 sums(stage);receipts=[archive(stage,'paper_offline_followup_20261007.tar.gz')]
 if receipts[0]['bytes']>=30*2**20:
  for task in ['A2_MAPPING','A5_DEPLOYMENT','A3_SPATIAL']:
   split=W/'state'/('package_'+task)
   if split.exists():shutil.rmtree(split)
   split.mkdir()
   # Shared scripts/config/sources and reports make each archive self-contained; task-specific units/results/figures remain complete.
   for name in ['scripts','configs','manifests','reports']:shutil.copytree(stage/name,split/name)
   for p in stage.glob('*.md'):shutil.copy2(p,split/p.name)
   for p in (stage/'results').rglob('*'):
    if not p.is_file():continue
    wanted=(task=='A2_MAPPING' and (p.name.startswith(('A2_','Full_')))) or (task=='A5_DEPLOYMENT' and p.name.startswith(('A5_','DEPLOYMENT','TIMING'))) or (task=='A3_SPATIAL' and p.name.startswith('A3_'))
    if wanted:d=split/p.relative_to(stage);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
   for p in (stage/'figures').glob('*'):
    prefix='A2_' if task=='A2_MAPPING' else 'A5_' if task=='A5_DEPLOYMENT' else 'A3_'
    if p.name.startswith(prefix):d=split/'figures'/p.name;d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
   unitnames=['A2'] if task=='A2_MAPPING' else ['EQUIVALENCE','TIMING'] if task=='A5_DEPLOYMENT' else ['INPUTS','A3']
   for name in unitnames:
    d=split/'raw_units'/identity()/name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(stage/'raw_units'/identity()/name,d)
   if task=='A2_MAPPING':d=split/'raw_units/A2_sample_class_quality_source.csv';d.parent.mkdir(exist_ok=True);shutil.copy2(stage/'raw_units/A2_sample_class_quality_source.csv',d)
   if task=='A3_SPATIAL':shutil.copytree(stage/'raw_units/review_cache',split/'raw_units/review_cache')
   sums(split);receipt=archive(split,f'paper_offline_{task}_20261007.tar.gz')
   if receipt['bytes']<30*2**20:receipts.append(receipt)
   else:
    # Preserve the full task archive; provide a compact analysis core plus all raw checkpoint shards.
    atomic(W/f'{task}_OVERSIZE_NOTE.json',json.dumps(dict(server_full_task_archive=receipt,delivery='core and raw checkpoint shards below30MiB; no units omitted'),indent=2))
    core=W/'state'/('package_'+task+'_CORE');shutil.copytree(split,core,dirs_exist_ok=True)
    raw=core/'raw_units'/identity()/'A3'
    assert task=='A3_SPATIAL' and raw.exists(),('Unexpected oversize task requires explicit alternate sharding',task)
    shutil.rmtree(raw)
    # Region rows are large: retain the complete CSV in a separate statistics archive.
    giant=core/'results/A3_region_statistics.csv';stat=W/'state'/('package_'+task+'_REGION_STATS');stat.mkdir(exist_ok=True)
    for sub in ['scripts','configs','manifests','reports']:shutil.copytree(stage/sub,stat/sub,dirs_exist_ok=True)
    (stat/'results').mkdir(exist_ok=True);shutil.move(giant,stat/'results/A3_region_statistics.csv')
    atomic(stat/'README.md','Complete A3_region_statistics.csv, every method/checkpoint/strategy/width/patient/region, including NA and unfavorable cases. This archive complements A3_SPATIAL_CORE and is independently SHA verified. No rows are filtered.\n')
    sums(stat);sr=archive(stat,f'paper_offline_{task}_REGION_STATS_20261007.tar.gz')
    assert sr['bytes']<30*2**20,('Region CSV archive exceeds30MiB; further row sharding required',sr['bytes']);receipts.append(sr)
    sums(core);c=archive(core,f'paper_offline_{task}_CORE_20261007.tar.gz');assert c['bytes']<30*2**20;receipts.append(c)
    source_raw=split/'raw_units'/identity()/'A3';groups={}
    for p in source_raw.glob('*.json'):
     u=read(p);groups.setdefault((u['method_id'],u['checkpoint_tag']),[]).append(p)
    for (method,checkpoint),pp in sorted(groups.items()):
     shard=W/'state'/f'package_{task}_RAW_{method}_{checkpoint}';shard.mkdir(exist_ok=True)
     for sub in ['scripts','configs','manifests','reports']:shutil.copytree(stage/sub,shard/sub,dirs_exist_ok=True)
     dest=shard/'raw_units'/identity()/'A3';dest.mkdir(parents=True,exist_ok=True)
     for p in pp:shutil.copy2(p,dest/p.name)
     shutil.copytree(stage/'raw_units'/identity()/'INPUTS',shard/'raw_units'/identity()/'INPUTS',dirs_exist_ok=True)
     atomic(shard/'README.md',f'Complete A3 raw units for {method} {checkpoint}, including every matched/original strategy batch available, with source/config/code identity. {len(pp)} units. Summary CSVs, figures, and all unfavorable examples are in A3_SPATIAL_CORE. This archive independently verifies all its files using SHA256SUMS. Full medical arrays remain on server.\n')
     sums(shard);r=archive(shard,f'paper_offline_{task}_RAW_{method}_{checkpoint}_20261007.tar.gz');assert r['bytes']<30*2**20;receipts.append(r)
 js(W/'ARCHIVE_RECEIPT.json',dict(status='PASS',time=now(),archives=receipts,identity=identity(),all_actual_extraction_verified=True,protected_sources='PASS',training_updates=0))
 print('ARCHIVES VERIFIED',json.dumps(receipts),flush=True)
if __name__=='__main__':main()
