"""Package scripts, reports and source indexes; large arrays stay in place."""
import sys
sys.dont_write_bytecode = True
import tarfile
from analyze_direction import *


def package():
    commands = [
        ('initial_main_precision_check', 'analyze_direction.py', 'primary.log', 1, 'Strict sign-equality assertion failed at near-zero FP32 gradients; batch0 probability cache retained.'),
        ('main_roundoff_aware', 'analyze_direction.py', 'primary_corrected.log', 0, 'Reused batch0 cache; filled batches1-8 once.'),
        ('trained_inventory', 'inventory.py', 'inventory.log', 0, 'Read-only eight-arm identities and protocol gate.'),
        ('main_cached_reporting_fields', 'analyze_direction.py', 'primary_cached_aggregate.log', 0, 'All9 probability caches reused; no further model forward.'),
        ('conditional_four_trained_models', 'factorial_direction.py', 'factorial.log', 0, '36 frozen paired-student batches; no parameter updates.'),
        ('CPU_reporting', 'make_delivery.py', 'delivery.log', 0, 'CPU reaggregation and model-state paired contrasts.'),
        ('five_reviews', 'audit_delivery.py', 'audit.log', 0, 'Identity, alignment, numerical, interpretation and writing checks.'),
        ('final_text_caption', 'make_delivery.py', 'delivery_final.log', 0, 'CPU-only; added repeat-exposure wording and existing-observation note.'),
    ]
    records = []
    for phase, script, log, code, note in commands:
        command = f'PYTHONDONTWRITEBYTECODE=1 /home/renguanming/.conda/envs/beta_fft/bin/python mechanism_direction_20261017/scripts/{script} > mechanism_direction_20261017/logs/{log} 2>&1'
        records.append(dict(phase=phase, command=command, cwd=str(ROOT), exit_code=code, note=note))
    records.append(dict(phase='package',command='PYTHONDONTWRITEBYTECODE=1 /home/renguanming/.conda/envs/beta_fft/bin/python mechanism_direction_20261017/scripts/package_delivery.py',cwd=str(ROOT),note='Creates the compact local evidence archive; no model or gradient execution.'))
    write_json(OUT/'receipts/ACTUAL_COMMANDS.json',dict(commands=records,additional_reads='Read-only shell/Python inventory, log inspection and official PyTorch documentation check; no training/test invocation.'))
    files = [p for p in sorted(OUT.rglob('*')) if p.is_file() and 'arrays' not in p.relative_to(OUT).parts
             and '__pycache__' not in p.parts and p.suffix!='.gz'
             and p.name not in ['OUTPUT_MANIFEST.csv','delivery_archive.json']]
    manifest=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    write_csv(OUT/'OUTPUT_MANIFEST.csv',manifest)
    archive=OUT/'mechanism_direction_evidence_20261017.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[OUT/'OUTPUT_MANIFEST.csv']:
            tar.add(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)),recursive=False)
    with tarfile.open(archive) as tar:
        names=tar.getnames()
        assert not any('/arrays/' in n for n in names)
        for r in manifest:
            data=tar.extractfile(str(Path(OUT.name)/r['path'])).read()
            assert hashlib.sha256(data).hexdigest()==r['sha256']
    write_json(OUT/'receipts/delivery_archive.json',dict(status='PASS',archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
               included_files=len(files)+1,large_arrays_included=False,
               full_array_index='receipts/new_array_manifest.csv',original_array_index='receipts/main_source_arrays.csv',
               note='Large original inputs/checkpoints and new cached arrays remain in the workspace; package contains complete statistical CSVs and reproducible scripts with indexes.'))
    print(json.dumps(dict(status='PASS',archive=str(archive),bytes=archive.stat().st_size,included_files=len(files)+1)))


if __name__=='__main__':
    package()
