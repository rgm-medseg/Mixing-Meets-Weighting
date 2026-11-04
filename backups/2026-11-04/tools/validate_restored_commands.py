#!/usr/bin/env python3
"""Validate restored-file hashes, lineage, and CLI parsing; never launch training."""
import json, os, subprocess
from pathlib import Path
TASK=Path(__file__).resolve().parents[1];CAT=TASK/'catalog';PY='/home/renguanming/.conda/envs/beta_fft/bin/python'
reports=[]
for k in ['acdc5','acdc10','promise20','covid5','covid10']:
    r=subprocess.run([PY,str(TASK/'tools/resume_main.py'),'--catalog',str(CAT),'--restored',str(TASK/'verification_local'),
        '--setting',k,'--data-root','/datasets','--output-root',str(TASK/'unused_resume_output')],capture_output=True,text=True)
    row={'setting':k,'command_build':'PASS' if r.returncode==0 else 'FAIL','training_started':False,'evaluation_started':False}
    if r.returncode==0:
        spec=json.loads(r.stdout);cmd=spec['command']+['--help'];project=TASK/'verification_local/project'
        path=project/('train_ACDC' if k.startswith('acdc') else 'train_promise12')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(path)+os.pathsep+str(project))
        test=subprocess.run(cmd,cwd=project,env=env,capture_output=True,text=True,timeout=180)
        row.update(checkpoint=spec['checkpoint'],completed_updates=spec['completed_updates'],
                   parser_check='PASS' if test.returncode==0 else 'FAIL',parser_exit_code=test.returncode,
                   parser_error=test.stderr[-2000:] if test.returncode else '')
    else:row['error']=r.stderr[-2000:]
    reports.append(row);print(k,row,flush=True)
(CAT/'reports/restored_command_validation.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
if any(r.get('parser_check')!='PASS' for r in reports):raise SystemExit(1)
