#!/usr/bin/env python3
"""Print or execute a main experiment's retained original-budget resume command.

Uses ACDC validation-best full state or the binary 25000-update full state by default.
Final states at 30000 updates are retained for recovery/inspection but have no remaining budget.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def argv(settings,acdc):
    out=[]
    for key,value in settings.items():
        if value is None:continue
        flag='--'+(key.replace('_','-') if key in {'cache_mode','amp_opt_level'} else key)
        if isinstance(value,bool) and not acdc:
            if value:out.append(flag)
        else:
            out.append(flag)
            out.extend(map(str,value)) if isinstance(value,list) else out.append(str(value).lower() if isinstance(value,bool) else str(value))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--restored',type=Path,required=True)
    p.add_argument('--setting',choices=['acdc5','acdc10','promise20','covid5','covid10'],required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu',default='0');p.add_argument('--execute',action='store_true');p.add_argument('--checkpoint',help='Source-relative path, must be a retained training state below the original budget')
    a=p.parse_args();cat=a.catalog.resolve();restored=a.restored.resolve();project=restored/'project';data=a.data_root.resolve();output=a.output_root.resolve()/a.setting
    e=next(e for e in json.loads((cat/'experiments.json').read_text()) if e['id']==a.setting)
    weights=json.loads((cat/'weights.json').read_text());audit=json.loads((cat/'reports/checkpoint_audit.json').read_text());acdc=e['dataset']=='ACDC'
    selected=next(w for w in weights if a.setting in w['experiment_ids'] and
                  (w['source']==a.checkpoint if a.checkpoint else w['sha256']==e['expected_best_sha256'] if acdc else w['source'].endswith('self_train_full_state_iter_25000.pth')))
    checkpoint=project/selected['source'];state=audit[selected['sha256']]
    if checkpoint.is_file() and sha(checkpoint)!=selected['sha256']:raise ValueError('Retained state SHA mismatch')
    config=json.loads((cat/e['portable_config']).read_text());settings=dict(config['training_args'])
    settings.update(root_path=str(data/config['data_subdir']),gpu=a.gpu)
    plans=[];init=project/e['init_path'];iteration=state.get('iteration')
    if iteration is None or iteration>=30000:raise ValueError('No remaining original-budget updates in this state; choose an earlier retained full state')
    if acdc:
        settings=dict(state['args']);settings.update(root_path=str(data/'ACDC'),gpu=a.gpu,run_root=str(output),pretrained_model=str(init.parent),resume_ckpt=str(checkpoint))
        settings['seed']=e['seed']
        entry=project/e['run_root']/'effective_source.py'
        env=dict(os.environ,PYTHONPATH=str(project/'train_ACDC')+os.pathsep+str(project),CUDA_VISIBLE_DEVICES=a.gpu)
    else:
        entry=project/'train_promise12/train_promise12_bcp_fft.py';lineage=state['lineage']
        if entry.is_file() and sha(entry)!=lineage['training_core_sha256']:raise ValueError('Historical training-core SHA mismatch')
        if init.is_file() and sha(init)!=lineage['pretrain_checkpoint_sha256']:raise ValueError('Initialization SHA mismatch')
        settings.update(v22_output_root_override=str(output),self_train_resume_checkpoint=str(checkpoint),
                        cfg=str(project/'train_promise12/configs/swin_tiny_patch4_window7_224_lite.yaml'))
        settings['exp']=lineage['run_id']
        if not acdc and e['dataset']=='COVID-19':
            for role in ['labeled','unlabeled','val']:
                name=Path(config['manifests'][role]).name;path=project/'data/COVID19_OFFICIAL20/manifests'/name
                expected=lineage['covid19_'+role+'_manifest_sha256']
                if path.is_file() and sha(path)!=expected:raise ValueError('Original manifest SHA mismatch: '+role)
                settings['covid19_'+role+'_manifest']=str(path)
            link=project/'data/COVID19_OFFICIAL20/derived';target=data/'COVID19_OFFICIAL20/derived'
            plans.append({'symlink':str(link),'target':str(target)})
        plans.append({'copy_initialization':str(init),'destination':str(output/'pre_train/unet_best_model.pth')})
        env=dict(os.environ,PYTHONPATH=str(project/'train_promise12')+os.pathsep+str(project),CUDA_VISIBLE_DEVICES=a.gpu)
    command=[sys.executable,str(entry),*argv(settings,acdc)]
    print(json.dumps({'experiment':e['id'],'checkpoint':str(checkpoint),'completed_updates':iteration,'original_target_updates':30000,
                      'command':command,'cwd':str(project),'preparation':plans,'execution_requested':a.execute},ensure_ascii=False,indent=2))
    if not a.execute:return
    if output.exists():raise FileExistsError('Choose a fresh output root: '+str(output))
    if not checkpoint.is_file() or not entry.is_file() or not init.is_file():raise FileNotFoundError('Restore and verify all backup objects first')
    if not (data/config['data_subdir']).is_dir():raise FileNotFoundError('Dataset is required separately')
    for plan in plans:
        if 'symlink' in plan:
            link=Path(plan['symlink']);target=Path(plan['target'])
            if not target.is_dir():raise FileNotFoundError(target)
            if link.exists():
                if link.resolve()!=target.resolve():raise FileExistsError(link)
            else:link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(target,target_is_directory=True)
        else:
            dest=Path(plan['destination']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(plan['copy_initialization'],dest)
    output.parent.mkdir(parents=True,exist_ok=True)
    raise SystemExit(subprocess.run(command,cwd=project,env=env).returncode)

if __name__=='__main__':main()
