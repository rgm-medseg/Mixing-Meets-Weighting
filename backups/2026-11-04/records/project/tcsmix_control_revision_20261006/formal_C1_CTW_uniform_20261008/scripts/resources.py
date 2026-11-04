"""GPU0 admission and container/host PID evidence; never signal foreign jobs."""
import subprocess,time
from common import *
UUID='GPU-0a89fa49-2afe-ac0e-2fe5-2ba2541aeb59'
def gpu():
    g=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,name,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(', ')
    a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert g[0]==UUID and 'A100' in g[1]
    return dict(uuid=g[0],name=g[1],total_mib=int(g[2]),used_mib=int(g[3]),free_mib=int(g[4]),utilization=int(g[5]),pids=[int(x.split(', ')[1]) for x in a if x and x.split(', ')[0]==UUID])
def admission():
    while True:
        if (WORK/'state/stop_requested').exists():raise InterruptedError('C1 stop requested during resource wait')
        g=gpu()
        if not g['pids'] and g['used_mib']<2048 and g['free_mib']>=30000 and g['utilization']<=10:return g
        atomic_json(WORK/'state/resource_wait.json',dict(status='WAITING_FOR_RESOURCE',time=now(),resource=g));time.sleep(5)
def bind_gpu_pid():
    import torch
    assert not torch.cuda.is_initialized(),'PID binding must precede all CUDA allocation'
    before=admission();torch.cuda.init();x=torch.empty(1,device='cuda');torch.cuda.synchronize();after=gpu();del x
    added=set(after['pids'])-set(before['pids']);assert len(added)==1,'Cannot uniquely identify own NVML process'
    atomic_json(WORK/f'reports/GPU_PID_BINDING_{os.getpid()}.json',dict(status='PASS',time=now(),container_pid=os.getpid(),NVML_pid=next(iter(added)),GPU_uuid=UUID,before=before,after=after))
