"""Read actual completed optimizer-update records and pipeline state without GPU initialization."""
import csv,json,time
from pathlib import Path
W=Path(__file__).resolve().parents[1]
def main():
    s=json.loads((W/'receipts/pipeline_status.json').read_text());p=W/'runs/F97_main_signal_20261013/audit/training_dynamics.csv';rows=[]
    if p.exists():
        with p.open() as f:rows=list(csv.DictReader(f))
    valid=[r for r in rows if r.get('global_step','').isdigit() and r.get('checkpoint_saved') is not None]
    s.update(completed_updates=max([int(r['global_step']) for r in valid],default=0),budget=30000)
    if valid:
        s['finite_loss_gradient_logits']=all(all(r[k]=='1' for k in ['loss_finite','grad_finite','logits_finite']) for r in valid)
    print(json.dumps(s))
if __name__=='__main__':main()
