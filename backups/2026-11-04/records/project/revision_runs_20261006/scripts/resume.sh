#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
if test -f state/queue.json; then
  if /home/renguanming/.conda/envs/beta_fft/bin/python -c 'import json,os,sys; s=json.load(open("state/queue.json")); p=s.get("controller_pid"); sys.exit(0 if p and s.get("status")=="RUNNING" and os.path.exists("/proc/"+str(p)+"/cmdline") and b"revision_runs_20261006/scripts/run_queue.py" in open("/proc/"+str(p)+"/cmdline","rb").read() else 1)'; then
    echo 'Controller is active; resume would duplicate it.'
    exec scripts/status.sh
  fi
fi
rm -f state/stop_requested
exec scripts/start_tmux.sh
