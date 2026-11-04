#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
SESSION=beta_fft_revision_20261006
if test -f state/queue.json; then
  if /home/renguanming/.conda/envs/beta_fft/bin/python -c 'import json,os,sys; s=json.load(open("state/queue.json")); p=s.get("controller_pid"); active=s.get("status")=="RUNNING"; sys.exit(0 if p and active and os.path.exists("/proc/"+str(p)+"/cmdline") and b"revision_runs_20261006/scripts/run_queue.py" in open("/proc/"+str(p)+"/cmdline","rb").read() else 1)'; then
    exec scripts/status.sh
  fi
fi
if test -f state/stop_requested; then
  echo 'Safe-stop request exists; use resume.sh to resume this run.'
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n queue-resume
else
  tmux new-session -d -s "$SESSION" -n queue
fi
TARGET=$(tmux display-message -p -t "$SESSION" '#{window_id}')
tmux set-option -w -t "$TARGET" remain-on-exit on
tmux send-keys -t "$TARGET" "exec /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/run_queue.py >> /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/logs/queue.log 2>&1" Enter
if ! tmux list-windows -t "$SESSION" -F '#{window_name}' | rg -q '^monitor$'; then
  tmux new-window -d -t "$SESSION" -n monitor "watch -n 10 /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/status.sh"
fi
tmux set-option -w -t "$SESSION:monitor" remain-on-exit on
scripts/status.sh
