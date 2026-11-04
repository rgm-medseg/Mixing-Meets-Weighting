#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
scripts/resume.sh
if /home/renguanming/.conda/envs/beta_fft/bin/python -c 'import json,pathlib,sys; p=pathlib.Path("state/supplementary_audits.json"); s=json.loads(p.read_text()) if p.exists() else {}; cmd=pathlib.Path("/proc/"+str(s.get("pid"))+"/cmdline"); args=cmd.read_bytes().split(b"\0") if cmd.exists() else []; sys.exit(0 if any(a.endswith(b"/scripts/supplementary_audits.py") for a in args) and b"--watch" in args else 1)'; then
  exit 0
fi
if tmux list-windows -t beta_fft_revision_20261006 -F '#{window_name}' | rg -q '^audits$'; then
  tmux respawn-pane -t beta_fft_revision_20261006:audits "cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006 && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/supplementary_audits.py --watch >> /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/logs/supplementary_audits.log 2>&1"
else
  tmux new-window -d -t beta_fft_revision_20261006 -n audits "cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006 && exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/supplementary_audits.py --watch >> /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/logs/supplementary_audits.log 2>&1"
fi
tmux set-option -w -t beta_fft_revision_20261006:audits remain-on-exit on
