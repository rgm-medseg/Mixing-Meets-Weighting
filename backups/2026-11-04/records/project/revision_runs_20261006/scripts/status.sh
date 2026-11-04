#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1
exec /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/status.py
