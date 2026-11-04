#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export BETA_TASK=E5 BETA_TASK_WORK=/home/renguanming/jupyterlab/rgm/beta-FFT-main/formal_E4_E5_20261008/tasks/E5
exec /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/formal_E4_E5_20261008/scripts/train_E5_entry.py "$@"
