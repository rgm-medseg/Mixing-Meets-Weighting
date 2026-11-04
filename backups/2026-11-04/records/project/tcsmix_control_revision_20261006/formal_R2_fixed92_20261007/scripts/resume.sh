#!/bin/bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
exec /bin/bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/start_tmux.sh --resume
