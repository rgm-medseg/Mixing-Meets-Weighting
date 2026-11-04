#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
default_mode=canary
if [[ "${1:-}" == "--mode" ]]; then
  exec /home/renguanming/.conda/envs/beta_fft/bin/python tools/v22_0_transfer_task_launcher.py --config configs/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3.json "$@"
fi
exec /home/renguanming/.conda/envs/beta_fft/bin/python tools/v22_0_transfer_task_launcher.py --config configs/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3.json --mode "$default_mode" "$@"
