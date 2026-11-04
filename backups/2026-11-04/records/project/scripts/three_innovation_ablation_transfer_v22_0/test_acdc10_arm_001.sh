#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec /home/renguanming/.conda/envs/beta_fft/bin/python tools/v22_0_acdc_test_frozen_checkpoint.py --config configs/three_innovation_ablation_transfer_v22_0/ACDC10_ARM_001.json --checkpoint experiments/three_innovation_ablation_transfer_v22_0/ACDC10_ARM_001/checkpoints/best.ckpt --output-dir final_report/three_innovation_ablation_transfer_v22_0/raw_test/ACDC10_ARM_001 "$@"
