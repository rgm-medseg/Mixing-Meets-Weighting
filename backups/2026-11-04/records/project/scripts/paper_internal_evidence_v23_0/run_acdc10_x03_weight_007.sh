#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec /home/renguanming/.conda/envs/beta_fft/bin/python tools/v23_0_acdc10_x03_sensitivity_launcher.py --config configs/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_007.json "$@"
