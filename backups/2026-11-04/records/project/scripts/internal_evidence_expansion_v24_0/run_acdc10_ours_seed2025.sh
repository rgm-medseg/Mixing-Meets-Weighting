#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec /home/renguanming/.conda/envs/beta_fft/bin/python scripts/internal_evidence_expansion_v24_0/v24_new_full_launcher.py --config configs/internal_evidence_expansion_v24_0/ACDC10_OURS_SEED2025.json "$@"
