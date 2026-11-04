#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec /home/renguanming/.conda/envs/beta_fft/bin/python scripts/internal_evidence_expansion_v24_0/v24_frozen_best_test_launcher.py --task-id ACDC10_BETAFFT_SEED2025 "$@"
