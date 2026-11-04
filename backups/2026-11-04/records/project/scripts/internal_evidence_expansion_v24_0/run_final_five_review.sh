#!/usr/bin/env bash
set -euo pipefail
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main
exec scripts/internal_evidence_expansion_v24_0/run_postfull_finalization.sh --task-id FINAL_FIVE_REVIEW "$@"
