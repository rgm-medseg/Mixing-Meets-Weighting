#!/usr/bin/env bash
# Safe stages only: no training/test/checkpoint-selection commands.
set -euo pipefail
TASK_PY=/home/renguanming/.conda/envs/beta_fft/bin/python
TASK_DIR=/home/renguanming/jupyterlab/rgm/beta-FFT-main/mechanism_direction_20261017
export PYTHONDONTWRITEBYTECODE=1
case "${1:-help}" in
  inventory) "$TASK_PY" "$TASK_DIR/scripts/inventory.py" ;;
  main) "$TASK_PY" "$TASK_DIR/scripts/analyze_direction.py" ;;
  factorial) "$TASK_PY" "$TASK_DIR/scripts/factorial_direction.py" ;;
  report) "$TASK_PY" "$TASK_DIR/scripts/make_delivery.py" ;;
  audit) "$TASK_PY" "$TASK_DIR/scripts/audit_delivery.py" ;;
  package) "$TASK_PY" "$TASK_DIR/scripts/package_delivery.py" ;;
  *) echo 'Stages: inventory main factorial report audit package. Cached main probabilities and factorial logits avoid repeating forwards. No default execution.' ;;
esac
