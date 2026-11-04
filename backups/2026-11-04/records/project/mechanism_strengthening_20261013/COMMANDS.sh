#!/usr/bin/env bash
# Actual executable entry points used this round. Select a stage; running this file without a stage does not retrain.
set -euo pipefail
TASK_PY=/home/renguanming/.conda/envs/beta_fft/bin/python
TASK_DIR=/home/renguanming/jupyterlab/rgm/beta-FFT-main/mechanism_strengthening_20261013
case "${1:-help}" in
  setup) "$TASK_PY" "$TASK_DIR/scripts/setup.py" ;;
  joint-smoke) "$TASK_PY" "$TASK_DIR/scripts/diagnose_joint.py" --smoke ;;
  trc-smoke) "$TASK_PY" "$TASK_DIR/scripts/diagnose_trc.py" --smoke ;;
  joint) "$TASK_PY" "$TASK_DIR/scripts/diagnose_joint.py" ;;
  trc) "$TASK_PY" "$TASK_DIR/scripts/diagnose_trc.py" ;;
  figures) "$TASK_PY" "$TASK_DIR/scripts/figures.py" ;;
  tables) "$TASK_PY" "$TASK_DIR/scripts/tables.py" ;;
  final-control-analysis) "$TASK_PY" "$TASK_DIR/scripts/final_control_analysis.py" ;;
  additional-checks) "$TASK_PY" "$TASK_DIR/scripts/additional_checks.py" ;;
  train) "$TASK_PY" "$TASK_DIR/scripts/train_fixed.py" ;;
  test) "$TASK_PY" "$TASK_DIR/scripts/evaluate_fixed.py" ;;
  report) "$TASK_PY" "$TASK_DIR/scripts/report.py" ;;
  audit) "$TASK_PY" "$TASK_DIR/scripts/audit.py" ;;
  pipeline) "$TASK_PY" "$TASK_DIR/scripts/pipeline.py" ;;
  status) "$TASK_PY" "$TASK_DIR/scripts/status.py" ;;
  *) echo 'Stages: setup joint-smoke trc-smoke joint trc figures tables additional-checks final-control-analysis train test report audit pipeline status. The fixed pipeline is already launched; inspect status before any training command.' ;;
esac
