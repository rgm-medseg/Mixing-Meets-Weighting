#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/renguanming/jupyterlab/rgm/beta-FFT-main"
PYTHON="/home/renguanming/.conda/envs/beta_fft/bin/python"
SESSION="three_innovation_ablation_transfer_v22_0"
CONTROLLER="controllers/three_innovation_ablation_transfer_controller_v22_0.py"
RUNTIME_ROOT="$ROOT/runtime/three_innovation_ablation_transfer_v22_0"
LOG_ROOT="$RUNTIME_ROOT/reboot_resume"
LOG="$LOG_ROOT/bootstrap.log"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

{
  date -Is
  echo "V22_0_REBOOT_RESUME_BOOTSTRAP"
  echo "cwd=$ROOT"
  echo "session=$SESSION"

  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "SESSION_ALREADY_EXISTS_NO_ACTION"
    exit 0
  fi

  if [[ ! -f "$RUNTIME_ROOT/controller_formal_ledger.tsv" ]]; then
    echo "FORMAL_LEDGER_MISSING_FAIL_CLOSED"
    exit 2
  fi

  if [[ ! -f "$ROOT/audits/three_innovation_ablation_transfer_v22_0/V22_0_MASTER_TASK_REGISTRY_FINAL.tsv" ]]; then
    echo "FINAL_REGISTRY_MISSING_FAIL_CLOSED"
    exit 2
  fi

  if [[ ! -f "$CONTROLLER" ]]; then
    echo "CONTROLLER_MISSING_FAIL_CLOSED"
    exit 2
  fi

  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && '$PYTHON' '$CONTROLLER' --mode resume --scope global-final"
  echo "RESUME_TMUX_SESSION_STARTED"
} >> "$LOG" 2>&1
