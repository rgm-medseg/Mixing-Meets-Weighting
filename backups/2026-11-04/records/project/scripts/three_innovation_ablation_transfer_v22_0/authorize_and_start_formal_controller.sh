#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/renguanming/jupyterlab/rgm/beta-FFT-main
AUDIT_ROOT="$ROOT/audits/three_innovation_ablation_transfer_v22_0"
SESSION=three_innovation_ablation_transfer_v22_0
PYTHON=/home/renguanming/.conda/envs/beta_fft/bin/python

if [[ "${1:-}" != "--yes-i-understand-start-formal-full" ]]; then
  echo "REFUSING_TO_START: pass --yes-i-understand-start-formal-full" >&2
  exit 2
fi

cd "$ROOT"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("/home/renguanming/jupyterlab/rgm/beta-FFT-main")
audit = root / "audits/three_innovation_ablation_transfer_v22_0"
status = json.loads((audit / "V22_0_CURRENT_PRELAUNCH_STATUS.json").read_text())
prestart = json.loads((audit / "V22_0_PRESTART_FINAL_CHECK.json").read_text())
if status.get("V22_0_PRELAUNCH_READY") is not True:
    raise SystemExit("PRELAUNCH_NOT_READY")
if status.get("FINAL_PRESTART_HARD_GATE", {}).get("HARD_GATE_PASS") is not True:
    raise SystemExit("FINAL_PRESTART_HARD_GATE_NOT_PASS")
if prestart.get("PRESTART_FINAL_CHECK_PASS") is not True:
    raise SystemExit("PRESTART_FINAL_CHECK_NOT_PASS")
if status.get("FORMAL_FULL_STARTED") is not False:
    raise SystemExit("FORMAL_ALREADY_STARTED_IN_STATUS")
if status.get("FORMAL_TMUX_CONTROLLER_STARTED") is not False:
    raise SystemExit("TMUX_ALREADY_STARTED_IN_STATUS")
if status.get("MSCMR10", {}).get("state") != "BLOCKED_PARENT_IDENTITY":
    raise SystemExit("MSCMR10_BLOCKER_STATE_CHANGED_REVIEW_REQUIRED")
if status.get("TRANSFER_REAL_TEST_ACCESS_BEFORE_FORMAL") != 0:
    raise SystemExit("TRANSFER_TEST_FIREWALL_NOT_ZERO")
PY

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "TMUX_SESSION_ALREADY_EXISTS:$SESSION" >&2
  exit 3
fi

"$PYTHON" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/renguanming/jupyterlab/rgm/beta-FFT-main")
audit = root / "audits/three_innovation_ablation_transfer_v22_0"
payload = {
    "audit_id": "V22_0_FORMAL_START_AUTHORIZATION",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "USER_AUTHORIZED_FORMAL_START": True,
    "FORMAL_START_SCOPE": "global-final NEW_FULL_READY tasks only",
    "EXACT_REUSE_POLICY": "VERIFY_EXACT_REUSE_NO_LAUNCH",
    "BLOCKED_POLICY": "MSCMR10_SKIP_BLOCKED_TASK",
    "REAL_TRANSFER_TEST_ACCESS_BEFORE_FORMAL": 0,
    "TMUX_SESSION": "three_innovation_ablation_transfer_v22_0",
}
(audit / "V22_0_FORMAL_START_AUTHORIZATION.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && '$PYTHON' controllers/three_innovation_ablation_transfer_controller_v22_0.py --mode run --scope global-final"

tmux ls
