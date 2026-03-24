#!/usr/bin/env bash
# WSL Ubuntu 24.04 bootstrap for scpn-quantum-control
# Run from Windows: wsl -d Ubuntu-24.04 -- bash /mnt/c/aaa_God_of_the_Math_Collection/03_CODE/scpn-quantum-control/scripts/wsl_bootstrap.sh
set -euo pipefail

REPO="/mnt/c/aaa_God_of_the_Math_Collection/03_CODE/scpn-quantum-control"
VENV="$REPO/.venv-wsl"

echo "=== scpn-quantum-control WSL bootstrap ==="
echo "Repo: $REPO"
echo ""

# ── 1. System packages ──────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip \
    build-essential pkg-config libhdf5-dev git curl > /dev/null 2>&1
echo "  done."

# ── 2. Create venv ──────────────────────────────────────────────────
echo "[2/6] Creating virtualenv at $VENV..."
if [ -d "$VENV" ]; then
    echo "  exists, skipping."
else
    python3.12 -m venv "$VENV"
    echo "  created."
fi
source "$VENV/bin/activate"

# ── 3. Upgrade pip, install project ─────────────────────────────────
echo "[3/6] Installing scpn-quantum-control[dev,ibm,viz]..."
pip install --upgrade pip setuptools wheel -q
cd "$REPO"
pip install -e ".[dev,ibm,viz]" -q
echo "  done. $(pip show scpn-quantum-control 2>/dev/null | grep Version)"

# ── 4. Verify critical imports ──────────────────────────────────────
echo "[4/6] Verifying imports..."
python3.12 -c "
import qiskit; print(f'  qiskit        {qiskit.__version__}')
import qiskit_aer; print(f'  qiskit-aer    {qiskit_aer.__version__}')
import qiskit_ibm_runtime; print(f'  qiskit-ibm-rt {qiskit_ibm_runtime.__version__}')
import numpy; print(f'  numpy         {numpy.__version__}')
import scipy; print(f'  scipy         {scipy.__version__}')
import networkx; print(f'  networkx      {networkx.__version__}')
from scpn_quantum_control.hardware.runner import HardwareRunner; print('  HardwareRunner OK')
from scpn_quantum_control.hardware.experiments import noise_baseline_experiment; print('  experiments   OK')
"

# ── 5. Run tests (quick subset) ────────────────────────────────────
echo "[5/6] Running quick test suite..."
cd "$REPO"
python3.12 -m pytest tests/ -x -q --tb=short -m "not slow and not hardware" 2>&1 | tail -5

# ── 6. Test IBM auth ────────────────────────────────────────────────
echo "[6/6] Testing IBM Quantum auth..."
if [ -z "${SCPN_IBM_TOKEN:-}" ]; then
    echo "  SCPN_IBM_TOKEN not set — skipping auth test."
    echo "  To set: export SCPN_IBM_TOKEN='your_token'"
else
    python3.12 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
import os
svc = QiskitRuntimeService(
    channel='ibm_cloud',
    token=os.environ['SCPN_IBM_TOKEN'],
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/78db885720334fd19191b33a839d0c35:841cc36d-0afd-4f96-ada2-8c56e1c443a0::',
)
backends = svc.backends()
print(f'  Auth OK — {len(backends)} backends: {[b.name for b in backends]}')
" 2>&1 || echo "  Auth failed — check token."
fi

echo ""
echo "=== Bootstrap complete ==="
echo "Activate: source $VENV/bin/activate"
echo "Run campaign: SCPN_IBM_TOKEN=\$TOKEN python scripts/march_2026_hardware_campaign.py --dry-run"
echo "Check jobs:   SCPN_IBM_TOKEN=\$TOKEN python scripts/retrieve_completed_jobs.py --status-only"
