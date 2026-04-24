#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Frontier Campaign Orchestrator

set -e

echo "==========================================================="
echo " SCPN Quantum Control — Frontier Campaign 2026 (Batch 4)   "
echo " Target: IBM Heron r2 (ibm_fez / ibm_kingston)             "
echo "==========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../src:${SCRIPT_DIR}/../sophisticated_campaign_2026:${PYTHONPATH}"

cd "${SCRIPT_DIR}"
mkdir -p results
python3 generate_params.py

TESTS=(
    "test_quantum_advantage_scaling.py"
    "test_live_scneurocore_loop.py"
    "test_sync_distillation.py"
    "test_multi_backend_distributed.py"
    "test_dla_tensor_network.py"
    "test_rl_pulse_optimization.py"
    "test_pt_symmetric_kuramoto.py"
    "test_logical_sync_protection.py"
)

for test_script in "${TESTS[@]}"; do
    echo "-----------------------------------------------------------"
    echo "Starting: ${test_script}"
    echo "-----------------------------------------------------------"
    if [ -f "${test_script}" ]; then
        python3 "${test_script}"
        echo "[OK] ${test_script} completed."
    else
        echo "[ERROR] ${test_script} not found in ${SCRIPT_DIR}"
        exit 1
    fi
done

echo "==========================================================="
echo " Frontier Campaign complete. Results saved in results/"
echo "==========================================================="
