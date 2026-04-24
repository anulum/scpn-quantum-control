#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Primary Hardware Campaign Orchestrator

set -e

echo "==========================================================="
echo " SCPN Quantum Control — Primary Hardware Campaign 2026     "
echo " Target: IBM Heron r2 (ibm_fez / ibm_kingston)             "
echo " 12 Core Tests Validating DLA Parity, BKT, and FIM         "
echo "==========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../src:${PYTHONPATH}"

cd "${SCRIPT_DIR}"
mkdir -p results
python3 generate_params.py

TESTS=(
    "test_noise_channel_tomography.py"
    "test_cross_device_asymmetry.py"
    "test_initial_state_randomization.py"
    "test_bkt_scaling.py"
    "test_dtc_survival_sweep.py"
    "test_classical_quantum_otoc.py"
    "test_large_n_fim.py"
    "test_fim_redundancy.py"
    "test_biological_fim_connectome.py"
    "test_coherence_wall_scaling.py"
    "test_vqe_informed_scaling.py"
    "test_shot_noise_robustness.py"
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
echo " Primary Campaign complete. Results saved in results/"
echo "==========================================================="
