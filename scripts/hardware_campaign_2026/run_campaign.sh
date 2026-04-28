#!/usr/bin/env bash

# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hardware Campaign Orchestrator

set -e

echo "==========================================================="
echo " SCPN Quantum Control — IBM Hardware Campaign 2026         "
echo " Target: IBM Heron r2 (ibm_fez / ibm_kingston)             "
echo " 180 Minutes Credit Budget Active                          "
echo "==========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../src:${PYTHONPATH}"

# Ensure results directory exists
mkdir -p "${SCRIPT_DIR}/results"
cd "${SCRIPT_DIR}"
if [ ! -f "params/PARAMETER_PROVENANCE.json" ]; then
    echo "[ERROR] Missing params/PARAMETER_PROVENANCE.json."
    echo "Generate a validated parameter cache before launching hardware jobs."
    echo "For interface smoke tests only: python3 generate_params.py --allow-synthetic --seed 42"
    exit 1
fi

TESTS=(
    "test_fusion_feedback.py"
    "test_tipping_point_warning.py"
    "test_biological_fim.py"
    "test_rl_discovery.py"
    "test_distributed_clock.py"
    "test_quantum_thermo.py"
    "test_hypergraph_kuramoto.py"
    "test_sync_metrology.py"
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
echo " Campaign complete. All results saved in scripts/hardware_campaign_2026/results/"
echo "==========================================================="
