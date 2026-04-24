#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sophisticated Campaign Orchestrator

set -e

echo "==========================================================="
echo " SCPN Quantum Control — Sophisticated Campaign 2026        "
echo " Target: IBM Heron r2 (ibm_fez / ibm_kingston)             "
echo "==========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../src:${SCRIPT_DIR}/../hardware_campaign_2026:${PYTHONPATH}"

cd "${SCRIPT_DIR}"
mkdir -p results
python3 generate_params.py

TESTS=(
    "test_fusion_hybrid_stabilizer.py"
    "test_brain_scale_bridging.py"
    "test_sync_resource_theory.py"
    "test_autonomous_discovery.py"
    "test_quantum_internet_timing.py"
    "test_collective_thermo_engines.py"
    "test_hypergraph_nonreciprocal.py"
    "test_logical_sync_encoding.py"
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
echo " Sophisticated Campaign complete. Results saved in results/"
echo "==========================================================="
