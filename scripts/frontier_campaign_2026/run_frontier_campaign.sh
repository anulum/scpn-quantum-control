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
python3 run_frontier_campaign.py

echo "==========================================================="
echo " Frontier Campaign complete. Results saved in results/"
echo "==========================================================="
