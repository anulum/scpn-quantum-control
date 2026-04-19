#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — mutmut runner for bridge/knm_hamiltonian.py
#
# Runs only the knm_hamiltonian-focused test files so each mutant
# evaluates in a few seconds instead of a full-suite 25-minute round.
# See docs/mutation_testing.md for the mutation-testing policy.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-$(command -v python)}"
exec "$VENV_PY" -m pytest -x -q \
    "$REPO_ROOT/tests/test_knm_hamiltonian.py" \
    "$REPO_ROOT/tests/test_knm_hamiltonian_mutation_kills.py" \
    "$REPO_ROOT/tests/test_knm_parity.py" \
    "$REPO_ROOT/tests/test_knm_properties.py"
