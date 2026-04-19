#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — mutmut runner for analysis/otoc.py
#
# Runs only the otoc-focused test files. See docs/mutation_testing.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-$(command -v python)}"
exec "$VENV_PY" -m pytest -x -q \
    "$REPO_ROOT/tests/test_otoc.py" \
    "$REPO_ROOT/tests/test_otoc_mutation_kills.py" \
    "$REPO_ROOT/tests/test_otoc_sync_probe.py"
