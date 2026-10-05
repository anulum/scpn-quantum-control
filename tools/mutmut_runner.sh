#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — mutmut test runner
#
# mutmut 2.5 splits `--runner` on whitespace without shell=True, so
# a multi-word runner command must be wrapped in a script. This
# script is the canonical runner for the `analysis/koopman.py`
# baseline; extend as target files grow. See docs/mutation_testing.md.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="${VENV_PY:-$(command -v python)}"
# The mutation gate sets VENV_PY to the interpreter that runs it; a direct
# call falls back to the python on PATH.
cd "$REPO_ROOT"
exec "$VENV_PY" -m pytest -x -q tests/test_koopman.py
