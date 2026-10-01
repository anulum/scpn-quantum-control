# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Scientific problem quality integration tests
"""Exercise real quality commands and their required owning CI integration."""

from __future__ import annotations

import subprocess
import sys

from tools import preflight
from tools import scientific_problem_quality_gates as gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_scientific_static_commands_execute_real_owned_files() -> None:
    """Real strict/type documentation processes validate the actual owner cohort."""
    for name, command in gates.build_static_quality_gates(sys.executable):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, f"{name}: {completed.stdout}{completed.stderr}"


def test_scientific_ci_and_preflight_require_the_owning_cohort() -> None:
    """Scientific inputs remain in their category and mandatory aggregate route."""
    source = read_ci_workflow_source()
    start = source.index("  scientific-problem-quality:")
    end = source.index("\n  qpu-compute-types-quality:", start)
    block = source[start:end]
    for path in (*gates.SCIENTIFIC_PROBLEM_TYPING, *gates.SCIENTIFIC_PROBLEM_TESTS):
        assert path in block
    assert gates.SCIENTIFIC_PROBLEM_INCLUDE in block
    assert "--fail-under=100" in block and "--branch" in block
    aggregate = source[source.index("  ci-gate:") :]
    assert "scientific-problem-quality" in aggregate
    for name, command in gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert gates.build_coverage_gates(preflight._PY) == preflight.SCIENTIFIC_PROBLEM_COVERAGE_GATES
