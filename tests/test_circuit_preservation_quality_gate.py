# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit preservation quality integration
"""Real quality commands and the original mandatory CI category registration."""

from __future__ import annotations

import subprocess
import sys

from tools import circuit_preservation_quality_gates as gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_commands_validate_actual_native_owners() -> None:
    """Execute the real strict/doc processes over the exact promoted cohort."""
    for name, command in gates.build_static_quality_gates(sys.executable):
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
        assert result.returncode == 0, f"{name}: {result.stdout}{result.stderr}"


def test_ci_and_preflight_own_the_entire_circuit_qualification_cohort() -> None:
    """A missing owner or relaxed denominator fails the public workflow contract."""
    source = read_ci_workflow_source()
    start = source.index("  circuit-preservation-quality:")
    end = source.index("\n  scientific-problem-quality:", start)
    block = source[start:end]
    for path in (*gates.TYPING, *gates.TESTS):
        assert path in block
    assert gates.INCLUDE in block
    assert "--fail-under=100" in block and "--branch" in block
    assert "circuit-preservation-quality" in source[source.index("  ci-gate:") :]
    for name, command in gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert (
        gates.build_coverage_gates(preflight._PY, preflight.CIRCUIT_PRESERVATION_COVERAGE_DATA)
        == preflight.CIRCUIT_PRESERVATION_COVERAGE_GATES
    )
