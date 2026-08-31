# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — governed route-matrix quality-gate tests
"""Lock the governed route-matrix gate into preflight and CI."""

from tools import governed_route_matrix_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and isolated NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-governed-route-matrix-quality"][5:]
        == quality_gates.GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D governed-route-matrix quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["governed-route-matrix focused coverage"]
    assert "--fail-under=100" in gates["governed-route-matrix exact coverage threshold"]


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.GOVERNED_ROUTE_MATRIX_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  governed-route-matrix-quality:")
    end = workflow.index("\n\n  decisive-advantage-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.GOVERNED_ROUTE_MATRIX_QUALITY_RATCHET)
    assert "governed-route-matrix-quality" in workflow[workflow.index("  ci-gate:") :]
