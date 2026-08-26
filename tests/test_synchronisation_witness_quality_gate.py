# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation witness quality-gate tests
"""Lock synchronisation-witness quality gates into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import synchronisation_witness_quality_gates as quality_gates


def test_static_gates_cover_typing_and_docs() -> None:
    """Require strict typing and NumPy docstrings for the owned cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-synchronisation-witness-quality"][5:]
        == quality_gates.SYNCHRONISATION_WITNESS_QUALITY_RATCHET
    )
    assert "D,D413" in gates["ruff D synchronisation-witness quality ratchet"]


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["synchronisation-witness focused coverage"]
    threshold = gates["synchronisation-witness exact coverage threshold"]
    assert "--fail-under=100" in threshold
    assert any("sync_witness_evidence.py" in argument for argument in threshold)


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.SYNCHRONISATION_WITNESS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text()
    start = workflow.index("  synchronisation-witness-quality:")
    end = workflow.index("\n\n  experiment-mitigation-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.SYNCHRONISATION_WITNESS_QUALITY_RATCHET)
    assert "synchronisation-witness-quality" in workflow[workflow.index("  ci-gate:") :]
