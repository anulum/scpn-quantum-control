# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel evidence quality-gate tests
"""Lock the deterministic evidence gate into preflight and CI."""

from pathlib import Path

from tools import preflight
from tools import topology_kernel_evidence_quality_gates as quality_gates


def test_static_gate_is_strict_and_numpy_documented() -> None:
    """Require strict typing and complete owner NumPy docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-topology-kernel-evidence-quality"][5:]
        == quality_gates.TOPOLOGY_KERNEL_EVIDENCE_TYPING_RATCHET
    )
    ruff = gates["ruff D topology-kernel-evidence quality ratchet"]
    assert (
        ruff[-len(quality_gates.TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET) :]
        == quality_gates.TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET
    )
    assert "--isolated" in ruff and "D,D413" in ruff


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["topology-kernel-evidence focused coverage"]
    report = gates["topology-kernel-evidence exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT) :]
        == quality_gates.TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/topology_kernel_product/evidence.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert (
        "gates.extend(TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_GATES)"
        in Path("tools/preflight.py").read_text()
    )


def test_ci_runs_and_aggregates_topology_kernel_evidence_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  topology-kernel-evidence-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT:
        assert path in block
    assert "--fail-under=100" in block
    assert "topology_kernel_product/evidence.py" in block
    assert "topology-kernel-evidence-quality" in workflow[workflow.index("  ci-gate:") :]
