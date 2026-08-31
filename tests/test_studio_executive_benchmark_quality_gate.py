# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio executive benchmark quality-gate tests
"""Lock Studio benchmark-handler quality gates into preflight and CI."""

from tools import preflight
from tools import studio_executive_benchmark_quality_gates as quality_gates
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_static_gates_cover_typing_and_docs() -> None:
    """Require strict typing and NumPy docstrings for the owned cohort."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-studio-executive-benchmark-quality"][5:]
        == quality_gates.STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET
    )
    docs = gates["ruff D studio-executive-benchmark quality ratchet"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs
    assert "lint.explicit-preview-rules = true" in docs


def test_coverage_gate_is_isolated_and_exact() -> None:
    """Require branch execution and exact source-only coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    assert "--branch" in gates["studio-executive-benchmark focused coverage"]
    threshold = gates["studio-executive-benchmark exact coverage threshold"]
    assert "--fail-under=100" in threshold
    assert (
        "--include=*/studio/executive_benchmark.py,*/studio/benchmark_databank_bundle.py"
        in threshold
    )
    assert quality_gates.STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.STUDIO_EXECUTIVE_BENCHMARK_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command


def test_ci_runs_and_aggregates_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  studio-executive-benchmark-quality:")
    end = workflow.index("\n\n  synchronisation-witness-quality:", start)
    block = workflow[start:end]
    assert all(path in block for path in quality_gates.STUDIO_EXECUTIVE_BENCHMARK_QUALITY_RATCHET)
    assert "studio-executive-benchmark-quality" in workflow[workflow.index("  ci-gate:") :]
