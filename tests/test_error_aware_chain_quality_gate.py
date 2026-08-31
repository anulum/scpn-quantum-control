# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — error-aware chain quality-gate tests
"""Lock error-aware chain quality gates into preflight and CI."""

from tools import error_aware_chain_quality_gates as quality_gates
from tools import preflight
from tools.ci_workflow_inventory import read_ci_workflow_source


def test_helper_builds_strict_preview_and_exact_gates() -> None:
    """Require strict preview docs, the real suite, and exact source coverage."""
    static = dict(quality_gates.build_static_quality_gates("/python"))
    docs = static["ruff D error-aware-chain quality ratchet"]
    coverage = dict(quality_gates.build_coverage_gates("/python"))
    run = coverage["error-aware-chain focused coverage"]
    report = coverage["error-aware-chain exact coverage threshold"]
    assert "--preview" in docs
    assert "D,D413,D417,D420" in docs
    assert "lint.explicit-preview-rules = true" in docs
    assert run[-1:] == quality_gates.ERROR_AWARE_CHAIN_COVERAGE_COHORT
    assert "--fail-under=100" in report
    assert f"--include={quality_gates.ERROR_AWARE_CHAIN_COVERAGE_INCLUDE}" in report
    assert quality_gates.ERROR_AWARE_CHAIN_COVERAGE_DATA_FILE.startswith("/tmp/")


def test_preflight_uses_helper_commands_verbatim() -> None:
    """Keep helper-defined commands exact in preflight."""
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert dict(preflight.ERROR_AWARE_CHAIN_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )


def test_ci_runs_and_aggregates_error_aware_chain_quality() -> None:
    """Keep the dedicated job and transitive aggregate dependency required."""
    workflow = read_ci_workflow_source()
    start = workflow.index("  error-aware-chain-quality:")
    end = workflow.index("\n\n  kuramoto-variants-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.ERROR_AWARE_CHAIN_QUALITY_RATCHET:
        assert path in block
    for path in quality_gates.ERROR_AWARE_CHAIN_COVERAGE_COHORT:
        assert path in block
    assert quality_gates.ERROR_AWARE_CHAIN_COVERAGE_INCLUDE in block
    consumer_start = workflow.index("  kuramoto-variants-quality:", end)
    consumer_block = workflow[
        consumer_start : workflow.index("\n\n  mps-evolution-quality:", consumer_start)
    ]
    assert "needs: [lint, error-aware-chain-quality]" in consumer_block
