# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto-variant quality-gate tests
"""Lock the Kuramoto-variant quality gate into preflight and CI."""

import ast
from pathlib import Path

from tools import kuramoto_variants_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-kuramoto-variants-quality"][5:]
        == quality_gates.KURAMOTO_VARIANTS_TYPING_RATCHET
    )
    ruff = gates["ruff D Kuramoto-variants quality ratchet"]
    assert (
        ruff[-len(quality_gates.KURAMOTO_VARIANTS_DOCSTRING_RATCHET) :]
        == quality_gates.KURAMOTO_VARIANTS_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417" in ruff


def test_mixed_consumer_keeps_variant_contracts_documented() -> None:
    """Own only the variant-contract class inside the mixed consumer."""
    tree = ast.parse(Path(quality_gates.KURAMOTO_VARIANTS_MIXED_CONSUMER).read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TestKuramotoVariantContracts"
    )
    assert ast.get_docstring(owner)
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)]
    assert methods
    assert all(ast.get_docstring(method) for method in methods)


def test_coverage_gate_is_isolated_connected_and_exact() -> None:
    """Require connected native/reference execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["Kuramoto-variants focused coverage"]
    report = gates["Kuramoto-variants exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.KURAMOTO_VARIANTS_COVERAGE_COHORT) :]
        == quality_gates.KURAMOTO_VARIANTS_COVERAGE_COHORT
    )
    assert any(argument.startswith("--data-file=/tmp/") for argument in run)
    assert "--fail-under=100" in report
    assert "--include=*/phase/kuramoto_variants.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.KURAMOTO_VARIANTS_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    source = Path("tools/preflight.py").read_text(encoding="utf-8")
    assert "gates.extend(KURAMOTO_VARIANTS_COVERAGE_GATES)" in source


def test_ci_runs_and_aggregates_kuramoto_variants_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  kuramoto-variants-quality:")
    end = workflow.index("\n\n  tn-mps-baseline-design-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.KURAMOTO_VARIANTS_DOCSTRING_RATCHET:
        assert path in block
    for path in quality_gates.KURAMOTO_VARIANTS_COVERAGE_COHORT:
        assert path in block
    assert "PyO3/maturin-action@" in block
    assert "--fail-under=100" in block
    assert "phase/kuramoto_variants.py" in block
    aggregate = workflow[workflow.index("  ci-gate:") :]
    assert "kuramoto-variants-quality" in aggregate
