# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable Benchmark Contract Tests
"""Behavioral and structural tests for differentiable benchmark result contracts."""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import scpn_quantum_control.benchmarks.differentiable_programming as benchmark_facade
import scpn_quantum_control.benchmarks.differentiable_programming_contracts as contracts
from scpn_quantum_control.benchmarks.differentiable_programming import (
    DifferentiableProgrammingBenchmarkResult,
    DifferentiableProgrammingExternalReferenceResult,
    QuantumGradientBenchmarkResult,
)


def test_differentiable_programming_benchmark_result_validation_paths() -> None:
    """Benchmark result metadata should reject malformed diagnostic rows."""

    valid_gradient = np.array([1.0, 2.0], dtype=np.float64)
    row = DifferentiableProgrammingBenchmarkResult(
        case_id="case",
        category="loop-heavy",
        value=1.0,
        gradient=valid_gradient,
        analytic_gradient=valid_gradient.copy(),
        max_abs_gradient_error=0.0,
        adjoint_supported=True,
        max_abs_adjoint_error=0.0,
        claim_boundary="diagnostic conformance only, not a performance timing claim",
    )

    assert row.passed is True
    blocked_row = DifferentiableProgrammingBenchmarkResult(
        case_id="blocked",
        category="rust-interpreter",
        value=1.0,
        gradient=valid_gradient,
        analytic_gradient=valid_gradient.copy(),
        max_abs_gradient_error=0.0,
        adjoint_supported=False,
        max_abs_adjoint_error=None,
        claim_boundary="diagnostic conformance only, not a performance timing claim",
        blocked_reasons=("optional native backend unavailable",),
    )
    assert blocked_row.passed is False
    with pytest.raises(ValueError, match="case_id"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="shapes"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=np.array([1.0], dtype=np.float64),
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="category"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="value"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=np.inf,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="gradient"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=np.array([[1.0, 2.0]], dtype=np.float64),
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=-1.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="adjoint_supported"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported="yes",  # type: ignore[arg-type]
            max_abs_adjoint_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="max_abs_adjoint_error"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=np.nan,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        DifferentiableProgrammingBenchmarkResult(
            case_id="case",
            category="loop-heavy",
            value=1.0,
            gradient=valid_gradient,
            analytic_gradient=valid_gradient,
            max_abs_gradient_error=0.0,
            adjoint_supported=True,
            max_abs_adjoint_error=0.0,
            claim_boundary="",
        )


def test_quantum_gradient_benchmark_result_validation_paths() -> None:
    """Quantum-gradient benchmark metadata should reject malformed rows."""

    gradient = np.array([1.0, 2.0], dtype=np.float64)
    row = QuantumGradientBenchmarkResult(
        case_id="case",
        category="quantum-gradient",
        value=1.0,
        parameter_shift_gradient=gradient,
        finite_difference_gradient=gradient.copy(),
        analytic_gradient=gradient.copy(),
        max_abs_reference_error=0.0,
        max_abs_finite_difference_error=0.0,
        verification_passed=True,
        evaluations=8,
        claim_boundary="diagnostic conformance only, no wall-clock performance claim",
    )

    assert row.passed is True
    with pytest.raises(ValueError, match="case_id"):
        QuantumGradientBenchmarkResult(
            case_id="",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient,
            analytic_gradient=gradient,
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="gradient shapes"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=np.array([1.0], dtype=np.float64),
            analytic_gradient=gradient,
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="reference error"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient,
            analytic_gradient=gradient,
            max_abs_reference_error=np.nan,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="verification_passed"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient,
            analytic_gradient=gradient,
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed="yes",  # type: ignore[arg-type]
            evaluations=8,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="evaluations"):
        QuantumGradientBenchmarkResult(
            case_id="case",
            category="quantum-gradient",
            value=1.0,
            parameter_shift_gradient=gradient,
            finite_difference_gradient=gradient,
            analytic_gradient=gradient,
            max_abs_reference_error=0.0,
            max_abs_finite_difference_error=0.0,
            verification_passed=True,
            evaluations=0,
            claim_boundary="diagnostic",
        )


def test_differentiable_programming_external_reference_result_validation_paths() -> None:
    """External-reference benchmark metadata should reject malformed rows."""

    gradient = np.array([1.0, 2.0], dtype=np.float64)
    row = DifferentiableProgrammingExternalReferenceResult(
        case_id="case",
        backend="jax",
        program_value=1.0,
        reference_value=1.0,
        program_gradient=gradient,
        reference_gradient=gradient.copy(),
        max_abs_value_error=0.0,
        max_abs_gradient_error=0.0,
        claim_boundary="diagnostic correctness only, not a performance claim",
    )

    assert row.passed is True
    with pytest.raises(ValueError, match="backend"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="gradient shapes"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=np.array([1.0], dtype=np.float64),
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="case_id"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="values"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=np.inf,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="program_gradient"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=np.array([1.0, np.nan], dtype=np.float64),
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="value error"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=-1.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="gradient error"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=np.nan,
            claim_boundary="diagnostic",
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        DifferentiableProgrammingExternalReferenceResult(
            case_id="case",
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="",
        )


def test_benchmark_contract_leaf_has_no_suite_backedge() -> None:
    """Keep result contracts independent of benchmark execution stacks."""
    tree = ast.parse(inspect.getsource(contracts))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        token in module
        for module in imported_modules
        for token in ("differentiable_programming", "differentiable", "phase", "compiler")
    )


def test_benchmark_contract_objects_are_exact_facade_aliases() -> None:
    """Preserve public records and private numeric helper identity."""
    names = (
        "DifferentiableProgrammingBenchmarkResult",
        "DifferentiableProgrammingExternalReferenceResult",
        "QuantumGradientBenchmarkResult",
        "_as_gradient",
        "_max_abs_error",
    )

    for name in names:
        assert getattr(benchmark_facade, name) is getattr(contracts, name)


def test_benchmark_facade_does_not_redefine_contracts() -> None:
    """Keep contract definitions single-owned by the contracts leaf."""
    tree = ast.parse(inspect.getsource(benchmark_facade))
    definitions = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }

    assert definitions.isdisjoint(
        {
            "DifferentiableProgrammingBenchmarkResult",
            "DifferentiableProgrammingExternalReferenceResult",
            "QuantumGradientBenchmarkResult",
            "_as_gradient",
            "_max_abs_error",
        }
    )


def test_benchmark_contract_leaf_exports_only_public_records() -> None:
    """Keep shared numeric helpers private to benchmark implementation modules."""
    assert set(contracts.__all__) == {
        "DifferentiableProgrammingBenchmarkResult",
        "DifferentiableProgrammingExternalReferenceResult",
        "QuantumGradientBenchmarkResult",
    }
