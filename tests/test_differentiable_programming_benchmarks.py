# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark tests
"""Tests for differentiable-programming conformance benchmark surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import (
    DifferentiableProgrammingBenchmarkResult,
    DifferentiableProgrammingExternalReferenceResult,
    run_differentiable_programming_benchmark_suite,
    run_differentiable_programming_external_reference_suite,
)
from scpn_quantum_control.benchmarks import differentiable_programming as dp_benchmarks


def test_differentiable_programming_benchmark_suite_matches_analytic_references() -> None:
    """Benchmark rows should compare implemented program AD against analytic references."""

    results = run_differentiable_programming_benchmark_suite()

    assert [row.case_id for row in results] == [
        "loop_heavy_scalar",
        "matrix_heavy_linear_algebra",
        "selection_piecewise_contracts",
        "linalg_primitive_contracts",
        "indexing_static_gather_contracts",
        "mutation_heavy_forward_only",
        "transform_nesting_vmap_program_grad",
    ]
    assert {row.category for row in results} == {
        "loop-heavy",
        "matrix-heavy",
        "selection-heavy",
        "linalg-primitive",
        "indexing-heavy",
        "mutation-heavy",
        "transform-nesting",
    }
    for row in results:
        assert isinstance(row, DifferentiableProgrammingBenchmarkResult)
        assert row.passed is True
        assert row.max_abs_gradient_error <= 1.0e-12
        assert row.gradient.shape == row.analytic_gradient.shape
        assert (
            "no wall-clock performance" in row.claim_boundary
            or "not a performance" in row.claim_boundary
        )
    mutation_row = next(row for row in results if row.category == "mutation-heavy")
    assert mutation_row.adjoint_supported is True
    assert mutation_row.max_abs_adjoint_error is not None
    assert mutation_row.max_abs_adjoint_error <= 1.0e-12
    assert np.any(mutation_row.gradient != 0.0)
    linalg_row = next(row for row in results if row.category == "linalg-primitive")
    assert linalg_row.adjoint_supported is True
    assert linalg_row.max_abs_adjoint_error is not None
    assert linalg_row.max_abs_adjoint_error <= 1.0e-12
    selection_row = next(row for row in results if row.category == "selection-heavy")
    assert selection_row.adjoint_supported is True
    assert selection_row.max_abs_adjoint_error is not None
    assert selection_row.max_abs_adjoint_error <= 1.0e-12
    indexing_row = next(row for row in results if row.category == "indexing-heavy")
    assert indexing_row.adjoint_supported is True
    assert indexing_row.max_abs_adjoint_error is not None
    assert indexing_row.max_abs_adjoint_error <= 1.0e-12


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


def test_differentiable_programming_external_reference_suite_fails_closed_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External-reference rows should be optional and explicit about backend availability."""

    monkeypatch.setattr(dp_benchmarks, "is_jax_autodiff_available", lambda: False)

    assert run_differentiable_programming_external_reference_suite() == ()


def test_differentiable_programming_external_reference_suite_uses_all_jax_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External-reference suite should include every optional JAX conformance row."""

    gradient = np.array([1.0], dtype=np.float64)

    def row(case_id: str) -> DifferentiableProgrammingExternalReferenceResult:
        return DifferentiableProgrammingExternalReferenceResult(
            case_id=case_id,
            backend="jax",
            program_value=1.0,
            reference_value=1.0,
            program_gradient=gradient,
            reference_gradient=gradient,
            max_abs_value_error=0.0,
            max_abs_gradient_error=0.0,
            claim_boundary="diagnostic correctness only, not a performance claim",
        )

    monkeypatch.setattr(dp_benchmarks, "is_jax_autodiff_available", lambda: True)
    monkeypatch.setattr(dp_benchmarks, "_jax_loop_heavy_case", lambda: row("loop"))
    monkeypatch.setattr(dp_benchmarks, "_jax_linalg_primitive_case", lambda: row("linalg"))
    monkeypatch.setattr(dp_benchmarks, "_jax_transform_nesting_case", lambda: row("transform"))

    results = run_differentiable_programming_external_reference_suite()

    assert [result.case_id for result in results] == ["loop", "linalg", "transform"]
    assert all(result.passed for result in results)


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


def test_differentiable_programming_benchmark_exports_from_package_root() -> None:
    """Benchmark suite API should be stable from the package root."""

    import scpn_quantum_control as scpn

    assert (
        scpn.DifferentiableProgrammingBenchmarkResult is DifferentiableProgrammingBenchmarkResult
    )
    assert (
        scpn.DifferentiableProgrammingExternalReferenceResult
        is DifferentiableProgrammingExternalReferenceResult
    )
    assert (
        scpn.run_differentiable_programming_benchmark_suite
        is run_differentiable_programming_benchmark_suite
    )
    assert (
        scpn.run_differentiable_programming_external_reference_suite
        is run_differentiable_programming_external_reference_suite
    )
