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
    run_differentiable_programming_benchmark_suite,
)


def test_differentiable_programming_benchmark_suite_matches_analytic_references() -> None:
    """Benchmark rows should compare implemented program AD against analytic references."""

    results = run_differentiable_programming_benchmark_suite()

    assert [row.case_id for row in results] == [
        "loop_heavy_scalar",
        "matrix_heavy_linear_algebra",
        "linalg_primitive_contracts",
        "mutation_heavy_forward_only",
        "transform_nesting_vmap_program_grad",
    ]
    assert {row.category for row in results} == {
        "loop-heavy",
        "matrix-heavy",
        "linalg-primitive",
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


def test_differentiable_programming_benchmark_exports_from_package_root() -> None:
    """Benchmark suite API should be stable from the package root."""

    import scpn_quantum_control as scpn

    assert (
        scpn.DifferentiableProgrammingBenchmarkResult is DifferentiableProgrammingBenchmarkResult
    )
    assert (
        scpn.run_differentiable_programming_benchmark_suite
        is run_differentiable_programming_benchmark_suite
    )
