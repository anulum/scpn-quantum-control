# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- differentiable programming benchmark tests
"""Tests for differentiable-programming conformance benchmark surfaces."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scpn_quantum_control.benchmarks import (
    DifferentiableProgrammingBenchmarkResult,
    DifferentiableProgrammingExternalReferenceResult,
    QuantumGradientBenchmarkResult,
    run_differentiable_programming_benchmark_suite,
    run_differentiable_programming_external_reference_suite,
    run_quantum_gradient_benchmark_suite,
)
from scpn_quantum_control.benchmarks import differentiable_programming as dp_benchmarks


def test_differentiable_programming_benchmark_suite_matches_analytic_references() -> None:
    """Benchmark rows should compare implemented program AD against analytic references."""

    results = run_differentiable_programming_benchmark_suite()

    assert [row.case_id for row in results] == [
        "loop_heavy_scalar",
        "python_semantics_list_comprehension",
        "elementwise_boundary_contracts",
        "matrix_heavy_linear_algebra",
        "selection_piecewise_contracts",
        "structured_numeric_primitive_contracts",
        "linalg_primitive_contracts",
        "indexing_static_gather_contracts",
        "mutation_heavy_forward_only",
        "shape_view_alias_metadata_contracts",
        "slice_mutation_alias_metadata_contracts",
        "loop_carried_state_alias_metadata_contracts",
        "transform_nesting_vmap_program_grad",
        "transform_nesting_custom_rule_vmap_jvp_vjp",
        "transform_nesting_program_ad_vmap_jvp_vjp",
        "transform_nesting_whole_program_higher_order",
        "transform_nesting_program_ad_hessian",
    ]
    assert {row.category for row in results} == {
        "loop-heavy",
        "python-semantics",
        "elementwise-boundary",
        "matrix-heavy",
        "selection-heavy",
        "structured-numeric",
        "linalg-primitive",
        "indexing-heavy",
        "mutation-heavy",
        "alias-effect",
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
    alias_rows = tuple(row for row in results if row.category == "alias-effect")
    assert len(alias_rows) == 3
    for alias_row in alias_rows:
        assert alias_row.adjoint_supported is True
        assert alias_row.max_abs_adjoint_error is not None
        assert alias_row.max_abs_adjoint_error <= 1.0e-12
        assert "metadata_only_no_general_alias_lattice" in alias_row.claim_boundary
    assert any("shape-view alias metadata conformance" in row.claim_boundary for row in alias_rows)
    assert any("slice-mutation alias/effect metadata" in row.claim_boundary for row in alias_rows)
    assert any("loop-carried state alias metadata" in row.claim_boundary for row in alias_rows)
    python_semantics_row = next(row for row in results if row.category == "python-semantics")
    assert python_semantics_row.adjoint_supported is True
    assert python_semantics_row.max_abs_adjoint_error is not None
    assert python_semantics_row.max_abs_adjoint_error <= 1.0e-12
    assert "plain list-comprehension" in python_semantics_row.claim_boundary
    assert "filtered, set, and dict comprehensions fail closed" in (
        python_semantics_row.claim_boundary
    )
    linalg_row = next(row for row in results if row.category == "linalg-primitive")
    assert linalg_row.adjoint_supported is True
    assert linalg_row.max_abs_adjoint_error is not None
    assert linalg_row.max_abs_adjoint_error <= 1.0e-12
    elementwise_row = next(row for row in results if row.category == "elementwise-boundary")
    assert elementwise_row.adjoint_supported is True
    assert elementwise_row.max_abs_adjoint_error is not None
    assert elementwise_row.max_abs_adjoint_error <= 1.0e-12
    assert "zero-cusp absolute-value" in elementwise_row.claim_boundary
    selection_row = next(row for row in results if row.category == "selection-heavy")
    assert selection_row.adjoint_supported is True
    assert selection_row.max_abs_adjoint_error is not None
    assert selection_row.max_abs_adjoint_error <= 1.0e-12
    structured_row = next(row for row in results if row.category == "structured-numeric")
    assert structured_row.adjoint_supported is True
    assert structured_row.max_abs_adjoint_error is not None
    assert structured_row.max_abs_adjoint_error <= 1.0e-12
    assert "product, interpolation, signal, and stencil primitive contracts" in (
        structured_row.claim_boundary
    )
    indexing_row = next(row for row in results if row.category == "indexing-heavy")
    assert indexing_row.adjoint_supported is True
    assert indexing_row.max_abs_adjoint_error is not None
    assert indexing_row.max_abs_adjoint_error <= 1.0e-12
    transform_row = next(row for row in results if row.category == "transform-nesting")
    assert transform_row.gradient.shape == (12,)
    assert "grad(vmap(f))" in transform_row.claim_boundary
    custom_rule_row = next(
        row for row in results if row.case_id == "transform_nesting_custom_rule_vmap_jvp_vjp"
    )
    assert custom_rule_row.gradient.shape == (12,)
    assert custom_rule_row.adjoint_supported is True
    assert custom_rule_row.max_abs_adjoint_error == 0.0
    assert "exact custom JVP/VJP" in custom_rule_row.claim_boundary
    program_jvp_vjp_row = next(
        row for row in results if row.case_id == "transform_nesting_program_ad_vmap_jvp_vjp"
    )
    assert program_jvp_vjp_row.gradient.shape == (8,)
    assert program_jvp_vjp_row.max_abs_gradient_error <= 1.0e-8
    assert "jvp/vjp over vmap of whole-program AD gradients" in program_jvp_vjp_row.claim_boundary
    higher_order_row = next(
        row for row in results if row.case_id == "transform_nesting_whole_program_higher_order"
    )
    assert higher_order_row.gradient.shape == (32,)
    assert higher_order_row.max_abs_gradient_error <= 1.0e-6
    assert "jacfwd/jacrev over whole-program grad(vmap(f))" in higher_order_row.claim_boundary
    program_hessian_row = next(
        row for row in results if row.case_id == "transform_nesting_program_ad_hessian"
    )
    assert program_hessian_row.gradient.shape == (4,)
    assert program_hessian_row.max_abs_gradient_error <= 1.0e-6
    assert "hessian over a whole-program AD scalar objective" in (
        program_hessian_row.claim_boundary
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


def test_quantum_gradient_benchmark_suite_matches_analytic_references() -> None:
    """Quantum-gradient rows should expose parameter-shift verification evidence."""

    results = run_quantum_gradient_benchmark_suite()

    assert [row.case_id for row in results] == [
        "single_rotation_parameter_shift",
        "two_parameter_phase_expectation",
        "sparse_ising_chain_six_qubit_expectation",
    ]
    for row in results:
        assert isinstance(row, QuantumGradientBenchmarkResult)
        assert row.category == "quantum-gradient"
        assert row.passed
        assert row.verification_passed
        assert row.evaluations > 0
        assert row.max_abs_reference_error <= 1.0e-12
        assert row.max_abs_finite_difference_error <= 1.0e-5
        assert "no wall-clock performance" in row.claim_boundary
        assert "hardware" in row.claim_boundary
        np.testing.assert_allclose(
            row.parameter_shift_gradient,
            row.analytic_gradient,
            atol=1.0e-12,
        )
    sparse_row = next(row for row in results if row.case_id.startswith("sparse_ising"))
    assert sparse_row.parameter_shift_gradient.shape == (6,)
    assert sparse_row.evaluations >= 24
    assert "sparse Hamiltonian" in sparse_row.claim_boundary


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


def test_differentiable_programming_jax_reference_rows_use_contract_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional JAX rows should construct explicit diagnostic records under backend shims."""

    fake_jnp = ModuleType("jax.numpy")
    fake_jnp.asarray = np.asarray
    fake_jnp.diag = np.diag
    fake_jnp.sin = np.sin
    fake_jnp.sum = np.sum
    fake_jnp.real = np.real
    fake_jnp.linalg = np.linalg

    def fake_vmap(function):
        return lambda values: np.asarray([function(row) for row in values], dtype=np.float64)

    def fake_grad(function):
        def gradient(row):
            _ = function(row)
            row_array = np.asarray(row, dtype=np.float64)
            return np.array([2.0 * row_array[0], np.cos(row_array[1])], dtype=np.float64)

        return gradient

    fake_jax = ModuleType("jax")
    fake_jax.config = SimpleNamespace(update=lambda *_args, **_kwargs: None)
    fake_jax.grad = fake_grad
    fake_jax.vmap = fake_vmap
    fake_jax.numpy = fake_jnp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)

    def fake_value_and_grad(objective, values):
        objective(values)
        return 1.0, np.ones(np.asarray(values, dtype=np.float64).size, dtype=np.float64)

    def fake_program_value_and_grad(objective, values, **_kwargs):
        objective(values)
        return SimpleNamespace(
            value=1.0,
            gradient=np.ones(np.asarray(values, dtype=np.float64).size, dtype=np.float64),
        )

    monkeypatch.setattr(dp_benchmarks, "jax_value_and_grad", fake_value_and_grad)
    monkeypatch.setattr(
        dp_benchmarks,
        "whole_program_value_and_grad",
        fake_program_value_and_grad,
    )

    loop_row = dp_benchmarks._jax_loop_heavy_case()
    linalg_row = dp_benchmarks._jax_linalg_primitive_case()
    transform_row = dp_benchmarks._jax_transform_nesting_case()

    assert [row.case_id for row in (loop_row, linalg_row, transform_row)] == [
        "jax_loop_heavy_reference",
        "jax_linalg_primitive_reference",
        "jax_transform_nesting_reference",
    ]
    assert all(row.backend == "jax" for row in (loop_row, linalg_row, transform_row))
    assert loop_row.passed
    assert linalg_row.passed
    assert transform_row.program_gradient.shape == transform_row.reference_gradient.shape


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
    assert scpn.QuantumGradientBenchmarkResult is QuantumGradientBenchmarkResult
    assert scpn.run_quantum_gradient_benchmark_suite is run_quantum_gradient_benchmark_suite
    assert dp_benchmarks.__all__ == [
        "DifferentiableProgrammingBenchmarkResult",
        "DifferentiableProgrammingExternalReferenceResult",
        "QuantumGradientBenchmarkResult",
        "run_differentiable_programming_benchmark_suite",
        "run_differentiable_programming_external_reference_suite",
        "run_quantum_gradient_benchmark_suite",
    ]
