# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native whole program linalg factorisation tests
# scpn-quantum-control -- native whole-program linalg factorisation tests
"""Focused native whole-program AD tests for 7x7 linalg factorisation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from scpn_quantum_control.compiler import (
    analyse_whole_program_ad_native_lowering,
    compile_whole_program_ad_trace_to_native_llvm_jit,
    native_whole_program_ad_linalg_support,
)
from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_value_and_grad,
    whole_program_value_and_grad,
)

FloatVector = npt.NDArray[np.float64]


def _factorisation_matrix_values(size: int, *, shift: float) -> FloatVector:
    """Return a deterministic diagonally dominant dense matrix vector."""

    matrix_2d = np.arange(1.0, size * size + 1.0, dtype=np.float64).reshape(size, size)
    matrix_2d /= float(size * size)
    matrix_2d += np.eye(size, dtype=np.float64) * (float(size) + 1.5 + shift)
    return np.asarray(matrix_2d.reshape(-1), dtype=np.float64)


def _factorisation_vector_solve_values(size: int, *, shift: float) -> FloatVector:
    """Return flattened matrix and vector-RHS values for native solve tests."""

    matrix = _factorisation_matrix_values(size, shift=shift)
    rhs = np.linspace(0.25 + shift, 1.05 + shift, size, dtype=np.float64)
    return np.concatenate([matrix, rhs])


def _factorisation_matrix_solve_values(size: int, rhs_cols: int, *, shift: float) -> FloatVector:
    """Return flattened matrix and matrix-RHS values for native solve tests."""

    matrix = _factorisation_matrix_values(size, shift=shift)
    rhs = np.linspace(
        0.2 + shift,
        1.25 + shift,
        size * rhs_cols,
        dtype=np.float64,
    ).reshape(size, rhs_cols)
    return np.concatenate([matrix, rhs.reshape(-1)])


def test_native_linalg_support_metadata_lifts_7x7_factorisation_boundary() -> None:
    """Support metadata should advertise bounded 7x7 quotient-linalg lowering."""

    support = native_whole_program_ad_linalg_support()

    assert support["inverse_sizes"] == (2, 3, 4, 5, 6, 7)
    assert support["solve_sizes"] == (2, 3, 4, 5, 6, 7)
    assert support["solve_matrix_sizes"] == (2, 3, 4, 5, 6, 7)
    assert support["inverse_fail_closed_from"] == 8
    assert support["solve_fail_closed_from"] == 8
    assert support["quotient_linalg_helper_sizes"] == (5, 6, 7)
    assert support["quotient_linalg_unsuitable_from"] == 8
    assert support["quotient_linalg_reuse_policy"] == "shared_factorisation_per_static_matrix"


def test_native_llvm_jit_lowers_7x7_full_output_inverse_trace() -> None:
    """The public native compiler should execute 7x7 full-output inverse traces."""

    size = 7
    weights = np.linspace(0.04, 0.92, size * size, dtype=np.float64).reshape(size, size)

    def objective(values: FloatVector) -> object:
        matrix = values[: size * size].reshape(size, size)
        inverse = np.linalg.inv(matrix)
        weighted_inverse = sum(
            weights[row, col] * inverse[row, col] for row in range(size) for col in range(size)
        )
        return weighted_inverse + 0.01 * values[0] * values[-1] - np.cos(inverse[-1, 0])

    sample = _factorisation_matrix_values(size, shift=0.0)
    replay = _factorisation_matrix_values(size, shift=0.3)
    parameters = tuple(Parameter(f"inv7_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )
    expected_ops = {f"linalg:inv:7x7:{row}:{col}" for row in range(size) for col in range(size)}

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert expected_ops.issubset(report.lowerable_ops)
    assert expected_ops.issubset(kernel.supported_ops)
    assert "inv7_shared_" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-8, abs=1.0e-8)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_native_llvm_jit_lowers_7x7_full_output_vector_solve_trace() -> None:
    """The public native compiler should execute 7x7 vector solve traces."""

    size = 7
    weights = np.linspace(0.1, 0.7, size, dtype=np.float64)

    def objective(values: FloatVector) -> object:
        matrix_end = size * size
        matrix = values[:matrix_end].reshape(size, size)
        rhs = values[matrix_end : matrix_end + size]
        solution = np.linalg.solve(matrix, rhs)
        weighted_solution = sum(weights[row] * solution[row] for row in range(size))
        return weighted_solution + 0.02 * values[0] * values[-1] - np.sin(solution[-1])

    sample = _factorisation_vector_solve_values(size, shift=0.0)
    replay = _factorisation_vector_solve_values(size, shift=0.25)
    parameters = tuple(Parameter(f"solve7_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )
    expected_ops = {f"linalg:solve:7x7:rhs:7:{row}" for row in range(size)}

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert expected_ops.issubset(report.lowerable_ops)
    assert expected_ops.issubset(kernel.supported_ops)
    assert "solve7_shared_" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-8, abs=1.0e-8)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_native_llvm_jit_lowers_7x7_full_output_matrix_solve_trace() -> None:
    """The public native compiler should execute 7x7 matrix-RHS solve traces."""

    size = 7
    rhs_cols = 2
    weights = np.linspace(0.05, 0.85, size * rhs_cols, dtype=np.float64).reshape(
        size,
        rhs_cols,
    )

    def objective(values: FloatVector) -> object:
        matrix_end = size * size
        matrix = values[:matrix_end].reshape(size, size)
        rhs = values[matrix_end : matrix_end + size * rhs_cols].reshape(size, rhs_cols)
        solution = np.linalg.solve(matrix, rhs)
        weighted_solution = sum(
            weights[row, col] * solution[row, col]
            for row in range(size)
            for col in range(rhs_cols)
        )
        return weighted_solution + 0.02 * values[0] * values[-1] - np.sin(solution[-1, -1])

    sample = _factorisation_matrix_solve_values(size, rhs_cols, shift=0.0)
    replay = _factorisation_matrix_solve_values(size, rhs_cols, shift=0.2)
    parameters = tuple(Parameter(f"solve7m2_x{index}") for index in range(sample.size))

    result = whole_program_value_and_grad(objective, sample, parameters)
    report = analyse_whole_program_ad_native_lowering(result)
    kernel = compile_whole_program_ad_trace_to_native_llvm_jit(objective, sample, parameters)
    reference_value, reference_gradient = program_adjoint_value_and_grad(
        objective,
        replay,
        parameters,
    )
    expected_ops = {
        f"linalg:solve:7x7:rhs:7x{rhs_cols}:{row}:{col}"
        for row in range(size)
        for col in range(rhs_cols)
    }

    assert report.supported is True
    assert report.unsupported_ops == ()
    assert expected_ops.issubset(report.lowerable_ops)
    assert expected_ops.issubset(kernel.supported_ops)
    assert "solve7m2_shared_" in kernel.llvm_ir
    assert kernel.value(replay) == pytest.approx(reference_value, rel=1.0e-8, abs=1.0e-8)
    np.testing.assert_allclose(
        kernel.gradient(replay),
        reference_gradient,
        rtol=1.0e-7,
        atol=1.0e-7,
    )
