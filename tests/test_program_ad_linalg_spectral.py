# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD spectral linalg tests
"""Tests for Program AD spectral linear-algebra derivatives and diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_linalg_eigvalsh_distinct_symmetric_gradient_matches_spectral_adjoint() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    values = np.array([2.0, 0.35, -0.2, 3.0], dtype=np.float64)
    weights = np.array([0.75, -1.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        raw = np.reshape(trace_values, (2, 2))
        matrix = 0.5 * (raw + raw.T)
        return np.sum(np.linalg.eigvalsh(matrix) * weights)

    result = whole_program_value_and_grad(objective, values)
    raw = values.reshape(2, 2)
    matrix = 0.5 * (raw + raw.T)
    _eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_adjoint = eigenvectors @ np.diag(weights) @ eigenvectors.T
    expected = (0.5 * (spectral_adjoint + spectral_adjoint.T)).reshape(-1)

    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_eigvalsh_direct_rule_returns_spectral_jvp_and_vjp() -> None:
    from scpn_quantum_control.differentiable import program_ad_linalg_eigvalsh_derivative_rule

    rule = program_ad_linalg_eigvalsh_derivative_rule((2, 2))
    matrix = np.array([[2.0, 0.35], [0.35, 3.0]], dtype=np.float64)
    tangent = np.array([[0.2, -0.1], [-0.1, 0.4]], dtype=np.float64)
    cotangent = np.array([1.25, -0.5], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    expected_jvp = np.array(
        [float(eigenvector.T @ tangent @ eigenvector) for eigenvector in eigenvectors.T],
        dtype=np.float64,
    )
    expected_vjp = (eigenvectors @ np.diag(cotangent) @ eigenvectors.T).reshape(-1)

    _assert_allclose(cast(Any, rule).value_fn(matrix.reshape(-1)), eigenvalues)
    _assert_allclose(
        cast(Any, rule).jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), expected_jvp
    )
    _assert_allclose(cast(Any, rule).vjp_rule(matrix.reshape(-1), cotangent), expected_vjp)


def test_program_ad_linalg_eigvalsh_fails_closed_on_nonsymmetric_or_degenerate_inputs() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    nonsymmetric = np.array([1.0, 0.25, -0.5, 2.0], dtype=np.float64)
    degenerate = np.eye(2, dtype=np.float64).reshape(-1)

    with pytest.raises(ValueError, match="symmetric matrix"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigvalsh(np.reshape(trace_values, (2, 2)))),
            nonsymmetric,
        )

    with pytest.raises(ValueError, match="distinct eigenvalues"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigvalsh(np.reshape(trace_values, (2, 2)))),
            degenerate,
        )


def test_program_ad_linalg_eigvalsh_registry_contract_and_root_export() -> None:
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        primitive_contract_for,
        program_ad_linalg_eigvalsh_derivative_rule,
    )

    assert (
        scpn.program_ad_linalg_eigvalsh_derivative_rule
        is program_ad_linalg_eigvalsh_derivative_rule
    )
    contract = primitive_contract_for("scpn.program_ad.linalg:eigvalsh")
    assert contract is not None
    assert cast(Any, contract).shape_rule((np.eye(3, dtype=np.float64),)) == (3,)
    assert cast(Any, contract).static_argument_rule((np.eye(3, dtype=np.float64),)) == ()
    metadata = contract.lowering_metadata
    assert metadata["static_derivative_factory"] == "program_ad_linalg_eigvalsh_derivative_rule"
    assert metadata["nondifferentiable_boundary"] == "symmetric_matrix_distinct_eigenvalues"


def test_program_ad_linalg_eigvalsh_reverse_adjoint_replay_matches_spectral_adjoint() -> None:
    from scpn_quantum_control.differentiable import (
        program_adjoint_gradient,
        whole_program_value_and_grad,
    )

    values = np.array([2.0, 0.35, -0.2, 3.0], dtype=np.float64)
    weights = np.array([0.75, -1.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        raw = np.reshape(trace_values, (2, 2))
        matrix = 0.5 * (raw + raw.T)
        return np.sum(np.linalg.eigvalsh(matrix) * weights)

    result = whole_program_value_and_grad(objective, values)
    raw = values.reshape(2, 2)
    matrix = 0.5 * (raw + raw.T)
    _eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    spectral_adjoint = eigenvectors @ np.diag(weights) @ eigenvectors.T
    expected = (0.5 * (spectral_adjoint + spectral_adjoint.T)).reshape(-1)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_svdvals_gradient_and_adjoint_match_singular_vector_adjoint() -> None:
    from scpn_quantum_control.differentiable import (
        program_adjoint_gradient,
        whole_program_value_and_grad,
    )

    values = np.array([2.0, 0.3, -0.2, 1.1], dtype=np.float64)
    weights = np.array([0.5, -1.3], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        return np.sum(np.linalg.svd(matrix, compute_uv=False) * weights)

    result = whole_program_value_and_grad(objective, values)
    matrix = values.reshape(2, 2)
    left, _singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)
    expected = (left @ np.diag(weights) @ right_h).reshape(-1)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_svdvals_direct_rule_returns_singular_value_jvp_and_vjp() -> None:
    from scpn_quantum_control.differentiable import program_ad_linalg_svdvals_derivative_rule

    rule = program_ad_linalg_svdvals_derivative_rule((2, 3))
    matrix = np.array([[2.0, 0.25, -0.1], [0.3, 1.4, 0.6]], dtype=np.float64)
    tangent = np.array([[0.2, -0.1, 0.05], [0.15, 0.4, -0.2]], dtype=np.float64)
    cotangent = np.array([1.25, -0.5], dtype=np.float64)
    left, singular_values, right_h = np.linalg.svd(matrix, full_matrices=False)

    expected_jvp = np.array(
        [float(left[:, index].T @ tangent @ right_h[index, :]) for index in range(2)],
        dtype=np.float64,
    )
    expected_vjp = (left @ np.diag(cotangent) @ right_h).reshape(-1)

    _assert_allclose(cast(Any, rule).value_fn(matrix.reshape(-1)), singular_values)
    _assert_allclose(
        cast(Any, rule).jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), expected_jvp
    )
    _assert_allclose(cast(Any, rule).vjp_rule(matrix.reshape(-1), cotangent), expected_vjp)


def test_program_ad_linalg_svdvals_fails_closed_on_vector_return_or_degenerate_values() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    values = np.array([2.0, 0.3, -0.2, 1.1], dtype=np.float64)
    repeated = np.eye(2, dtype=np.float64).reshape(-1)
    rank_deficient = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="compute_uv=False"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.svd(np.reshape(trace_values, (2, 2)))[1]),
            values,
        )

    with pytest.raises(ValueError, match="distinct positive singular values"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(
                np.linalg.svd(np.reshape(trace_values, (2, 2)), compute_uv=False)
            ),
            repeated,
        )

    with pytest.raises(ValueError, match="distinct positive singular values"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(
                np.linalg.svd(np.reshape(trace_values, (2, 2)), compute_uv=False)
            ),
            rank_deficient,
        )


def test_program_ad_linalg_svdvals_registry_contract_and_root_export() -> None:
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        primitive_contract_for,
        program_ad_linalg_svdvals_derivative_rule,
    )

    assert (
        scpn.program_ad_linalg_svdvals_derivative_rule is program_ad_linalg_svdvals_derivative_rule
    )
    contract = primitive_contract_for("scpn.program_ad.linalg:svd")
    assert contract is not None
    assert cast(Any, contract).shape_rule((np.zeros((2, 3), dtype=np.float64),)) == (2,)
    metadata = contract.lowering_metadata
    assert metadata["static_derivative_factory"] == "program_ad_linalg_svdvals_derivative_rule"
    assert metadata["nondifferentiable_boundary"] == "distinct_positive_singular_values"


def test_program_ad_linalg_pinv_gradient_and_adjoint_match_rank_constant_formula() -> None:
    from scpn_quantum_control.differentiable import (
        program_adjoint_gradient,
        whole_program_value_and_grad,
    )

    values = np.array([2.0, 0.3, -0.2, 1.4], dtype=np.float64)
    weights = np.array([[0.4, -0.6], [1.2, -0.3]], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        return np.sum(np.linalg.pinv(matrix) * weights)

    result = whole_program_value_and_grad(objective, values)
    matrix = values.reshape(2, 2)
    inverse = np.linalg.inv(matrix)
    expected = (-(inverse.T @ weights @ inverse.T)).reshape(-1)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_pinv_direct_rule_returns_jvp_and_vjp() -> None:
    from scpn_quantum_control.differentiable import program_ad_linalg_pinv_derivative_rule

    rule = program_ad_linalg_pinv_derivative_rule((2, 3))
    matrix = np.array([[2.0, 0.3, -0.2], [0.4, 1.5, 0.7]], dtype=np.float64)
    tangent = np.array([[0.2, -0.1, 0.05], [0.15, 0.4, -0.2]], dtype=np.float64)
    cotangent = np.array([[0.25, -0.5], [0.75, 0.1], [-0.2, 0.4]], dtype=np.float64)
    pinv = np.linalg.pinv(matrix)
    left_projector = np.eye(matrix.shape[1]) - pinv @ matrix
    right_projector = np.eye(matrix.shape[0]) - matrix @ pinv

    expected_jvp = (
        -pinv @ tangent @ pinv
        + pinv @ pinv.T @ tangent.T @ right_projector
        + left_projector @ tangent.T @ pinv.T @ pinv
    ).reshape(-1)
    expected_vjp = (
        -pinv.T @ cotangent @ pinv.T
        + right_projector @ cotangent.T @ pinv @ pinv.T
        + pinv.T @ pinv @ cotangent.T @ left_projector
    ).reshape(-1)

    _assert_allclose(cast(Any, rule).value_fn(matrix.reshape(-1)), pinv.reshape(-1))
    _assert_allclose(
        cast(Any, rule).jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), expected_jvp
    )
    _assert_allclose(
        cast(Any, rule).vjp_rule(matrix.reshape(-1), cotangent.reshape(-1)), expected_vjp
    )


def test_program_ad_linalg_pinv_fails_closed_on_rank_loss_vector_or_hermitian_mode() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    rank_deficient = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    vector = np.array([1.0, 2.0], dtype=np.float64)
    full_rank = np.array([2.0, 0.3, -0.2, 1.4], dtype=np.float64)

    with pytest.raises(ValueError, match="rank-2 matrix"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.pinv(trace_values)), vector
        )

    with pytest.raises(ValueError, match="constant full-rank matrix"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.pinv(np.reshape(trace_values, (2, 2)))),
            rank_deficient,
        )

    with pytest.raises(ValueError, match="hermitian=False"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(
                np.linalg.pinv(np.reshape(trace_values, (2, 2)), hermitian=True)
            ),
            full_rank,
        )


def test_program_ad_linalg_pinv_registry_contract_and_root_export() -> None:
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        primitive_contract_for,
        program_ad_linalg_pinv_derivative_rule,
    )

    assert scpn.program_ad_linalg_pinv_derivative_rule is program_ad_linalg_pinv_derivative_rule
    contract = primitive_contract_for("scpn.program_ad.linalg:pinv")
    assert contract is not None
    assert cast(Any, contract).shape_rule((np.zeros((2, 3), dtype=np.float64),)) == (3, 2)
    metadata = contract.lowering_metadata
    assert metadata["static_derivative_factory"] == "program_ad_linalg_pinv_derivative_rule"
    assert metadata["nondifferentiable_boundary"] == "rank_threshold_crossing"


def _expected_symmetric_eigh_adjoint(
    matrix: NDArray[np.float64],
    eigenvalue_cotangent: NDArray[np.float64],
    eigenvector_cotangent: NDArray[np.float64],
) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    adjoint = eigenvectors @ np.diag(eigenvalue_cotangent) @ eigenvectors.T
    for column in range(eigenvectors.shape[1]):
        cotangent_column = eigenvector_cotangent[:, column]
        for other in range(eigenvectors.shape[1]):
            if other == column:
                continue
            scale = float(eigenvectors[:, other].T @ cotangent_column) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            adjoint = adjoint + scale * np.outer(eigenvectors[:, other], eigenvectors[:, column])
    return np.asarray(0.5 * (adjoint + adjoint.T), dtype=np.float64)


def test_program_ad_linalg_eigh_gradient_and_adjoint_include_eigenvectors() -> None:
    from scpn_quantum_control.differentiable import (
        program_adjoint_gradient,
        whole_program_value_and_grad,
    )

    values = np.array([2.0, 0.35, -0.2, 3.0], dtype=np.float64)
    eigenvalue_weights = np.array([0.75, -1.25], dtype=np.float64)
    eigenvector_weights = np.array([[0.2, -0.4], [0.6, 0.1]], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        raw = np.reshape(trace_values, (2, 2))
        matrix = 0.5 * (raw + raw.T)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return np.sum(eigenvalues * eigenvalue_weights) + np.sum(
            eigenvectors * eigenvector_weights
        )

    result = whole_program_value_and_grad(objective, values)
    raw = values.reshape(2, 2)
    matrix = 0.5 * (raw + raw.T)
    matrix_adjoint = _expected_symmetric_eigh_adjoint(
        matrix, eigenvalue_weights, eigenvector_weights
    )
    expected = (0.5 * (matrix_adjoint + matrix_adjoint.T)).reshape(-1)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_eigh_direct_rule_returns_value_jvp_and_vjp() -> None:
    from scpn_quantum_control.differentiable import program_ad_linalg_eigh_derivative_rule

    rule = program_ad_linalg_eigh_derivative_rule((2, 2))
    matrix = np.array([[2.0, 0.35], [0.35, 3.0]], dtype=np.float64)
    tangent = np.array([[0.2, -0.1], [-0.1, 0.4]], dtype=np.float64)
    eigenvalue_cotangent = np.array([1.25, -0.5], dtype=np.float64)
    eigenvector_cotangent = np.array([[0.3, -0.2], [0.5, 0.1]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    expected_eigenvalue_jvp = np.array(
        [float(vector.T @ tangent @ vector) for vector in eigenvectors.T], dtype=np.float64
    )
    expected_eigenvector_jvp = np.zeros_like(eigenvectors)
    for column in range(eigenvectors.shape[1]):
        for other in range(eigenvectors.shape[1]):
            if other == column:
                continue
            scale = float(eigenvectors[:, other].T @ tangent @ eigenvectors[:, column]) / float(
                eigenvalues[column] - eigenvalues[other]
            )
            expected_eigenvector_jvp[:, column] += scale * eigenvectors[:, other]
    expected_value = np.concatenate((eigenvalues, eigenvectors.reshape(-1)))
    expected_jvp = np.concatenate((expected_eigenvalue_jvp, expected_eigenvector_jvp.reshape(-1)))
    expected_vjp = _expected_symmetric_eigh_adjoint(
        matrix, eigenvalue_cotangent, eigenvector_cotangent
    ).reshape(-1)
    cotangent = np.concatenate((eigenvalue_cotangent, eigenvector_cotangent.reshape(-1)))

    _assert_allclose(cast(Any, rule).value_fn(matrix.reshape(-1)), expected_value)
    _assert_allclose(
        cast(Any, rule).jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), expected_jvp
    )
    _assert_allclose(cast(Any, rule).vjp_rule(matrix.reshape(-1), cotangent), expected_vjp)


def test_program_ad_linalg_eigh_fails_closed_on_nonsymmetric_or_degenerate_inputs() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    nonsymmetric = np.array([1.0, 0.25, -0.5, 2.0], dtype=np.float64)
    degenerate = np.eye(2, dtype=np.float64).reshape(-1)
    valid = np.array([2.0, 0.35, 0.35, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="symmetric matrix"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigh(np.reshape(trace_values, (2, 2)))[0]),
            nonsymmetric,
        )

    with pytest.raises(ValueError, match="distinct eigenvalues"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigh(np.reshape(trace_values, (2, 2)))[0]),
            degenerate,
        )

    with pytest.raises(ValueError, match="UPLO"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(
                np.linalg.eigh(np.reshape(trace_values, (2, 2)), UPLO=cast(Any, "X"))[0]
            ),
            valid,
        )


def test_program_ad_linalg_eigh_registry_contract_and_root_export() -> None:
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        primitive_contract_for,
        program_ad_linalg_eigh_derivative_rule,
    )

    assert scpn.program_ad_linalg_eigh_derivative_rule is program_ad_linalg_eigh_derivative_rule
    contract = primitive_contract_for("scpn.program_ad.linalg:eigh")
    assert contract is not None
    assert cast(Any, contract).shape_rule((np.eye(3, dtype=np.float64),)) == (3, 3, 3)
    metadata = contract.lowering_metadata
    assert metadata["static_derivative_factory"] == "program_ad_linalg_eigh_derivative_rule"
    assert metadata["nondifferentiable_boundary"] == "symmetric_matrix_distinct_eigenvalues"


def _expected_real_simple_eigvals_adjoint(
    matrix: NDArray[np.float64], cotangent: NDArray[np.float64]
) -> NDArray[np.float64]:
    eigenvalues, right_eigenvectors = np.linalg.eig(matrix)
    assert np.max(np.abs(eigenvalues.imag)) <= 1.0e-12
    assert np.max(np.abs(right_eigenvectors.imag)) <= 1.0e-12
    right = right_eigenvectors.real
    left_rows = np.linalg.inv(right)
    adjoint = np.zeros_like(matrix, dtype=np.float64)
    for index, weight in enumerate(cotangent):
        adjoint += float(weight) * np.outer(left_rows[index, :], right[:, index])
    return adjoint


def test_program_ad_linalg_eigvals_gradient_and_adjoint_for_real_simple_spectrum() -> None:
    from scpn_quantum_control.differentiable import (
        program_adjoint_gradient,
        whole_program_value_and_grad,
    )

    values = np.array([2.0, 0.4, 0.15, 3.0], dtype=np.float64)
    weights = np.array([0.75, -1.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        matrix = np.reshape(trace_values, (2, 2))
        return np.sum(np.linalg.eigvals(matrix) * weights)

    result = whole_program_value_and_grad(objective, values)
    matrix = values.reshape(2, 2)
    expected = _expected_real_simple_eigvals_adjoint(matrix, weights).reshape(-1)

    assert result.adjoint_result is not None
    assert result.adjoint_result.supported
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def test_program_ad_linalg_eigvals_direct_rule_returns_value_jvp_and_vjp() -> None:
    from scpn_quantum_control.differentiable import program_ad_linalg_eigvals_derivative_rule

    rule = program_ad_linalg_eigvals_derivative_rule((2, 2))
    matrix = np.array([[2.0, 0.4], [0.15, 3.0]], dtype=np.float64)
    tangent = np.array([[0.2, -0.1], [0.35, 0.4]], dtype=np.float64)
    cotangent = np.array([1.25, -0.5], dtype=np.float64)
    eigenvalues, right_eigenvectors = np.linalg.eig(matrix)
    right = right_eigenvectors.real
    left_rows = np.linalg.inv(right)

    expected_jvp = np.array(
        [float(left_rows[index, :] @ tangent @ right[:, index]) for index in range(2)],
        dtype=np.float64,
    )
    expected_vjp = _expected_real_simple_eigvals_adjoint(matrix, cotangent).reshape(-1)

    _assert_allclose(cast(Any, rule).value_fn(matrix.reshape(-1)), eigenvalues.real)
    _assert_allclose(
        cast(Any, rule).jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), expected_jvp
    )
    _assert_allclose(cast(Any, rule).vjp_rule(matrix.reshape(-1), cotangent), expected_vjp)


def test_program_ad_linalg_eigvals_fails_closed_on_complex_or_degenerate_spectrum() -> None:
    from scpn_quantum_control.differentiable import whole_program_value_and_grad

    complex_spectrum = np.array([0.0, -1.0, 1.0, 0.0], dtype=np.float64)
    degenerate = np.eye(2, dtype=np.float64).reshape(-1)

    with pytest.raises(ValueError, match="real eigenvalues"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigvals(np.reshape(trace_values, (2, 2)))),
            complex_spectrum,
        )

    with pytest.raises(ValueError, match="distinct eigenvalues"):
        whole_program_value_and_grad(
            lambda trace_values: np.sum(np.linalg.eigvals(np.reshape(trace_values, (2, 2)))),
            degenerate,
        )


def test_program_ad_linalg_eigvals_registry_contract_and_root_export() -> None:
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        primitive_contract_for,
        program_ad_linalg_eigvals_derivative_rule,
    )

    assert (
        scpn.program_ad_linalg_eigvals_derivative_rule is program_ad_linalg_eigvals_derivative_rule
    )
    contract = primitive_contract_for("scpn.program_ad.linalg:eigvals")
    assert contract is not None
    assert cast(Any, contract).shape_rule((np.eye(3, dtype=np.float64),)) == (3,)
    metadata = contract.lowering_metadata
    assert metadata["static_derivative_factory"] == "program_ad_linalg_eigvals_derivative_rule"
    assert metadata["nondifferentiable_boundary"] == "real_simple_diagonalizable_spectrum"


def test_program_ad_linalg_conditioning_diagnostics_cover_norm_svd_solve_and_rank_boundary() -> (
    None
):
    import scpn_quantum_control as scpn
    from scpn_quantum_control.differentiable import (
        ProgramADLinalgConditioningDiagnostic,
        diagnose_program_ad_linalg_conditioning,
        primitive_contract_for,
    )

    well_conditioned = diagnose_program_ad_linalg_conditioning(
        "svd", np.array([[3.0, 0.25], [0.1, 1.5]], dtype=np.float64)
    )
    assert isinstance(well_conditioned, ProgramADLinalgConditioningDiagnostic)
    assert well_conditioned.primitive == "svd"
    assert well_conditioned.shape == (2, 2)
    assert well_conditioned.rank == 2
    assert well_conditioned.status == "well_conditioned"
    assert well_conditioned.differentiability_ready
    assert well_conditioned.smallest_scale > 0.0
    assert well_conditioned.condition_number < 10.0
    assert "distinct positive singular values" in well_conditioned.required_boundary
    assert well_conditioned.as_dict()["condition_number"] == well_conditioned.condition_number

    ill_conditioned = diagnose_program_ad_linalg_conditioning(
        "solve",
        np.array([[1.0, 0.0], [0.0, 1.0e-10]], dtype=np.float64),
        condition_threshold=1.0e8,
    )
    assert ill_conditioned.status == "ill_conditioned"
    assert ill_conditioned.differentiability_ready
    assert ill_conditioned.condition_number >= 1.0e10
    assert "ill-conditioned" in ill_conditioned.message

    rank_boundary = diagnose_program_ad_linalg_conditioning(
        "pinv", np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    )
    assert rank_boundary.status == "rank_deficient"
    assert not rank_boundary.differentiability_ready
    assert rank_boundary.rank == 1
    assert rank_boundary.smallest_scale == 0.0
    assert "rank threshold" in rank_boundary.required_boundary

    vector_norm = diagnose_program_ad_linalg_conditioning("norm", np.array([3.0, 4.0]))
    assert vector_norm.status == "well_conditioned"
    assert vector_norm.condition_number == 1.0
    assert vector_norm.differentiability_ready

    zero_norm = diagnose_program_ad_linalg_conditioning("norm", np.zeros(3, dtype=np.float64))
    assert zero_norm.status == "zero_norm_boundary"
    assert not zero_norm.differentiability_ready
    assert "zero norm" in zero_norm.message

    with pytest.raises(ValueError, match="unsupported"):
        diagnose_program_ad_linalg_conditioning("qr", np.eye(2, dtype=np.float64))

    assert scpn.diagnose_program_ad_linalg_conditioning is diagnose_program_ad_linalg_conditioning
    contract = primitive_contract_for("scpn.program_ad.linalg:svd")
    assert contract is not None
    assert contract.lowering_metadata["conditioning_diagnostic"] == (
        "diagnose_program_ad_linalg_conditioning"
    )
