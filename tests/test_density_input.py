# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — physical density admission tests
"""Exercise physical input admission, exact copying and preallocation bounds."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase.density_input import validate_density_matrix


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_density_admission_preserves_values_and_owns_its_copy(
    dtype: type[np.float64] | type[np.complex128],
) -> None:
    """A physical mixed array retains its values without sharing caller storage."""
    original = cast(
        NDArray[np.float64] | NDArray[np.complex128],
        np.diag([0.1, 0.2, 0.3, 0.4]).astype(dtype),
    )
    original.setflags(write=False)
    matrix = validate_density_matrix(original, 2)
    assert matrix.dtype == np.complex128
    np.testing.assert_array_equal(matrix, original)
    assert not np.shares_memory(matrix, original)
    matrix[0, 0] = 0
    assert original[0, 0] == 0.1


def test_complex_coherence_and_noncontiguous_input_are_lossless() -> None:
    """A complex pure state is admitted without discarding phase or transposing basis."""
    state = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    backing = np.zeros((4, 4), dtype=np.complex128)
    backing[::2, ::2] = rho
    view = backing[::2, ::2]
    result = validate_density_matrix(view, 1)
    np.testing.assert_array_equal(result, rho)
    assert not np.shares_memory(result, view)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ([[0.5, 0], [0, 0.5]], "shape"),
        (np.array([0.5, 0.5]), "shape"),
        (np.eye(3) / 3, "shape"),
        (np.eye(2, dtype=np.int64), "dtype"),
        (np.eye(2, dtype=np.float32), "dtype"),
        (np.eye(2, dtype=np.complex64), "dtype"),
        (np.eye(2, dtype=np.bool_), "dtype"),
        (np.array([["0.5", "0"], ["0", "0.5"]]), "dtype"),
        (np.array([[np.nan, 0], [0, 1.0]]), "finite"),
        (np.array([[np.inf, 0], [0, 1.0]]), "finite"),
        (np.array([[0.5, 0.2j], [0.2j, 0.5]]), "Hermitian"),
        (np.diag([0.2, 0.3]), "unit trace"),
        (np.diag([1.1, -0.1]), "positive semidefinite"),
    ],
)
def test_invalid_density_inputs_refuse(value: object, error: str) -> None:
    """Malformed and nonphysical matrices fail through the public admission function."""
    with pytest.raises(ValueError, match=error):
        validate_density_matrix(cast(NDArray[np.complex128], value), 1)


@pytest.mark.parametrize("negative", [0.0, 0.5e-10, 1e-10])
def test_accepted_roundoff_is_preserved_without_psd_projection(negative: float) -> None:
    """The declared PSD tolerance admits unchanged roundoff at its inclusive boundary."""
    rho = np.diag([1 + negative, -negative])
    result = validate_density_matrix(rho, 1)
    np.testing.assert_array_equal(result, rho)
    assert result[1, 1] == -negative


def test_negative_eigenvalue_outside_declared_roundoff_refuses() -> None:
    """An eigenvalue just outside the fixed bound cannot be silently repaired."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_density_matrix(np.diag([1 + 1.01e-10, -1.01e-10]), 1)


def test_budget_precedes_shape_inspection_and_exponential_copy() -> None:
    """A tiny allowance refuses a huge declared matrix before touching an input array."""
    with pytest.raises(DenseAllocationError, match="density input validation"):
        validate_density_matrix(np.eye(2) / 2, 32, max_dense_gib=0.001)
    with pytest.raises(DenseAllocationError, match="density input validation"):
        validate_density_matrix(np.eye(2) / 2, 1, max_dense_gib=1e-12)


def test_trace_and_hermiticity_roundoff_are_not_normalized() -> None:
    """Admitted trace and conjugacy roundoff remain literal caller values."""
    rho = np.array([[0.5 + 1e-11, 0.1 + 1e-11j], [0.1, 0.5]], dtype=np.complex128)
    result = validate_density_matrix(rho, 1)
    np.testing.assert_array_equal(result, rho)
    assert np.trace(result) != 1
