# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dense budget guard tests
"""Tests for dense Hilbert-space allocation guards."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
from scpn_quantum_control.dense_budget import (
    DenseAllocationError,
    dense_object_bytes,
    estimate_dense_allocation,
    hilbert_dimension,
    require_dense_allocation,
    require_dense_eigensolver_workspace,
)


def test_dense_budget_estimates_complex_hamiltonian_bytes() -> None:
    estimate = estimate_dense_allocation(
        4,
        dtype=np.complex128,
        rank=2,
        max_gib=1.0,
        label="test Hamiltonian",
    )

    assert estimate.dimension == 16
    assert estimate.shape == (16, 16)
    assert estimate.bytes_required == 16 * 16 * np.dtype(np.complex128).itemsize
    assert estimate.budget_gib == pytest.approx(1.0)


def test_dense_budget_rejects_invalid_qubit_count() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        hilbert_dimension(0)


def test_dense_budget_fails_before_large_allocation() -> None:
    with pytest.raises(DenseAllocationError, match="dense XY Hamiltonian"):
        require_dense_allocation(
            16,
            dtype=np.complex128,
            rank=2,
            max_gib=1.0,
            label="dense XY Hamiltonian",
        )


def test_dense_budget_allows_small_allocation() -> None:
    estimate = require_dense_allocation(
        4,
        dtype=np.complex128,
        rank=2,
        max_gib=1.0,
        label="small Hamiltonian",
    )

    assert estimate.bytes_required == dense_object_bytes(4, dtype=np.complex128, rank=2)


def test_knm_dense_matrix_fails_closed_before_rust_or_qiskit_allocation() -> None:
    n = 16
    K = np.zeros((n, n), dtype=float)
    omega = np.ones(n, dtype=float)

    with pytest.raises(DenseAllocationError, match="sparse, sector, tensor-network"):
        knm_to_dense_matrix(K, omega, max_dense_gib=1.0)


def test_dense_eigensolver_workspace_uses_conservative_multiplier() -> None:
    with pytest.raises(DenseAllocationError, match="test dense eigensolver"):
        require_dense_eigensolver_workspace(
            4,
            max_gib=1e-5,
            label="test dense eigensolver",
        )
