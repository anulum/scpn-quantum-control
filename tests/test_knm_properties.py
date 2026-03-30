# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Knm Properties
"""Property-based tests for Knm Hamiltonian compiler."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.bridge.knm_hamiltonian import (
    knm_to_ansatz,
    knm_to_hamiltonian,
)


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=10, deadline=30000)
def test_hamiltonian_hermitian(n: int) -> None:
    """H must be Hermitian for any qubit count."""
    rng = np.random.default_rng(n)
    K = rng.uniform(0, 0.5, (n, n))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 0)
    omega = rng.uniform(0.5, 3.0, n)

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-12)


@given(n=st.integers(min_value=2, max_value=6))
@settings(max_examples=10, deadline=30000)
def test_hamiltonian_real_eigenvalues(n: int) -> None:
    """Hermitian H must have real eigenvalues."""
    rng = np.random.default_rng(n + 100)
    K = rng.uniform(0, 0.5, (n, n))
    K = (K + K.T) / 2
    np.fill_diagonal(K, 0)
    omega = rng.uniform(0.5, 3.0, n)

    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    eigvals = np.linalg.eigvalsh(mat if isinstance(mat, np.ndarray) else mat.toarray())
    assert np.all(np.isreal(eigvals))


@given(
    n=st.integers(min_value=2, max_value=6),
    reps=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=10, deadline=10000)
def test_ansatz_parameter_count(n: int, reps: int) -> None:
    """Ansatz must have exactly n * 2 * reps parameters."""
    K = np.ones((n, n)) * 0.1
    np.fill_diagonal(K, 0)
    qc = knm_to_ansatz(K, reps=reps)
    assert qc.num_parameters == n * 2 * reps
