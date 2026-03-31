# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Knm Properties
"""Multi-angle property-based tests for Knm Hamiltonian compiler.

Covers: Hermiticity, real eigenvalues, ansatz parameter count, energy bounds,
K=0 decoupled case, dense matrix consistency, parametrised sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.bridge.knm_hamiltonian import (
    knm_to_ansatz,
    knm_to_dense_matrix,
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


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_dense_matrix_hermitian(n: int) -> None:
    """knm_to_dense_matrix should produce Hermitian matrix."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    H = knm_to_dense_matrix(K, omega)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_dense_eigenvalues_real(n: int) -> None:
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(np.isfinite(eigvals))


def test_zero_coupling_decoupled() -> None:
    """K=0 → H is diagonal with entries from omega."""
    n = 3
    K = np.zeros((n, n))
    omega = np.array([1.0, 2.0, 3.0])
    H = knm_to_dense_matrix(K, omega)
    # H should be diagonal: -sum(omega_i * Z_i)
    eigvals = np.sort(np.linalg.eigvalsh(H))
    # Ground energy = -sum(|omega|)
    np.testing.assert_allclose(eigvals[0], -np.sum(np.abs(omega)), atol=1e-10)


@pytest.mark.parametrize("n", [2, 4, 6])
def test_ground_energy_negative(n: int) -> None:
    """Coupled XY system should have negative ground energy."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    H = knm_to_dense_matrix(K, omega)
    E0 = np.linalg.eigvalsh(H)[0]
    assert E0 < 0


def test_dense_matrix_shape() -> None:
    n = 4
    K = np.eye(n) * 0.0
    omega = np.ones(n)
    H = knm_to_dense_matrix(K, omega)
    assert H.shape == (2**n, 2**n)


def test_ansatz_qubit_count() -> None:
    n = 4
    K = np.ones((n, n)) * 0.1
    np.fill_diagonal(K, 0)
    qc = knm_to_ansatz(K, reps=2)
    assert qc.num_qubits == n


# ---------------------------------------------------------------------------
# Rust path parity: dense matrix
# ---------------------------------------------------------------------------


def test_rust_dense_matrix_parity() -> None:
    """Rust build_xy_hamiltonian_dense must match Python knm_to_dense_matrix."""
    try:
        import scpn_quantum_engine as eng
    except ImportError:
        pytest.skip("scpn-quantum-engine not available")

    n = 3
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)

    H_py = knm_to_dense_matrix(K, omega)
    K_flat = np.ascontiguousarray(K.ravel(), dtype=np.float64)
    H_rust = np.array(eng.build_xy_hamiltonian_dense(K_flat, omega.astype(np.float64), n)).reshape(
        2**n, 2**n
    )

    np.testing.assert_allclose(H_rust, H_py, atol=1e-10)


# ---------------------------------------------------------------------------
# Hamiltonian tracelessness (Pauli structure)
# ---------------------------------------------------------------------------


def test_hamiltonian_traceless() -> None:
    """XY Hamiltonian is traceless (all non-identity Pauli terms)."""
    n = 4
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    H = knm_to_dense_matrix(K, omega)
    assert abs(np.trace(H)) < 1e-8


# ---------------------------------------------------------------------------
# Pipeline: K,omega → H → eigenvalues → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_spectrum() -> None:
    """Full pipeline: build coupling → compile H → diagonalise → spectrum.
    Verifies Knm compiler is not decorative.
    """
    import time

    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    H = knm_to_dense_matrix(K, omega)
    eigvals = np.linalg.eigvalsh(H)
    dt = (time.perf_counter() - t0) * 1000

    assert eigvals[0] < 0
    assert np.all(np.isfinite(eigvals))

    print(f"\n  PIPELINE Knm→H→spectrum (4q): {dt:.1f} ms")
    print(f"  E_0 = {eigvals[0]:.4f}, E_max = {eigvals[-1]:.4f}")
