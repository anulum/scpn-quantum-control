# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Rust New Functions
"""Tests for new Rust engine functions: sparse Hamiltonian, magnetisation labels,
order parameter from statevector."""

from __future__ import annotations

import numpy as np
import pytest

eng = pytest.importorskip("scpn_quantum_engine")


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


# =====================================================================
# build_sparse_xy_hamiltonian
# =====================================================================
class TestRustSparseHamiltonian:
    def test_returns_triplets(self):
        K, omega = _system(4)
        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 4)
        assert len(rows) == len(cols) == len(vals)
        assert len(vals) > 0

    def test_matches_dense_rust(self):
        K, omega = _system(4)
        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 4)
        from scipy import sparse

        H_sparse = sparse.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(16, 16),
        ).toarray()

        H_dense = np.array(eng.build_xy_hamiltonian_dense(K.ravel(), omega, 4)).reshape(16, 16)
        np.testing.assert_allclose(H_sparse, H_dense, atol=1e-12)

    def test_matches_python_sparse(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H_python = knm_to_dense_matrix(K, omega)
        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 6)
        from scipy import sparse

        H_rust = sparse.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(64, 64),
        ).toarray()
        np.testing.assert_allclose(H_rust, H_python, atol=1e-12)

    def test_hermitian(self):
        K, omega = _system(4)
        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 4)
        from scipy import sparse

        H = sparse.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(16, 16),
        ).toarray()
        np.testing.assert_allclose(H, H.T, atol=1e-12)

    def test_n8_nnz(self):
        K, omega = _system(8)
        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 8)
        dim = 256
        assert len(vals) < dim * dim  # sparse, not dense
        assert len(vals) > dim  # more than diagonal

    def test_eigenvalues_match(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K, omega = _system(6)
        H_dense = knm_to_dense_matrix(K, omega)
        e_dense = np.sort(np.linalg.eigvalsh(H_dense))

        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, 6)
        from scipy import sparse

        H_rust = sparse.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(64, 64),
        )
        e_sparse = np.sort(np.linalg.eigvalsh(H_rust.toarray()))
        np.testing.assert_allclose(e_sparse, e_dense, atol=1e-10)


# =====================================================================
# magnetisation_labels
# =====================================================================
class TestRustMagnetisationLabels:
    def test_n2_values(self):
        labels = eng.magnetisation_labels(2)
        # |00⟩=M+2, |01⟩=M0, |10⟩=M0, |11⟩=M-2
        assert labels[0] == 2  # |00⟩
        assert labels[1] == 0  # |01⟩
        assert labels[2] == 0  # |10⟩
        assert labels[3] == -2  # |11⟩

    def test_n4_range(self):
        labels = eng.magnetisation_labels(4)
        assert len(labels) == 16
        assert min(labels) == -4
        assert max(labels) == 4

    def test_all_up_all_down(self):
        labels = eng.magnetisation_labels(8)
        assert labels[0] == 8  # |00000000⟩ = all up
        assert labels[255] == -8  # |11111111⟩ = all down

    def test_matches_python(self):
        from scpn_quantum_control.analysis.magnetisation_sectors import _magnetisation

        n = 6
        labels_rust = eng.magnetisation_labels(n)
        labels_python = [_magnetisation(k, n) for k in range(2**n)]
        np.testing.assert_array_equal(labels_rust, labels_python)

    def test_sum_is_zero_for_even_n(self):
        # Equal number of positive and negative M states
        labels = eng.magnetisation_labels(4)
        # Sum of all M labels weighted by degeneracy = 0
        assert sum(labels) == 0

    def test_popcount_consistency(self):
        labels = eng.magnetisation_labels(8)
        for k in range(256):
            expected = 8 - 2 * bin(k).count("1")
            assert labels[k] == expected


# =====================================================================
# order_param_from_statevector
# =====================================================================
class TestRustOrderParam:
    def test_all_up_state(self):
        n = 4
        psi = np.zeros(16, dtype=np.complex128)
        psi[0] = 1.0  # |0000⟩
        R = eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        # All up: ⟨X⟩=0, ⟨Y⟩=0 for each qubit → R=0
        assert abs(R) < 0.01

    def test_normalised_state(self):
        n = 4
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        R = eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        assert 0 <= R <= 1.01

    def test_matches_python(self):
        n = 4
        rng = np.random.default_rng(123)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)

        def python_only(psi_in, n_in):
            z = 0.0 + 0.0j
            dim = 2**n_in
            for i in range(n_in):
                exp_x = 0.0
                exp_y = 0.0
                for k in range(dim):
                    k_flip = k ^ (1 << i)
                    exp_x += float(np.real(psi_in[k].conj() * psi_in[k_flip]))
                    exp_y += float(np.imag(psi_in[k_flip].conj() * psi_in[k]))
                z += exp_x + 1j * exp_y
            z /= n_in
            return float(abs(z))

        R_python = python_only(psi, n)
        R_rust = eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        np.testing.assert_allclose(R_rust, R_python, atol=1e-10)

    def test_n8_performance(self):
        import time

        n = 8
        psi = np.random.randn(256) + 1j * np.random.randn(256)
        psi /= np.linalg.norm(psi)
        t0 = time.perf_counter()
        for _ in range(1000):
            eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 0.001  # should be <1ms at n=8

    def test_deterministic(self):
        n = 4
        psi = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        R1 = eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        R2 = eng.order_param_from_statevector(psi.real.copy(), psi.imag.copy(), n)
        assert R1 == R2
