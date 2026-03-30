# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage Tests for v0.9.3 Additions
"""Tests for knm_to_dense_matrix, JAX fallback paths, and bridge functions."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)


class TestKnmToDenseMatrix:
    def test_matches_qiskit(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H_dense = knm_to_dense_matrix(K, omega)
        H_qiskit = knm_to_hamiltonian(K, omega).to_matrix()
        if hasattr(H_qiskit, "toarray"):
            H_qiskit = H_qiskit.toarray()
        np.testing.assert_allclose(H_dense, np.array(H_qiskit), atol=1e-12)

    def test_hermitian(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        H = knm_to_dense_matrix(K, omega)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_shape(self) -> None:
        for n in [2, 3, 4]:
            K = build_knm_paper27(L=n)
            omega = OMEGA_N_16[:n]
            H = knm_to_dense_matrix(K, omega)
            assert H.shape == (2**n, 2**n)

    def test_zero_coupling(self) -> None:
        n = 3
        K = np.zeros((n, n))
        omega = OMEGA_N_16[:n]
        H = knm_to_dense_matrix(K, omega)
        assert np.allclose(H, np.diag(np.diag(H)))


class TestKnmToXXZ:
    def test_xy_equals_xxz_delta0(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        H_xy = knm_to_hamiltonian(K, omega).to_matrix()
        H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=0.0).to_matrix()
        np.testing.assert_allclose(np.array(H_xy), np.array(H_xxz), atol=1e-12)

    def test_empty_pauli_list(self) -> None:
        n = 3
        K = np.full((n, n), 1e-20)
        np.fill_diagonal(K, 0)
        omega = np.zeros(n)
        H = knm_to_xxz_hamiltonian(K, omega)
        assert H.num_qubits == n


class TestBuildKuramotoRing:
    def test_returns_tuple(self) -> None:
        K, omega = build_kuramoto_ring(4)
        assert K.shape == (4, 4)
        assert omega.shape == (4,)

    def test_ring_topology(self) -> None:
        K, _ = build_kuramoto_ring(4, coupling=2.0)
        assert K[0, 1] == 2.0
        assert K[3, 0] == 2.0
        assert K[0, 2] == 0.0

    def test_custom_omega(self) -> None:
        omega_in = np.array([1.0, 2.0, 3.0])
        K, omega_out = build_kuramoto_ring(3, omega=omega_in)
        np.testing.assert_array_equal(omega_out, omega_in)


class TestJaxAccelFallback:
    def test_jax_not_available(self) -> None:
        from scpn_quantum_control.hardware.jax_accel import is_jax_available, is_jax_gpu_available

        # These should not crash regardless of JAX availability
        assert isinstance(is_jax_available(), bool)
        assert isinstance(is_jax_gpu_available(), bool)

    def test_jax_device_name(self) -> None:
        from scpn_quantum_control.hardware.jax_accel import jax_device_name

        name = jax_device_name()
        assert isinstance(name, str)

    def test_entanglement_scan_no_jax_gpu(self) -> None:
        """entanglement_vs_coupling should work without JAX GPU."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        K = build_knm_paper27(L=3)
        K_norm = K / np.max(K)
        omega = OMEGA_N_16[:3]
        result = entanglement_vs_coupling(omega, K_norm, np.linspace(1.0, 3.0, 5))
        assert len(result.entropy) == 5
        assert all(np.isfinite(result.entropy))
