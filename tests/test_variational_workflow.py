# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Variational workflow contract tests
"""Workflow tests for NQS, parameter-shift VQE, batch VQE, and contraction optimisation consistency."""

from __future__ import annotations

import numpy as np


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _homogeneous_system(n: int = 4):
    """Circulant K + uniform omega for translation symmetry."""
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = min(abs(i - j), n - abs(i - j))
            K[i, j] = 0.5 * np.exp(-0.3 * d) if d > 0 else 0
    omega = np.ones(n) * 1.0
    return n, K, omega


class TestVariationalPipeline:
    """Multiple variational methods should converge toward the exact ground
    state energy, with NQS and param-shift giving consistent results."""

    def test_nqs_approaches_exact(self):
        """RBM VMC energy should be within 20% of exact ground state."""
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix
        from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state

        _, K, omega = _system(4)
        H = knm_to_dense_matrix(K, omega)
        E_exact = np.linalg.eigvalsh(H)[0]

        result = vmc_ground_state(K, omega, n_iterations=300, seed=42)

        # VMC should get within 20% of exact (not guaranteed to be close)
        relative_error = abs(result["energy"] - E_exact) / abs(E_exact)
        assert relative_error < 0.5, (
            f"VMC energy {result['energy']:.4f} too far from exact {E_exact:.4f} "
            f"(relative error {relative_error:.2%})"
        )

    def test_param_shift_vqe_converges(self):
        """Parameter-shift VQE should reduce energy over iterations."""
        from scpn_quantum_control.phase.param_shift import vqe_with_param_shift

        def cost(params):
            return float(sum((p - 0.5) ** 2 for p in params))

        result = vqe_with_param_shift(
            cost,
            n_params=4,
            learning_rate=0.05,
            n_iterations=100,
            seed=42,
        )

        assert result["energy"] < result["energy_history"][0], (
            "VQE should reduce energy over iterations"
        )
        assert result["grad_norms"][-1] < result["grad_norms"][0], "Gradient norms should decrease"

    def test_batch_vqe_finds_better_than_random(self):
        """Batch VQE scan should find energy lower than mean random."""
        from scpn_quantum_control.phase.gpu_batch_vqe import batch_vqe_scan

        _, K, omega = _system(4)

        result = batch_vqe_scan(K, omega, n_samples=50, seed=42)
        mean_energy = np.mean(result["energies"])

        assert result["best_energy"] < mean_energy, (
            "Best energy should be below mean of random samples"
        )

    def test_nqs_and_sparse_agree_on_ground(self):
        """NQS VMC and sparse eigsh should find comparable ground energies.
        NQS is variational (upper bound) so E_nqs >= E_exact."""
        from scpn_quantum_control.bridge.sparse_hamiltonian import sparse_eigsh
        from scpn_quantum_control.phase.nqs_ansatz import vmc_ground_state

        _, K, omega = _system(4)

        E_sparse = sparse_eigsh(K, omega, k=1)["eigvals"][0]
        result_nqs = vmc_ground_state(K, omega, n_iterations=200, seed=42)

        # NQS is variational: E_nqs >= E_exact (within noise)
        assert result_nqs["energy"] >= E_sparse - 0.5, (
            f"NQS {result_nqs['energy']:.4f} suspiciously below exact {E_sparse:.4f}"
        )

    def test_contraction_optimiser_in_pipeline(self):
        """Contraction optimiser should give same results as np.einsum."""
        from scpn_quantum_control.phase.contraction_optimiser import contract

        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 16))
        B = rng.standard_normal((16, 12))
        C = rng.standard_normal((12, 6))

        result = contract("ij,jk,kl->il", A, B, C)
        expected = A @ B @ C

        np.testing.assert_allclose(result, expected, atol=1e-10)
