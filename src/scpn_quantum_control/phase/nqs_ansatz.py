# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Neural Quantum State Ansatz
"""Restricted Boltzmann Machine (RBM) ansatz for Kuramoto-XY ground state.

Variational Monte Carlo with a neural network wavefunction, inspired by
Carleo & Troyer, Science 355, 602 (2017) and the NetKet framework.

The RBM parameterises log(ψ(σ)) = Σ a_i σ_i + Σ log cosh(Σ W_ij σ_j + b_i)
where σ ∈ {+1, -1}^N is a spin configuration.

Pure numpy implementation — no JAX/torch dependency.
For production use, consider NetKet (https://netket.org).
"""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


class RBMWavefunction:
    """Restricted Boltzmann Machine wavefunction for spin-1/2 systems.

    Parameters
    ----------
    n_visible : int
        Number of spins (qubits).
    n_hidden : int
        Number of hidden units. More = more expressive. Typical: 2*n_visible.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(self, n_visible: int, n_hidden: int | None = None, seed: int | None = None):
        self.n_vis = n_visible
        self.n_hid = n_hidden or 2 * n_visible
        self.rng = np.random.default_rng(seed)

        # Parameters: visible biases, hidden biases, weights
        scale = 0.01
        self.a = self.rng.normal(0, scale, self.n_vis).astype(np.complex128)
        self.b = self.rng.normal(0, scale, self.n_hid).astype(np.complex128)
        self.W = self.rng.normal(0, scale, (self.n_hid, self.n_vis)).astype(np.complex128)

    def log_psi(self, sigma: np.ndarray) -> complex:
        """Compute log(ψ(σ)) for a spin configuration σ ∈ {+1,-1}^n."""
        theta = self.W @ sigma + self.b
        return complex(np.sum(self.a * sigma) + np.sum(np.log(np.cosh(theta))))

    def psi(self, sigma: np.ndarray) -> complex:
        """Compute ψ(σ)."""
        return complex(np.exp(self.log_psi(sigma)))

    def all_amplitudes(self) -> np.ndarray:
        """Compute ψ(σ) for all 2^n configurations (exact, for small n)."""
        dim = 2**self.n_vis
        amps = np.zeros(dim, dtype=np.complex128)
        for k in range(dim):
            sigma = np.array([1 - 2 * ((k >> i) & 1) for i in range(self.n_vis)], dtype=np.float64)
            amps[k] = self.psi(sigma)
        norm = np.sqrt(np.sum(np.abs(amps) ** 2))
        return amps / norm if norm > 0 else amps

    def n_params(self) -> int:
        """Total number of variational parameters."""
        return self.n_vis + self.n_hid + self.n_hid * self.n_vis


def vmc_ground_state(
    K: np.ndarray,
    omega: np.ndarray,
    n_hidden: int | None = None,
    learning_rate: float = 0.01,
    n_iterations: int = 200,
    n_samples: int = 500,
    seed: int | None = None,
) -> dict:
    """Variational Monte Carlo ground state search with RBM ansatz.

    For small systems (n<=12), uses exact summation over all configurations.
    For larger systems, would need MCMC sampling (not implemented here).

    Parameters
    ----------
    K, omega : coupling matrix and frequencies
    n_hidden : hidden units (default 2*n)
    learning_rate : gradient descent step
    n_iterations : optimisation steps
    n_samples : ignored for exact mode (used for future MCMC)
    seed : RNG seed

    Returns
    -------
    dict with keys: energy, energy_history, wavefunction, n_params
    """
    n = K.shape[0]
    if n > 12:
        raise ValueError(
            f"Exact NQS only for n<=12 (got n={n}). MCMC sampling not yet implemented."
        )

    H = knm_to_dense_matrix(K, omega)
    rbm = RBMWavefunction(n, n_hidden=n_hidden, seed=seed)

    energy_history = []

    for _step in range(n_iterations):
        # Forward: compute amplitudes and energy
        psi_vec = rbm.all_amplitudes()
        energy = float(np.real(psi_vec.conj() @ H @ psi_vec))
        energy_history.append(energy)

        # Gradient via finite differences (simple, not efficient)
        grad_a = np.zeros_like(rbm.a)
        grad_b = np.zeros_like(rbm.b)
        grad_W = np.zeros_like(rbm.W)
        eps = 1e-4

        for i in range(rbm.n_vis):
            rbm.a[i] += eps
            psi_p = rbm.all_amplitudes()
            e_p = float(np.real(psi_p.conj() @ H @ psi_p))
            rbm.a[i] -= 2 * eps
            psi_m = rbm.all_amplitudes()
            e_m = float(np.real(psi_m.conj() @ H @ psi_m))
            rbm.a[i] += eps  # restore
            grad_a[i] = (e_p - e_m) / (2 * eps)

        for i in range(rbm.n_hid):
            rbm.b[i] += eps
            psi_p = rbm.all_amplitudes()
            e_p = float(np.real(psi_p.conj() @ H @ psi_p))
            rbm.b[i] -= 2 * eps
            psi_m = rbm.all_amplitudes()
            e_m = float(np.real(psi_m.conj() @ H @ psi_m))
            rbm.b[i] += eps
            grad_b[i] = (e_p - e_m) / (2 * eps)

        for i in range(rbm.n_hid):
            for j in range(rbm.n_vis):
                rbm.W[i, j] += eps
                psi_p = rbm.all_amplitudes()
                e_p = float(np.real(psi_p.conj() @ H @ psi_p))
                rbm.W[i, j] -= 2 * eps
                psi_m = rbm.all_amplitudes()
                e_m = float(np.real(psi_m.conj() @ H @ psi_m))
                rbm.W[i, j] += eps
                grad_W[i, j] = (e_p - e_m) / (2 * eps)

        # Update
        rbm.a -= learning_rate * grad_a
        rbm.b -= learning_rate * grad_b
        rbm.W -= learning_rate * grad_W

    final_psi = rbm.all_amplitudes()
    final_energy = float(np.real(final_psi.conj() @ H @ final_psi))

    return {
        "energy": final_energy,
        "energy_history": energy_history,
        "wavefunction": rbm,
        "n_params": rbm.n_params(),
    }
