# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Lindblad Master Equation Solver for Biological Synchronization.

Simulates open quantum systems with dissipators driving the network
towards macroscopic phase synchronization. Rather than purely unitary
evolution (which never truly equilibrates), this module explicitly
models the biological environment collapsing the system into a synchronized
subspace.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix


class LindbladSyncEngine:
    """Solves the Lindblad master equation for open quantum system sync."""

    def __init__(self, K: np.ndarray, omega: np.ndarray, gamma: float = 0.1):
        self.n = len(omega)
        self.dim = 1 << self.n
        self.K = K
        self.omega = omega
        self.gamma = gamma
        self.H = knm_to_dense_matrix(K, omega)
        self.L_ops = self._build_jump_operators()

    def _build_jump_operators(self) -> list[np.ndarray]:
        """Build pairwise dissipators L_ij = sigma_i^- sigma_j^+ for connected nodes."""
        L_ops = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(self.K[i, j]) > 1e-5:
                    L = np.zeros((self.dim, self.dim), dtype=complex)
                    for idx in range(self.dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            flipped = idx ^ ((1 << i) | (1 << j))
                            L[flipped, idx] = 1.0
                    L_ops.append(L)
        return L_ops

    def liouvillian(self, rho_flat: np.ndarray) -> np.ndarray:
        rho_mat = rho_flat.reshape((self.dim, self.dim))

        # Unitary part: -i[H, rho]
        drho = -1j * (self.H @ rho_mat - rho_mat @ self.H)

        # Dissipative part
        for L in self.L_ops:
            L_dag = L.conj().T
            term1 = L @ rho_mat @ L_dag
            term2 = 0.5 * (L_dag @ L @ rho_mat + rho_mat @ L_dag @ L)
            drho += self.gamma * (term1 - term2)

        result: np.ndarray = drho.flatten()
        return result

    def evolve(
        self, t_max: float, n_steps: int = 100, initial_rho: np.ndarray | None = None
    ) -> dict:
        """Evolve the density matrix using RK45."""
        if initial_rho is None:
            rho0 = np.zeros((self.dim, self.dim), dtype=complex)
            rho0[0, 0] = 1.0
        else:
            rho0 = initial_rho

        def odefun(t: float, y: np.ndarray) -> np.ndarray:
            return self.liouvillian(y)

        t_eval = np.linspace(0, t_max, n_steps + 1)
        res = solve_ivp(odefun, [0, t_max], rho0.flatten(), t_eval=t_eval, method="RK45")

        states = []
        for i in range(len(res.t)):
            states.append(res.y[:, i].reshape((self.dim, self.dim)))

        return {"times": res.t, "states": states, "final_state": states[-1]}
