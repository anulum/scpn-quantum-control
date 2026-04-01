# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Lindblad Master Equation and Quantum Trajectory Solver.

Simulates open quantum systems with dissipators driving the network
towards macroscopic phase synchronization. Supports both density matrix
evolution (RK45) and Quantum Trajectory (Monte Carlo Wavefunction) paths.

The trajectory path scales as O(2^N) memory, enabling simulation of the
full 16-layer SCPN framework without the O(2^2N) density matrix bottleneck.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import expm_multiply

from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_sparse_matrix


class LindbladSyncEngine:
    """Solves the Lindblad master equation for open quantum system sync."""

    def __init__(self, K: np.ndarray, omega: np.ndarray, gamma: float = 0.1):
        self.n = len(omega)
        self.dim = 1 << self.n
        self.K = K
        self.omega = omega
        self.gamma = gamma

        # Dense Hamiltonian for small-N density matrix path
        if self.n <= 10:
            self.H_dense: np.ndarray | None = knm_to_dense_matrix(K, omega)
            self.L_ops_dense = self._build_jump_operators_dense()
        else:
            self.H_dense = None
            self.L_ops_dense = []

        # Sparse components for trajectory path
        self.H_sparse = knm_to_sparse_matrix(K, omega)
        self.L_ops_sparse = self._build_jump_operators_sparse()
        self.anti_hermitian_sum = self._build_anti_hermitian_sum()

    def _build_jump_operators_dense(self) -> list[np.ndarray]:
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

    def _build_jump_operators_sparse(self) -> list:
        from scipy.sparse import csr_matrix

        L_ops = []
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(self.K[i, j]) > 1e-5:
                    row, col, data = [], [], []
                    for idx in range(self.dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            flipped = idx ^ ((1 << i) | (1 << j))
                            row.append(flipped)
                            col.append(idx)
                            data.append(1.0)
                    L = csr_matrix((data, (row, col)), shape=(self.dim, self.dim), dtype=complex)
                    L_ops.append(L)
        return L_ops

    def _build_anti_hermitian_sum(self) -> np.ndarray:
        diag = np.zeros(self.dim, dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i != j and abs(self.K[i, j]) > 1e-5:
                    for idx in range(self.dim):
                        if ((idx >> i) & 1) == 1 and ((idx >> j) & 1) == 0:
                            diag[idx] += 1.0
        return diag

    def liouvillian(self, rho_flat: np.ndarray) -> np.ndarray:
        if self.H_dense is None:
            raise RuntimeError("Density matrix path only supported for N <= 10.")
        rho_mat = rho_flat.reshape((self.dim, self.dim))
        drho = -1j * (self.H_dense @ rho_mat - rho_mat @ self.H_dense)
        for L in self.L_ops_dense:
            L_dag = L.conj().T
            drho += self.gamma * (
                L @ rho_mat @ L_dag - 0.5 * (L_dag @ L @ rho_mat + rho_mat @ L_dag @ L)
            )
        result: np.ndarray = drho.flatten()
        return result

    def evolve(
        self,
        t_max: float,
        n_steps: int = 100,
        method: str = "trajectory",
        initial_state: np.ndarray | None = None,
        n_traj: int = 20,
        seed: int = 42,
        observables: list[SparsePauliOp] | None = None,
    ) -> dict[str, Any]:
        """Evolve system using density matrix or quantum trajectories."""
        if method == "density_matrix":
            return self._evolve_density_matrix(t_max, n_steps, initial_state, observables)
        elif method == "trajectory":
            return self._evolve_trajectories(
                t_max, n_steps, n_traj, initial_state, seed, observables
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _evolve_density_matrix(
        self,
        t_max: float,
        n_steps: int,
        initial_rho: np.ndarray | None,
        observables: list[SparsePauliOp] | None,
    ) -> dict[str, Any]:
        if initial_rho is None:
            rho0 = np.zeros((self.dim, self.dim), dtype=complex)
            rho0[0, 0] = 1.0
        else:
            rho0 = initial_rho

        times = np.linspace(0, t_max, n_steps + 1)
        res = solve_ivp(
            lambda t, y: self.liouvillian(y),
            [0, t_max],
            rho0.flatten(),
            t_eval=times,
            method="RK45",
        )

        results: dict[str, Any] = {"times": res.t}
        if self.n <= 10:
            states = [res.y[:, i].reshape((self.dim, self.dim)) for i in range(len(res.t))]
            results["states"] = states
            results["final_state"] = states[-1]

        if observables:
            obs_history: dict[str, list[float]] = {str(o): [] for o in observables}
            obs_mats = [o.to_matrix() for o in observables]
            for i in range(len(res.t)):
                rho = res.y[:, i].reshape((self.dim, self.dim))
                for j, M in enumerate(obs_mats):
                    val = np.trace(M @ rho).real
                    obs_history[str(observables[j])].append(float(val))
            results["observables"] = obs_history

        return results

    def _evolve_trajectories(
        self,
        t_max: float,
        n_steps: int,
        n_traj: int,
        initial_psi: np.ndarray | None,
        seed: int,
        observables: list[SparsePauliOp] | None,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        dt = t_max / n_steps
        times = np.linspace(0, t_max, n_steps + 1)

        from scipy.sparse import diags

        H_eff = self.H_sparse - 0.5j * self.gamma * diags(self.anti_hermitian_sum)
        A = -1j * H_eff * dt

        obs_mats = [o.to_matrix(sparse=True) for o in (observables or [])]
        obs_avg = {str(o): np.zeros(n_steps + 1) for o in (observables or [])}

        # Only store density matrix if small N
        avg_rho = (
            [np.zeros((self.dim, self.dim), dtype=complex) for _ in range(n_steps + 1)]
            if self.n <= 10
            else None
        )

        for _ in range(n_traj):
            psi = (
                initial_psi.copy()
                if initial_psi is not None
                else np.zeros(self.dim, dtype=complex)
            )
            if initial_psi is None:
                psi[0] = 1.0

            self._update_stats(0, psi, avg_rho, obs_avg, obs_mats, n_traj)

            for s in range(1, n_steps + 1):
                psi = expm_multiply(A, psi)
                norm_sq = np.vdot(psi, psi).real
                if rng.random() > norm_sq:
                    probs = [np.vdot(L @ psi, L @ psi).real for L in self.L_ops_sparse]
                    j_idx = rng.choice(len(self.L_ops_sparse), p=np.array(probs) / np.sum(probs))
                    psi = self.L_ops_sparse[j_idx] @ psi
                psi /= np.sqrt(np.vdot(psi, psi).real)
                self._update_stats(s, psi, avg_rho, obs_avg, obs_mats, n_traj)

        res: dict[str, Any] = {"times": times}
        if avg_rho:
            res["states"] = avg_rho
            res["final_state"] = avg_rho[-1]
        if observables:
            res["observables"] = {k: list(v) for k, v in obs_avg.items()}
        return res

    def _update_stats(self, step, psi, avg_rho, obs_avg, obs_mats, n_traj):
        if avg_rho is not None:
            avg_rho[step] += np.outer(psi, psi.conj()) / n_traj
        for i, M in enumerate(obs_mats):
            val = np.vdot(psi, M @ psi).real
            obs_avg[list(obs_avg.keys())[i]][step] += val / n_traj
