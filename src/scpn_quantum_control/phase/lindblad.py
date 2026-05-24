# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Lindblad Open-System Kuramoto-XY Solver
"""Lindblad master equation solver for open Kuramoto-XY systems.

Solves dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
for the XY Hamiltonian with configurable amplitude damping and
dephasing channels.

Prior work: Ameri et al., PRA 91, 012301 (2015); Giorgi et al.,
PRA 85, 052101 (2012). This module provides a scipy-based solver
compatible with the QuantumKuramotoSolver interface.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation


def _validate_lindblad_inputs(
    n_oscillators: int,
    K_coupling: np.ndarray,
    omega_natural: np.ndarray,
    gamma_amp: float,
    gamma_deph: float,
) -> tuple[int, np.ndarray, np.ndarray, float, float]:
    if isinstance(n_oscillators, bool) or not isinstance(n_oscillators, int):
        raise ValueError("n_oscillators must be a positive integer.")
    if n_oscillators < 1:
        raise ValueError("n_oscillators must be at least 1.")

    K = np.asarray(K_coupling, dtype=np.float64)
    omega = np.asarray(omega_natural, dtype=np.float64)
    if K.ndim != 2 or K.shape != (n_oscillators, n_oscillators):
        raise ValueError(
            f"K_coupling must have shape ({n_oscillators}, {n_oscillators}); got {K.shape}."
        )
    if omega.ndim != 1 or omega.shape != (n_oscillators,):
        raise ValueError(f"omega_natural must have shape ({n_oscillators},); got {omega.shape}.")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)):
        raise ValueError("K_coupling and omega_natural must contain only finite values.")
    if not np.allclose(K, K.T, atol=1e-12, rtol=1e-12):
        raise ValueError("K_coupling must be symmetric for the Kuramoto-XY mapping.")

    gamma_amp_value = float(gamma_amp)
    gamma_deph_value = float(gamma_deph)
    if not np.isfinite(gamma_amp_value) or gamma_amp_value < 0.0:
        raise ValueError("gamma_amp must be finite and non-negative.")
    if not np.isfinite(gamma_deph_value) or gamma_deph_value < 0.0:
        raise ValueError("gamma_deph must be finite and non-negative.")

    K = np.array(K, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    return n_oscillators, K, omega.copy(), gamma_amp_value, gamma_deph_value


def _validate_time_grid(t_max: float, dt: float) -> tuple[float, float]:
    t_max_value = float(t_max)
    dt_value = float(dt)
    if not np.isfinite(t_max_value) or t_max_value < 0.0:
        raise ValueError("t_max must be finite and non-negative.")
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be finite and positive.")
    return t_max_value, dt_value


def _sigma(pauli: str, qubit: int, n: int) -> np.ndarray:
    """Single-qubit Pauli operator on qubit `qubit` of `n`-qubit system."""
    matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        "+": np.array([[0, 1], [0, 0]], dtype=np.complex128),
        "-": np.array([[0, 0], [1, 0]], dtype=np.complex128),
    }
    op: np.ndarray = np.eye(1, dtype=np.complex128)
    for i in range(n):
        if i == qubit:
            op = np.kron(op, matrices[pauli])  # type: ignore[assignment]
        else:
            op = np.kron(op, np.eye(2, dtype=np.complex128))  # type: ignore[assignment]
    return op


class LindbladKuramotoSolver:
    """Open-system Kuramoto-XY solver via Lindblad master equation.

    Parameters
    ----------
    n_oscillators : int
        Number of oscillators (qubits).
    K_coupling : array-like, shape (n, n)
        Coupling matrix.
    omega_natural : array-like, shape (n,)
        Natural frequencies.
    gamma_amp : float
        Amplitude damping rate per qubit (energy relaxation, T1).
    gamma_deph : float
        Pure dephasing rate per qubit (T2).
    """

    def __init__(
        self,
        n_oscillators: int,
        K_coupling: np.ndarray,
        omega_natural: np.ndarray,
        gamma_amp: float = 0.0,
        gamma_deph: float = 0.0,
        *,
        max_dense_gib: float | None = None,
    ):
        self.n, self.K, self.omega, self.gamma_amp, self.gamma_deph = _validate_lindblad_inputs(
            n_oscillators,
            K_coupling,
            omega_natural,
            gamma_amp,
            gamma_deph,
        )
        self.max_dense_gib = max_dense_gib
        self.dim = 2**self.n
        self._H: np.ndarray | None = None
        self._lindblad_ops: list[np.ndarray] = []

    def _dense_object_count(self) -> int:
        channel_count = 0
        if self.gamma_amp > 0:
            channel_count += self.n
        if self.gamma_deph > 0:
            channel_count += self.n
        return max(4, 4 + channel_count)

    def build(self, *, max_dense_gib: float | None = None) -> None:
        """Build Hamiltonian and Lindblad operators."""
        budget_gib = self.max_dense_gib if max_dense_gib is None else max_dense_gib
        require_dense_allocation(
            self.n,
            rank=2,
            object_count=self._dense_object_count(),
            max_gib=budget_gib,
            label="Lindblad dense density workspace",
        )
        self._H = knm_to_dense_matrix(self.K, self.omega, max_dense_gib=budget_gib)

        self._lindblad_ops = []
        for i in range(self.n):
            if self.gamma_amp > 0:
                self._lindblad_ops.append(np.sqrt(self.gamma_amp) * _sigma("-", i, self.n))
            if self.gamma_deph > 0:
                self._lindblad_ops.append(np.sqrt(self.gamma_deph / 2) * _sigma("Z", i, self.n))

    def _rhs(self, _t: float, rho_flat: np.ndarray) -> np.ndarray:
        """Lindblad RHS in flattened form for scipy integrator."""
        rho = rho_flat.reshape(self.dim, self.dim)

        # Coherent part: -i[H, rho]
        drho = -1j * (self._H @ rho - rho @ self._H)

        # Dissipative part
        for L in self._lindblad_ops:
            Ld = L.conj().T
            LdL = Ld @ L
            drho += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)

        return np.asarray(drho.ravel())

    def order_parameter(self, rho: np.ndarray) -> float:
        """Extract Kuramoto R from density matrix."""
        z = 0.0 + 0.0j
        for i in range(self.n):
            sx = _sigma("X", i, self.n)
            sy = _sigma("Y", i, self.n)
            z += np.trace(sx @ rho) + 1j * np.trace(sy @ rho)
        z /= self.n
        return float(abs(z))

    def purity(self, rho: np.ndarray) -> float:
        """Tr(ρ²) — 1 for pure state, 1/dim for maximally mixed."""
        return float(np.trace(rho @ rho).real)

    def _initial_density_matrix(self) -> np.ndarray:
        """Initial product state density matrix used by the Lindblad solver."""
        rho0 = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho0[0, 0] = 1.0
        U: np.ndarray = np.eye(1, dtype=np.complex128)
        for i in range(self.n):
            angle = float(self.omega[i]) % (2 * np.pi)
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            U = np.kron(U, ry)  # type: ignore[assignment]
        result: np.ndarray = U @ rho0 @ U.conj().T
        return result

    def run(
        self,
        t_max: float,
        dt: float,
        method: str = "RK45",
        *,
        max_dense_gib: float | None = None,
    ) -> dict:
        """Time-evolve under Lindblad dynamics.

        Returns dict with keys: times, R, purity, rho_final.
        """
        t_max, dt = _validate_time_grid(t_max, dt)
        budget_gib = self.max_dense_gib if max_dense_gib is None else max_dense_gib
        if self._H is None:
            self.build(max_dense_gib=budget_gib)

        n_steps = max(1, int(np.ceil(t_max / dt)))
        times = np.linspace(0, t_max, n_steps + 1)

        rho0 = self._initial_density_matrix()
        if t_max == 0.0:
            return {
                "times": np.array([0.0]),
                "R": np.array([self.order_parameter(rho0)]),
                "purity": np.array([self.purity(rho0)]),
                "rho_final": rho0,
            }

        sol = solve_ivp(
            self._rhs,
            [0, t_max],
            rho0.ravel(),
            t_eval=times,
            method=method,
            atol=1e-8,
            rtol=1e-6,
        )
        if not sol.success:
            raise RuntimeError(f"Lindblad integration failed: {sol.message}")

        R_history = np.zeros(len(times))
        purity_history = np.zeros(len(times))
        for i, _t in enumerate(times):
            rho_t = sol.y[:, i].reshape(self.dim, self.dim)
            R_history[i] = self.order_parameter(rho_t)
            purity_history[i] = self.purity(rho_t)

        rho_final = sol.y[:, -1].reshape(self.dim, self.dim)

        return {
            "times": times,
            "R": R_history,
            "purity": purity_history,
            "rho_final": rho_final,
        }
