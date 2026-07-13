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

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation


def _as_real_numeric_array(name: str, values: object) -> NDArray[np.float64]:
    """Return a real numeric array without implicit string/bool/object coercion."""
    try:
        raw = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular numeric array.") from exc

    if raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError(f"{name} must contain real numeric scalars.")
    if raw.dtype.kind == "c":
        raise ValueError(f"{name} must contain real numeric scalars.")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric scalars.") from exc


def _as_nonnegative_rate(name: str, value: object) -> float:
    """Return a finite non-negative scalar rate without implicit coercion."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative real scalar.")
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a finite non-negative real scalar.")

    rate = float(raw)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return rate


def _validate_lindblad_inputs(
    n_oscillators: int,
    K_coupling: NDArray[np.float64],
    omega_natural: NDArray[np.float64],
    gamma_amp: float,
    gamma_deph: float,
) -> tuple[int, NDArray[np.float64], NDArray[np.float64], float, float]:
    """Validate and normalise the solver's physical inputs.

    Parameters
    ----------
    n_oscillators
        Positive oscillator and qubit count.
    K_coupling
        Candidate square, finite, symmetric coupling matrix.
    omega_natural
        Candidate finite natural-frequency vector.
    gamma_amp
        Candidate amplitude-damping rate.
    gamma_deph
        Candidate pure-dephasing rate.

    Returns
    -------
    tuple
        Validated oscillator count, a copied coupling matrix with a zero
        diagonal, a copied frequency vector, and the two finite non-negative
        damping rates.

    Raises
    ------
    ValueError
        If a count, shape, scalar type, finiteness condition, rate bound, or
        coupling symmetry condition is invalid.

    """
    if isinstance(n_oscillators, bool) or not isinstance(n_oscillators, int):
        raise ValueError("n_oscillators must be a positive integer.")
    if n_oscillators < 1:
        raise ValueError("n_oscillators must be at least 1.")

    K = _as_real_numeric_array("K_coupling", K_coupling)
    omega = _as_real_numeric_array("omega_natural", omega_natural)
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

    gamma_amp_value = _as_nonnegative_rate("gamma_amp", gamma_amp)
    gamma_deph_value = _as_nonnegative_rate("gamma_deph", gamma_deph)

    K = np.array(K, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    return n_oscillators, K, omega.copy(), gamma_amp_value, gamma_deph_value


def _validate_time_grid(t_max: float, dt: float) -> tuple[float, float]:
    """Validate and return the requested evolution horizon and step bound.

    Parameters
    ----------
    t_max
        Finite non-negative evolution horizon.
    dt
        Finite positive upper bound on the returned sample spacing.

    Returns
    -------
    tuple
        Validated ``(t_max, dt)`` values as Python floats.

    Raises
    ------
    ValueError
        If ``t_max`` is negative or non-finite, or if ``dt`` is non-positive
        or non-finite.

    """
    t_max_value = float(t_max)
    dt_value = float(dt)
    if not np.isfinite(t_max_value) or t_max_value < 0.0:
        raise ValueError("t_max must be finite and non-negative.")
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("dt must be finite and positive.")
    return t_max_value, dt_value


def _sigma(pauli: str, qubit: int, n: int) -> NDArray[np.complex128]:
    """Single-qubit Pauli operator on qubit `qubit` of `n`-qubit system."""
    matrices = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        "+": np.array([[0, 1], [0, 0]], dtype=np.complex128),
        "-": np.array([[0, 0], [1, 0]], dtype=np.complex128),
    }
    op: NDArray[np.complex128] = np.eye(1, dtype=np.complex128)
    for i in range(n):
        if i == qubit:
            op = np.kron(op, matrices[pauli]).astype(np.complex128)
        else:
            op = np.kron(op, np.eye(2, dtype=np.complex128)).astype(np.complex128)
    return op


class LindbladKuramotoSolver:
    """Solve open-system Kuramoto-XY dynamics with a Lindblad equation.

    Parameters
    ----------
    n_oscillators
        Positive number of oscillators and qubits.
    K_coupling
        Finite real symmetric coupling matrix of shape ``(n, n)``. Its
        diagonal is discarded by the Kuramoto-XY mapping.
    omega_natural
        Finite real natural-frequency vector of shape ``(n,)``.
    gamma_amp
        Finite non-negative amplitude-damping rate per qubit.
    gamma_deph
        Finite non-negative pure-dephasing rate per qubit.
    max_dense_gib
        Optional dense-workspace budget in GiB. When omitted, the shared dense
        budget resolver uses ``SCPN_MAX_DENSE_GIB`` or its host-aware default.

    Raises
    ------
    ValueError
        If the oscillator count, physical arrays, symmetry, finiteness, or
        damping-rate constraints are invalid.

    """

    def __init__(
        self,
        n_oscillators: int,
        K_coupling: NDArray[np.float64],
        omega_natural: NDArray[np.float64],
        gamma_amp: float = 0.0,
        gamma_deph: float = 0.0,
        *,
        max_dense_gib: float | None = None,
    ):
        """Initialize validated solver state without allocating dense operators."""
        self.n, self.K, self.omega, self.gamma_amp, self.gamma_deph = _validate_lindblad_inputs(
            n_oscillators,
            K_coupling,
            omega_natural,
            gamma_amp,
            gamma_deph,
        )
        self.max_dense_gib = max_dense_gib
        self.dim = 2**self.n
        self._H: NDArray[np.complex128] | None = None
        self._lindblad_ops: list[NDArray[np.complex128]] = []

    def _dense_object_count(self) -> int:
        """Return the conservative number of simultaneous dense matrices."""
        channel_count = 0
        if self.gamma_amp > 0:
            channel_count += self.n
        if self.gamma_deph > 0:
            channel_count += self.n
        return max(4, 4 + channel_count)

    def build(self, *, max_dense_gib: float | None = None) -> None:
        """Build and cache the Hamiltonian and Lindblad channel operators.

        Parameters
        ----------
        max_dense_gib
            Optional per-build dense-workspace budget in GiB. It overrides the
            constructor value; ``None`` reuses that value.

        Raises
        ------
        DenseAllocationError
            If the conservative dense workspace exceeds the active budget.
        ValueError
            If the active dense-workspace budget is not positive.

        """
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

    def _rhs(self, _t: float, rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Evaluate the flattened Lindblad right-hand side for SciPy.

        Parameters
        ----------
        _t
            Integration time accepted for the ``solve_ivp`` callback
            contract; the generator is time independent.
        rho_flat
            Flattened complex density matrix of length ``dim**2``.

        Returns
        -------
        numpy.ndarray
            Flattened density-matrix derivative.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not populated the Hamiltonian.

        """
        rho = rho_flat.reshape(self.dim, self.dim)

        # Coherent part: -i[H, rho]
        hamiltonian = self._H
        if hamiltonian is None:
            raise RuntimeError("Lindblad RHS requires build() before evaluation.")
        drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)

        # Dissipative part
        for L in self._lindblad_ops:
            Ld = L.conj().T
            LdL = Ld @ L
            drho += L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)

        return np.asarray(drho.ravel(), dtype=np.complex128)

    def order_parameter(self, rho: NDArray[np.complex128]) -> float:
        """Return the Kuramoto synchronisation magnitude for a density matrix.

        Parameters
        ----------
        rho
            Complex density matrix of shape ``(dim, dim)``.

        Returns
        -------
        float
            Magnitude of the mean single-qubit transverse expectation value.

        """
        z = 0.0 + 0.0j
        for i in range(self.n):
            sx = _sigma("X", i, self.n)
            sy = _sigma("Y", i, self.n)
            z += np.trace(sx @ rho) + 1j * np.trace(sy @ rho)
        z /= self.n
        return float(abs(z))

    def purity(self, rho: NDArray[np.complex128]) -> float:
        """Return the density-matrix purity ``Tr(rho**2)``.

        Parameters
        ----------
        rho
            Complex density matrix of shape ``(dim, dim)``.

        Returns
        -------
        float
            Purity, equal to one for a pure state and ``1 / dim`` for the
            maximally mixed state.

        """
        return float(np.trace(rho @ rho).real)

    def _initial_density_matrix(self) -> NDArray[np.complex128]:
        """Construct the frequency-seeded product-state density matrix.

        Returns
        -------
        numpy.ndarray
            Pure density matrix obtained by applying one ``R_y`` rotation per
            qubit to the all-zero computational-basis state. Each angle is the
            corresponding natural frequency reduced modulo ``2 * pi``.

        """
        rho0 = np.zeros((self.dim, self.dim), dtype=np.complex128)
        rho0[0, 0] = 1.0
        U: NDArray[np.complex128] = np.eye(1, dtype=np.complex128)
        for i in range(self.n):
            angle = float(self.omega[i]) % (2 * np.pi)
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            ry = np.array([[c, -s], [s, c]], dtype=np.complex128)
            U = np.kron(U, ry).astype(np.complex128)
        result: NDArray[np.complex128] = U @ rho0 @ U.conj().T
        return result

    def run(
        self,
        t_max: float,
        dt: float,
        method: str = "RK45",
        *,
        max_dense_gib: float | None = None,
    ) -> dict[str, Any]:
        """Evolve the density matrix under the configured Lindblad dynamics.

        Parameters
        ----------
        t_max
            Finite non-negative evolution horizon.
        dt
            Finite positive upper bound on adjacent output sample spacing.
        method
            Integration method forwarded to :func:`scipy.integrate.solve_ivp`.
        max_dense_gib
            Optional dense-workspace override used when this call must build
            the Hamiltonian and channels. A previously built solver is reused.

        Returns
        -------
        dict
            Heterogeneous array payload with ``times``, synchronisation
            history ``R``, ``purity`` history, and the final density matrix
            ``rho_final``. A zero horizon returns the initial state without
            invoking the integrator.

        Raises
        ------
        ValueError
            If the time grid, integration method, or active dense budget is
            invalid.
        DenseAllocationError
            If an unbuilt solver's dense workspace exceeds the active budget.
        RuntimeError
            If SciPy reports an unsuccessful integration.

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
