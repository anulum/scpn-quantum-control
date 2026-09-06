# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Classical
"""Classical reference computations for hardware experiment comparison.

Each function returns the exact/high-fidelity classical answer that the
quantum hardware result should approximate.
"""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, expm_multiply

from .._rust_accel import optional_rust_engine
from ..bridge.knm_hamiltonian import (
    build_knm_paper27,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
    omega_for_oscillators,
)

INTEGRATION_GRID_RELATIVE_TOLERANCE: Final[float] = 1e-9
"""Relative tolerance used to recognise a duration that is a multiple of ``dt``.

``t_max / dt`` is rarely exact in binary floating point: ``0.5 / 0.1`` is
``4.999999999999999`` and ``2.0 / 0.1`` is ``19.999999999999996``. Taking the
floor of those would silently drop the last step of a perfectly divisible
interval, so a quotient within this relative tolerance of an integer is snapped
to it. Anything further away is genuinely non-divisible and is floored.
"""


def integration_step_count(t_max: float, dt: float) -> int:
    """Return how many whole ``dt`` steps fit inside ``t_max``.

    This is one half of the shared integration grid contract. The step count
    never overshoots the requested duration: a caller asking to evolve to
    ``t_max`` does not get a state from beyond it. A duration shorter than one
    step therefore yields zero steps, and the trajectory is the initial
    condition alone.

    Parameters
    ----------
    t_max
        Requested duration in the same time units as ``dt``. Must be
        non-negative; zero is admitted and means no evolution.
    dt
        Integration step, strictly positive.

    Returns
    -------
    int
        Number of steps, ``floor(t_max / dt)`` after snapping a quotient within
        :data:`INTEGRATION_GRID_RELATIVE_TOLERANCE` of an integer.

    Raises
    ------
    ValueError
        If ``dt`` is not strictly positive and finite, or ``t_max`` is negative
        or not finite.

    """
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt must be positive and finite, got {dt}")
    if not (math.isfinite(t_max) and t_max >= 0.0):
        raise ValueError(f"t_max must be non-negative and finite, got {t_max}")
    quotient = t_max / dt
    nearest = round(quotient)
    if abs(quotient - nearest) <= INTEGRATION_GRID_RELATIVE_TOLERANCE * max(1.0, abs(quotient)):
        return int(nearest)
    return int(math.floor(quotient))


def integration_times(n_steps: int, dt: float) -> NDArray[np.float64]:
    """Return the time of each sample on the shared integration grid.

    This is the other half of the contract: sample ``s`` is reported at
    ``s * dt``, which is where the integrator actually put the state. The
    previous grid spread ``linspace(0, t_max, n_steps + 1)`` over the requested
    duration while the integrator advanced by ``dt``, so for a non-divisible
    interval every label disagreed with its own state. The same expression is
    used by the Rust trajectory kernel, so both tiers report identical times.

    Parameters
    ----------
    n_steps
        Number of integration steps, from :func:`integration_step_count`.
    dt
        Integration step, strictly positive.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_steps + 1,)`` array ``[0, dt, 2·dt, …, n_steps·dt]``. The
        last entry is the end of the trajectory, which is at most ``t_max``.

    Raises
    ------
    ValueError
        If ``n_steps`` is negative.

    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    return np.arange(n_steps + 1, dtype=np.float64) * dt


def classical_kuramoto_reference(
    n_osc: int,
    t_max: float,
    dt: float,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    theta0: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Euler integration of classical Kuramoto with Paper 27 parameters.

    Returns times, theta(t), R(t) for direct comparison with quantum results.

    The trajectory follows the shared integration grid: ``floor(t_max / dt)``
    steps, with sample ``s`` reported at ``s · dt``. When ``t_max`` is not a
    multiple of ``dt`` the trajectory therefore ends short of ``t_max`` rather
    than mislabelling its last state, and ``t_max = 0`` returns the initial
    condition alone instead of taking one step. The Rust and Python tiers use
    the same expression and return identical times.

    Parameters
    ----------
    n_osc
        Number of oscillators.
    t_max
        Requested duration; non-negative and finite.
    dt
        Integration step; strictly positive and finite.
    K
        Coupling matrix of shape ``(n_osc, n_osc)``; defaults to Paper 27.
    omega
        Natural frequencies of shape ``(n_osc,)``; defaults to Paper 27.
    theta0
        Initial phases of shape ``(n_osc,)``; defaults to ``omega mod 2π``.

    Returns
    -------
    dict
        ``times`` of shape ``(n_steps + 1,)``, ``theta`` of shape
        ``(n_steps + 1, n_osc)`` and ``R`` of shape ``(n_steps + 1,)``.

    Raises
    ------
    ValueError
        If ``dt`` is not positive and finite, or ``t_max`` is negative or not
        finite.

    """
    n_steps = integration_step_count(t_max, dt)
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = omega_for_oscillators(n_osc)
    if theta0 is None:
        theta0 = np.array([om % (2 * np.pi) for om in omega])

    # Rust fast path: ~100x faster for N >= 8
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        times_rs, R_rs = _engine.kuramoto_trajectory(theta0, omega, K, dt, n_steps)
        theta_history_rs = np.zeros((n_steps + 1, n_osc))
        theta_history_rs[0] = theta0
        for s in range(1, n_steps + 1):
            theta_history_rs[s] = np.asarray(
                _engine.kuramoto_euler(theta_history_rs[s - 1], omega, K, dt, 1)
            )
        return {
            "times": np.asarray(times_rs),
            "theta": theta_history_rs,
            "R": np.asarray(R_rs),
        }
    except AttributeError:
        pass

    times = integration_times(n_steps, dt)
    theta_history = np.zeros((n_steps + 1, n_osc))
    R_history = np.zeros(n_steps + 1)

    theta = theta0.copy()
    theta_history[0] = theta
    R_history[0] = _order_param(theta)

    for s in range(1, n_steps + 1):
        dtheta = omega.copy()
        for i in range(n_osc):
            coupling = 0.0
            for j in range(n_osc):
                coupling += K[i, j] * np.sin(theta[j] - theta[i])
            dtheta[i] += coupling
        theta = theta + dt * dtheta
        theta_history[s] = theta
        R_history[s] = _order_param(theta)

    return {"times": times, "theta": theta_history, "R": R_history}


def _order_param(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter R = |<exp(i theta)>|.

    Delegates to :func:`scpn_quantum_control.accel.order_parameter`,
    which dispatches through the multi-language accel chain
    (Rust → Julia → Python floor). Per the 2026-04-17 benchmark
    (docs/pipeline_performance.md §"Multi-language accel chain"),
    Rust wins at every measured N; callers need not think about
    tier selection.

    Falls back to an inline NumPy implementation only if the accel
    package itself fails to import — keeps minimal installs working.
    """
    try:
        from oscillatools.accel import order_parameter as _op
    except Exception:
        z = np.mean(np.exp(1j * theta))
        return float(abs(z))
    return _op(theta)


def classical_exact_diag(
    n_osc: int,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    k_eigenvalues: int | None = None,
) -> dict[str, Any]:
    """Exact diagonalization of the XY Kuramoto Hamiltonian.

    For n_osc >= 14 (2^14 = 16384 entries), uses scipy.sparse.linalg.eigsh
    to compute only the lowest k_eigenvalues (default 6) without building
    a dense 2^n x 2^n array.

    Returns eigenvalues, ground energy, and ground state vector.
    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = omega_for_oscillators(n_osc)

    H_op = knm_to_hamiltonian(K, omega)

    if k_eigenvalues is not None or n_osc >= 14:
        k = k_eigenvalues or 6
        raw = H_op.to_matrix()
        H_sparse = csc_matrix(raw) if not hasattr(raw, "tocsc") else raw.tocsc()
        eigenvalues, eigenvectors = eigsh(H_sparse, k=k, which="SA")
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        from .gpu_accel import eigh as gpu_eigh
        from .gpu_accel import is_gpu_available

        raw = H_op.to_matrix()
        H_mat = raw.toarray() if hasattr(raw, "toarray") else np.array(raw)
        if is_gpu_available() and H_mat.shape[0] >= 64:
            eigenvalues, eigenvectors = gpu_eigh(H_mat)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(H_mat)

    return {
        "eigenvalues": eigenvalues,
        "ground_energy": float(eigenvalues[0]),
        "ground_state": eigenvectors[:, 0],
        "spectral_gap": float(eigenvalues[1] - eigenvalues[0]),
        "n_qubits": n_osc,
    }


def classical_exact_evolution(
    n_osc: int,
    t_max: float,
    dt: float,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Exact matrix exponential evolution of XY Hamiltonian.

    Returns per-qubit X,Y expectations and reconstructed R(t).
    This is the gold standard the Trotter evolution should match.

    For n_osc >= 13, uses scipy.sparse.linalg.expm_multiply (Krylov
    subspace) to avoid materialising the full 2^n × 2^n propagator.
    Memory: O(2^n) instead of O(2^2n).

    Uses the same integration grid as
    :func:`classical_kuramoto_reference`: the propagator is applied
    ``floor(t_max / dt)`` times and sample ``s`` is reported at ``s · dt``, so
    the reported time is where the state actually is. ``t_max = 0`` returns the
    initial state alone.

    Parameters
    ----------
    n_osc
        Number of oscillators.
    t_max
        Requested duration; non-negative and finite.
    dt
        Integration step; strictly positive and finite.
    K
        Coupling matrix of shape ``(n_osc, n_osc)``; defaults to Paper 27.
    omega
        Natural frequencies of shape ``(n_osc,)``; defaults to Paper 27.
    max_dense_gib
        Admission budget for the dense propagator, in GiB.

    Returns
    -------
    dict
        ``times`` of shape ``(n_steps + 1,)``, ``R`` of the same shape, and the
        per-qubit expectation entries.

    Raises
    ------
    ValueError
        If ``dt`` is not positive and finite, or ``t_max`` is negative or not
        finite.

    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = omega_for_oscillators(n_osc)

    H_op = knm_to_hamiltonian(K, omega)
    psi = _build_initial_state(n_osc, omega)

    n_steps = integration_step_count(t_max, dt)
    times = integration_times(n_steps, dt)
    R_history = np.zeros(n_steps + 1)
    R_history[0] = _state_order_param(psi, n_osc)

    if n_osc >= 13:
        # Sparse Krylov path: O(2^n) memory
        raw = H_op.to_matrix(sparse=True)
        H_sparse = csc_matrix(raw) if not hasattr(raw, "tocsc") else raw.tocsc()
        A = -1j * H_sparse * dt
        for s in range(1, n_steps + 1):
            psi = expm_multiply(A, psi)
            R_history[s] = _state_order_param_sparse(psi, n_osc)
    else:
        # Dense path: build U_dt once, reuse
        H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
        U_dt = expm(-1j * H_mat * dt)
        for s in range(1, n_steps + 1):
            psi = U_dt @ psi
            R_history[s] = _state_order_param(psi, n_osc)

    return {"times": times, "R": R_history}


def _build_initial_state(n_osc: int, omega: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Tensor product of Ry(omega_i mod 2pi)|0> in Qiskit little-endian order.

    Qiskit stores |b_{n-1}...b_1 b_0> with qubit 0 as the LSB, so the
    kron order must be q_{n-1} ⊗ ... ⊗ q_1 ⊗ q_0.
    """
    state: NDArray[np.complex128] = np.array([1.0 + 0j])
    for i in reversed(range(n_osc)):
        angle = float(omega[i]) % (2 * np.pi)
        q = np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=complex)
        state = np.kron(state, q)
    return state


def _state_order_param(psi: NDArray[np.complex128], n_osc: int) -> float:
    """Compute R from statevector via X,Y expectations per qubit.

    Tries Rust fast path first (vectorised bitwise ops), falls back to
    NumPy bitwise-vectorised single-qubit expectations.
    """
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        return float(
            _engine.state_order_param_sparse(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n_osc,
            )
        )
    except AttributeError:
        pass

    exp_x, exp_y = _xy_expectations_vectorized(psi, n_osc)
    z_complex = np.mean(exp_x + 1j * exp_y)
    return float(abs(z_complex))


def _state_order_param_sparse(psi: NDArray[np.complex128], n_osc: int) -> float:
    """Compute R from statevector using vectorised bitwise Pauli application.

    Tries Rust fast path first (SIMD-friendly loop), falls back to numpy
    vectorised bit-flip implementation.
    O(n_osc * 2^n) time, O(2^n) memory.
    """
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        return float(
            _engine.state_order_param_sparse(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n_osc,
            )
        )
    except AttributeError:
        pass

    exp_x, exp_y = _xy_expectations_vectorized(psi, n_osc)
    z_complex = np.mean(exp_x + 1j * exp_y)
    return float(abs(z_complex))


def _xy_expectations_vectorized(
    psi: NDArray[np.complex128], n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return all single-qubit X and Y expectations using bitwise indexing.

    This is the Python fallback for the Rust expectation hot path. It avoids
    constructing one dense Kronecker operator or Qiskit Pauli object per qubit
    while preserving Qiskit's little-endian statevector convention.
    """
    state = np.asarray(psi, dtype=complex)
    dim = len(state)
    indices = np.arange(dim, dtype=np.int64)
    state_conj = state.conj()
    exp_x = np.empty(n, dtype=float)
    exp_y = np.empty(n, dtype=float)

    for qubit in range(n):
        mask = 1 << qubit
        flipped = indices ^ mask
        flipped_state = state[flipped]
        bits = (indices >> qubit) & 1
        y_phase = -1j * (1.0 - 2.0 * bits)

        exp_x[qubit] = float(np.sum(state_conj * flipped_state).real)
        exp_y[qubit] = float(np.sum(state_conj * y_phase * flipped_state).real)

    return exp_x, exp_y


def _expectation_pauli(psi: NDArray[np.complex128], n: int, qubit: int, pauli: str) -> float:
    """<psi| P_qubit |psi> where P acts on one qubit, identity elsewhere.

    Tries Rust bitwise fast path first, falls back to NumPy bitwise indexing.
    """
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        pauli_idx = {"X": 0, "Y": 1, "Z": 2}[pauli]
        return float(
            _engine.expectation_pauli_fast(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n,
                qubit,
                pauli_idx,
            )
        )
    except AttributeError:
        pass

    if pauli == "X":
        exp_x, _ = _xy_expectations_vectorized(psi, n)
        return float(exp_x[qubit])
    elif pauli == "Y":
        _, exp_y = _xy_expectations_vectorized(psi, n)
        return float(exp_y[qubit])
    if pauli == "Z":
        state = np.asarray(psi, dtype=complex)
        indices = np.arange(len(state), dtype=np.int64)
        signs = 1.0 - 2.0 * ((indices >> qubit) & 1)
        probabilities = np.abs(state) ** 2
        return float(np.sum(signs * probabilities).real)
    raise ValueError(f"unsupported Pauli label: {pauli!r}")


def bloch_vectors_from_json(path: str) -> dict[str, Any]:
    """Extract per-qubit Bloch vector magnitudes from a hardware result JSON.

    Expects keys 'exp_x', 'exp_y', 'exp_z' (lists of per-qubit expectations).
    Returns dict with 'bloch_magnitudes' (sqrt(X^2+Y^2+Z^2) per qubit) and
    the raw expectation arrays.
    """
    import json as _json
    from pathlib import Path as _Path

    with open(_Path(path)) as f:
        data = _json.load(f)

    ex = np.array(data["exp_x"])
    ey = np.array(data["exp_y"])
    ez = np.array(data["exp_z"])
    magnitudes = np.sqrt(ex**2 + ey**2 + ez**2)
    return {
        "exp_x": ex,
        "exp_y": ey,
        "exp_z": ez,
        "bloch_magnitudes": magnitudes,
        "n_qubits": len(ex),
    }


def classical_brute_mpc(
    B_matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    horizon: int,
) -> dict[str, Any]:
    """Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.

    Evaluates the tracking cost ``C(u) = sum_t ||u_t * v - r||^2`` where
    ``v = B @ ones`` is the row-sum actuation vector, ``u_t`` in ``{0, 1}``
    switches the whole actuation vector on or off at timestep ``t``, and ``r``
    is the target. The residual stays a vector, so the target's sign and its
    direction relative to ``B`` both change the result.

    Tries the Rust parallel path first (rayon), falls back to Python. Both paths
    evaluate the same cost.

    Parameters
    ----------
    B_matrix
        Square actuation matrix of shape ``(dim, dim)``.
    target
        Target state vector of length ``dim``, in the same units as
        ``B_matrix @ ones``.
    horizon
        Positive number of binary timesteps; the enumeration is over
        ``2 ** horizon`` sequences.

    Returns
    -------
    dict
        ``optimal_actions`` (int array of shape ``(horizon,)``),
        ``optimal_cost`` (float), ``all_costs`` (float array of shape
        ``(2 ** horizon,)``, indexed so bit ``t`` of the index is ``u_t``) and
        ``n_evaluated`` (int).

    """
    try:
        _engine = optional_rust_engine()
        if _engine is None:
            raise AttributeError("scpn_quantum_engine absent")
        dim = B_matrix.shape[0]
        actions, cost, all_costs, n_eval = _engine.brute_mpc(
            B_matrix.ravel().astype(np.float64),
            target.astype(np.float64),
            dim,
            horizon,
        )
        return {
            "optimal_actions": np.asarray(actions, dtype=int),
            "optimal_cost": float(cost),
            "all_costs": np.asarray(all_costs),
            "n_evaluated": int(n_eval),
        }
    except AttributeError:
        pass

    n_actions = 2**horizon
    best_cost = np.inf
    best_actions: NDArray[np.int64] = np.zeros(horizon, dtype=int)
    all_costs = np.zeros(n_actions)

    actuation = np.asarray(B_matrix, dtype=np.float64).sum(axis=1)
    residual_target = np.asarray(target, dtype=np.float64)

    for idx in range(n_actions):
        actions = np.array([(idx >> bit) & 1 for bit in range(horizon)])
        cost = 0.0
        for t in range(horizon):
            residual = actions[t] * actuation - residual_target
            cost += float(residual @ residual)
        all_costs[idx] = cost
        if cost < best_cost:
            best_cost = cost
            best_actions = actions.copy()

    return {
        "optimal_actions": best_actions,
        "optimal_cost": float(best_cost),
        "all_costs": all_costs,
        "n_evaluated": n_actions,
    }
