# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Classical reference computations for hardware experiment comparison.

Each function returns the exact/high-fidelity classical answer that the
quantum hardware result should approximate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, expm_multiply

from ..bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_hamiltonian,
)


def classical_kuramoto_reference(
    n_osc: int,
    t_max: float,
    dt: float,
    K: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
) -> dict:
    """Euler integration of classical Kuramoto with Paper 27 parameters.

    Returns times, theta(t), R(t) for direct comparison with quantum results.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if t_max < 0:
        raise ValueError(f"t_max must be non-negative, got {t_max}")
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()
    if theta0 is None:
        theta0 = np.array([om % (2 * np.pi) for om in omega])

    n_steps = max(1, round(t_max / dt))

    # Rust fast path: ~100x faster for N >= 8
    try:
        import scpn_quantum_engine as _engine

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
    except (ImportError, AttributeError):
        pass

    times = np.linspace(0, t_max, n_steps + 1)
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


def _order_param(theta: np.ndarray) -> float:
    z = np.mean(np.exp(1j * theta))
    return float(abs(z))


def classical_exact_diag(
    n_osc: int,
    K: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    k_eigenvalues: int | None = None,
) -> dict:
    """Exact diagonalization of the XY Kuramoto Hamiltonian.

    For n_osc >= 14 (2^14 = 16384 entries), uses scipy.sparse.linalg.eigsh
    to compute only the lowest k_eigenvalues (default 6) without building
    a dense 2^n x 2^n array.

    Returns eigenvalues, ground energy, and ground state vector.
    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()

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
    K: np.ndarray | None = None,
    omega: np.ndarray | None = None,
) -> dict:
    """Exact matrix exponential evolution of XY Hamiltonian.

    Returns per-qubit X,Y expectations and reconstructed R(t).
    This is the gold standard the Trotter evolution should match.

    For n_osc >= 13, uses scipy.sparse.linalg.expm_multiply (Krylov
    subspace) to avoid materialising the full 2^n × 2^n propagator.
    Memory: O(2^n) instead of O(2^2n).
    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()

    H_op = knm_to_hamiltonian(K, omega)
    psi = _build_initial_state(n_osc, omega)

    n_steps = max(1, round(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
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
        H_mat = np.array(H_op.to_matrix())
        U_dt = expm(-1j * H_mat * dt)
        for s in range(1, n_steps + 1):
            psi = U_dt @ psi
            R_history[s] = _state_order_param(psi, n_osc)

    return {"times": times, "R": R_history}


def _build_initial_state(n_osc: int, omega: np.ndarray) -> np.ndarray:
    """Tensor product of Ry(omega_i mod 2pi)|0> in Qiskit little-endian order.

    Qiskit stores |b_{n-1}...b_1 b_0> with qubit 0 as the LSB, so the
    kron order must be q_{n-1} ⊗ ... ⊗ q_1 ⊗ q_0.
    """
    state: np.ndarray = np.array([1.0 + 0j])
    for i in reversed(range(n_osc)):
        angle = float(omega[i]) % (2 * np.pi)
        q = np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=complex)
        state = np.kron(state, q)
    return state


def _state_order_param(psi: np.ndarray, n_osc: int) -> float:
    """Compute R from statevector via X,Y expectations per qubit.

    Tries Rust fast path first (vectorised bitwise ops), falls back to
    Python kron-based implementation.
    """
    try:
        import scpn_quantum_engine as _engine

        return float(
            _engine.state_order_param_sparse(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n_osc,
            )
        )
    except (ImportError, AttributeError):
        pass

    z_complex = 0.0 + 0.0j
    for q in range(n_osc):
        exp_x = _expectation_pauli(psi, n_osc, q, "X")
        exp_y = _expectation_pauli(psi, n_osc, q, "Y")
        z_complex += exp_x + 1j * exp_y

    z_complex /= n_osc
    return float(abs(z_complex))


def _state_order_param_sparse(psi: np.ndarray, n_osc: int) -> float:
    """Compute R from statevector using vectorised bitwise Pauli application.

    Tries Rust fast path first (SIMD-friendly loop), falls back to numpy
    vectorised bit-flip implementation.
    O(n_osc * 2^n) time, O(2^n) memory.
    """
    try:
        import scpn_quantum_engine as _engine

        return float(
            _engine.state_order_param_sparse(
                np.ascontiguousarray(psi.real),
                np.ascontiguousarray(psi.imag),
                n_osc,
            )
        )
    except (ImportError, AttributeError):
        pass

    dim = len(psi)
    indices = np.arange(dim, dtype=np.int64)
    psi_conj = psi.conj()
    z_complex = 0.0 + 0.0j

    for q in range(n_osc):
        mask = 1 << q
        flipped = indices ^ mask
        psi_flipped = psi[flipped]

        exp_x = np.sum(psi_conj * psi_flipped).real

        bits = (indices >> q) & 1
        signs = 1.0 - 2.0 * bits
        exp_y = np.sum(psi_conj * (1j * signs) * psi_flipped).real

        z_complex += exp_x + 1j * exp_y

    z_complex /= n_osc
    return float(abs(z_complex))


def _expectation_pauli(psi: np.ndarray, n: int, qubit: int, pauli: str) -> float:
    """<psi| P_qubit |psi> where P acts on one qubit, identity elsewhere.

    Tries Rust bitwise fast path first, falls back to kron-based Python.
    """
    try:
        import scpn_quantum_engine as _engine

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
    except (ImportError, AttributeError):
        pass

    if pauli == "X":
        p = np.array([[0, 1], [1, 0]], dtype=complex)
    elif pauli == "Y":
        p = np.array([[0, -1j], [1j, 0]], dtype=complex)
    else:
        p = np.array([[1, 0], [0, -1]], dtype=complex)

    kron_pos = n - 1 - qubit
    op = np.array([[1.0]])
    for i in range(n):
        op = np.kron(op, p if i == kron_pos else np.eye(2))
    return float(np.real(psi.conj() @ op @ psi))


def bloch_vectors_from_json(path: str) -> dict:
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
    B_matrix: np.ndarray,
    target: np.ndarray,
    horizon: int,
) -> dict:
    """Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.

    Tries Rust parallel path first (rayon), falls back to Python.
    Returns optimal actions, optimal cost, all costs for comparison.
    """
    try:
        import scpn_quantum_engine as _engine

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
    except (ImportError, AttributeError):
        pass

    n_actions = 2**horizon
    best_cost = np.inf
    best_actions: np.ndarray = np.zeros(horizon, dtype=int)
    all_costs = np.zeros(n_actions)

    b_norm = float(np.linalg.norm(B_matrix))
    t_norm = float(np.linalg.norm(target))

    for idx in range(n_actions):
        actions = np.array([(idx >> bit) & 1 for bit in range(horizon)])
        cost = 0.0
        for t in range(horizon):
            diff = b_norm * actions[t] - t_norm / horizon
            cost += diff**2
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
