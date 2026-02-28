"""Classical reference computations for hardware experiment comparison.

Each function returns the exact/high-fidelity classical answer that the
quantum hardware result should approximate.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

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
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()
    if theta0 is None:
        theta0 = np.array([om % (2 * np.pi) for om in omega])

    n_steps = max(1, round(t_max / dt))
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
) -> dict:
    """Exact diagonalization of the XY Kuramoto Hamiltonian.

    Returns eigenvalues, ground energy, and ground state vector.
    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = (
        H_op.to_matrix().toarray()
        if hasattr(H_op.to_matrix(), "toarray")
        else np.array(H_op.to_matrix())
    )

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
    """
    if K is None:
        K = build_knm_paper27(L=n_osc)
    if omega is None:
        omega = OMEGA_N_16[:n_osc].copy()

    H_op = knm_to_hamiltonian(K, omega)
    H_mat = np.array(H_op.to_matrix())
    2**n_osc

    # Initial state: each qubit Ry(omega_i mod 2pi) |0>
    psi = _build_initial_state(n_osc, omega)

    n_steps = max(1, round(t_max / dt))
    times = np.linspace(0, t_max, n_steps + 1)
    R_history = np.zeros(n_steps + 1)
    R_history[0] = _state_order_param(psi, n_osc)

    U_dt = expm(-1j * H_mat * dt)
    for s in range(1, n_steps + 1):
        psi = U_dt @ psi
        R_history[s] = _state_order_param(psi, n_osc)

    return {"times": times, "R": R_history}


def _build_initial_state(n_osc: int, omega: np.ndarray) -> np.ndarray:
    """Tensor product of Ry(omega_i mod 2pi)|0> for each qubit."""
    state = np.array([1.0 + 0j])
    for i in range(n_osc):
        angle = float(omega[i]) % (2 * np.pi)
        q = np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=complex)
        state = np.kron(state, q)
    return state


def _state_order_param(psi: np.ndarray, n_osc: int) -> float:
    """Compute R from statevector via X,Y expectations per qubit."""
    2**n_osc
    z_complex = 0.0 + 0.0j

    for q in range(n_osc):
        # Build single-qubit Pauli projected into full Hilbert space
        exp_x = _expectation_pauli(psi, n_osc, q, "X")
        exp_y = _expectation_pauli(psi, n_osc, q, "Y")
        z_complex += exp_x + 1j * exp_y

    z_complex /= n_osc
    return float(abs(z_complex))


def _expectation_pauli(psi: np.ndarray, n: int, qubit: int, pauli: str) -> float:
    """<psi| P_qubit |psi> where P acts on one qubit, identity elsewhere."""
    if pauli == "X":
        p = np.array([[0, 1], [1, 0]], dtype=complex)
    elif pauli == "Y":
        p = np.array([[0, -1j], [1j, 0]], dtype=complex)
    else:
        p = np.array([[1, 0], [0, -1]], dtype=complex)

    op = np.array([[1.0]])
    for i in range(n):
        op = np.kron(op, p if i == qubit else np.eye(2))
    return float(np.real(psi.conj() @ op @ psi))


def classical_brute_mpc(
    B_matrix: np.ndarray,
    target: np.ndarray,
    horizon: int,
) -> dict:
    """Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.

    Returns optimal actions, optimal cost, all costs for comparison.
    """
    n_actions = 2**horizon
    best_cost = np.inf
    best_actions = np.zeros(horizon, dtype=int)
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
