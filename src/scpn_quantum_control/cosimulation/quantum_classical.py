# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum/classical Kuramoto co-simulation
"""Mean-field co-simulation of a quantum-strong core inside a classical bath.

The quantum core evolves as an exact statevector under its internal XY
Hamiltonian, driven by the in-plane mean field of the classical bath; the
classical bath evolves as a Kuramoto network driven by the coherence-weighted
mean field of the quantum core. The two are interleaved with a second-order
Trotter split on the quantum side (exact internal propagator from an
eigendecomposition, exact single-qubit drive rotation) and an explicit-Euler
step on the classical side.

This is a local mean-field embedding: the quantum/classical boundary couplings
are treated at mean-field level (no cross-boundary entanglement). The partition
``cross_fraction`` bounds the decoupling error. It is not an exact treatment of
the full network and not a hardware path.

Coupling, in the Kuramoto-XY mapping ``K_ij sin(theta_j - theta_i)``:

* classical -> quantum: oscillator ``c`` at phase ``theta_c`` acts on core spin
  ``i`` as an in-plane field ``(b_x, b_y) = sum_c K_ic (cos theta_c, sin theta_c)``;
* quantum -> classical: core spin ``q`` with moment ``(<X_q>, <Y_q>)`` acts on
  oscillator ``c`` through ``K_cq`` so the classical drive is
  ``cos(theta_c) * sum_q K_cq <Y_q> - sin(theta_c) * sum_q K_cq <X_q>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._rust_accel import optional_rust_engine
from ..dense_budget import require_dense_allocation
from .knm_partition import KnmPartition, partition_knm


@dataclass(frozen=True)
class CoSimulationResult:
    """Trajectories and order parameters from a quantum/classical co-simulation."""

    times: NDArray[np.float64]
    classical_phases: NDArray[np.float64]
    quantum_expectation_x: NDArray[np.float64]
    quantum_expectation_y: NDArray[np.float64]
    quantum_order: NDArray[np.float64]
    classical_order: NDArray[np.float64]
    global_order: NDArray[np.float64]
    baseline_classical_order: NDArray[np.float64]
    baseline_global_order: NDArray[np.float64]
    baseline_deviation: float
    partition: KnmPartition
    provenance: dict[str, Any] = field(default_factory=dict)


def _qubit_view(
    state: NDArray[np.complex128], qubit: int, n_qubits: int
) -> NDArray[np.complex128]:
    """Reshape ``state`` so axis 1 selects ``qubit`` (bit = (idx >> qubit) & 1)."""
    high = 1 << (n_qubits - qubit - 1)
    low = 1 << qubit
    return state.reshape(high, 2, low)


def _apply_single_qubit(
    state: NDArray[np.complex128], gate: NDArray[np.complex128], qubit: int, n_qubits: int
) -> NDArray[np.complex128]:
    view = _qubit_view(state, qubit, n_qubits)
    out = np.einsum("ab,xbz->xaz", gate, view)
    result: NDArray[np.complex128] = out.reshape(-1)
    return result


def _expectation_xy(
    state: NDArray[np.complex128], n_qubits: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return per-qubit ``<X_i>`` and ``<Y_i>`` arrays."""
    exp_x = np.empty(n_qubits, dtype=np.float64)
    exp_y = np.empty(n_qubits, dtype=np.float64)
    for i in range(n_qubits):
        view = _qubit_view(state, i, n_qubits)
        overlap = np.sum(np.conjugate(view[:, 0, :]) * view[:, 1, :])
        exp_x[i] = 2.0 * overlap.real
        exp_y[i] = 2.0 * overlap.imag
    return exp_x, exp_y


def _drive_gate(field_x: float, field_y: float, dt: float) -> NDArray[np.complex128]:
    """exp(i dt (field_x X + field_y Y)) as a 2x2 unitary."""
    magnitude = float(np.hypot(field_x, field_y))
    if magnitude == 0.0:
        return np.eye(2, dtype=np.complex128)
    cos = np.cos(magnitude * dt)
    sin = np.sin(magnitude * dt)
    nx = field_x / magnitude
    ny = field_y / magnitude
    return np.array(
        [
            [cos, 1j * sin * (nx - 1j * ny)],
            [1j * sin * (nx + 1j * ny), cos],
        ],
        dtype=np.complex128,
    )


def _internal_half_propagator(
    K_q: NDArray[np.float64], omega_q: NDArray[np.float64], dt: float
) -> NDArray[np.complex128]:
    """Exact exp(-i H_internal dt/2) for the quantum core (H real symmetric)."""
    n = len(omega_q)
    require_dense_allocation(n, dtype=np.complex128, rank=2, label="co-simulation quantum core")
    engine = optional_rust_engine()
    if engine is not None and hasattr(engine, "build_xy_hamiltonian_dense"):
        flat = np.asarray(
            engine.build_xy_hamiltonian_dense(
                np.ascontiguousarray(0.5 * (K_q + K_q.T)).ravel(),
                np.ascontiguousarray(omega_q),
                n,
            )
        )
        hamiltonian: NDArray[np.float64] = flat.reshape(2**n, 2**n).astype(np.float64)
    else:
        hamiltonian = _xy_hamiltonian_dense_python(K_q, omega_q)
    evals, evecs = np.linalg.eigh(hamiltonian)
    phase = np.exp(-0.5j * dt * evals)
    propagator: NDArray[np.complex128] = (evecs * phase) @ evecs.conj().T
    return propagator


def _xy_hamiltonian_dense_python(
    K: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pure-Python XY Hamiltonian, fallback for the Rust kernel.

    H = -sum_{i<j} K_ij (X_iX_j + Y_iY_j) - sum_i omega_i Z_i, bit i = (idx>>i)&1.
    """
    n = len(omega)
    K = 0.5 * (K + K.T)
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    for idx in range(dim):
        diag = 0.0
        for i in range(n):
            bit = (idx >> i) & 1
            diag -= omega[i] * (1.0 - 2.0 * bit)
        H[idx, idx] = diag
    for i in range(n):
        for j in range(i + 1, n):
            kij = K[i, j]
            if kij == 0.0:
                continue
            bit_i = 1 << i
            bit_j = 1 << j
            for idx in range(dim):
                bi = (idx >> i) & 1
                bj = (idx >> j) & 1
                if bi != bj:
                    # X_iX_j + Y_iY_j flips the unequal pair with weight 2.
                    flipped = idx ^ bit_i ^ bit_j
                    H[idx, flipped] += -2.0 * kij
    return H


def _classical_substep(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    K: NDArray[np.float64],
    drive_a: NDArray[np.float64],
    drive_b: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    engine = optional_rust_engine()
    if engine is not None and hasattr(engine, "cosim_classical_substep"):
        return np.asarray(
            engine.cosim_classical_substep(theta, omega, K, drive_a, drive_b, dt),
            dtype=np.float64,
        )
    return _classical_substep_python(theta, omega, K, drive_a, drive_b, dt)


def _classical_substep_python(
    theta: NDArray[np.float64],
    omega: NDArray[np.float64],
    K: NDArray[np.float64],
    drive_a: NDArray[np.float64],
    drive_b: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    diff = theta[None, :] - theta[:, None]
    internal = np.sum(K * np.sin(diff), axis=1)
    quantum = np.cos(theta) * drive_a - np.sin(theta) * drive_b
    stepped: NDArray[np.float64] = theta + dt * (omega + internal + quantum)
    return stepped


def _order_parameter(phases: NDArray[np.float64]) -> float:
    if phases.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * phases))))


def _full_classical_order(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    theta0: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """All-classical Kuramoto reference order-parameter trajectory."""
    theta = theta0.copy()
    orders = np.empty(n_steps + 1, dtype=np.float64)
    orders[0] = _order_parameter(theta)
    for step in range(n_steps):
        diff = theta[None, :] - theta[:, None]
        theta = theta + dt * (omega + np.sum(K * np.sin(diff), axis=1))
        orders[step + 1] = _order_parameter(theta)
    return orders


def _initial_quantum_state(
    n_qubits: int, override: NDArray[np.complex128] | None
) -> NDArray[np.complex128]:
    if override is not None:
        state = np.asarray(override, dtype=np.complex128).reshape(-1)
        if state.shape[0] != 2**n_qubits:
            raise ValueError(f"quantum_state0 must have length {2**n_qubits}")
        norm = float(np.linalg.norm(state))
        if norm == 0.0:
            raise ValueError("quantum_state0 must be non-zero")
        normalised: NDArray[np.complex128] = state / norm
        return normalised
    # Default |+>^{⊗n}: every core spin in-plane at phase 0 (<X>=1, <Y>=0).
    return np.full(2**n_qubits, 2.0 ** (-n_qubits / 2.0), dtype=np.complex128)


def cosimulate(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    dt: float,
    n_steps: int,
    partition: KnmPartition | None = None,
    max_quantum_nodes: int = 8,
    coupling_threshold: float = 0.0,
    theta0_classical: NDArray[np.float64] | None = None,
    quantum_state0: NDArray[np.complex128] | None = None,
    seed: int | None = None,
) -> CoSimulationResult:
    """Run a mean-field quantum/classical co-simulation of a K_nm network.

    Args:
        K: symmetric ``(N, N)`` coupling matrix.
        omega: length-``N`` natural-frequency vector.
        dt: positive time step.
        n_steps: number of co-simulation steps (>= 1).
        partition: a precomputed :class:`KnmPartition`; built via
            :func:`partition_knm` when omitted.
        max_quantum_nodes: core-size cap when building the partition.
        coupling_threshold: core-growth cutoff when building the partition.
        theta0_classical: initial classical phases (length ``N_classical``);
            random with ``seed`` when omitted.
        quantum_state0: initial core statevector (length ``2**N_quantum``);
            ``|+>^{⊗N_quantum}`` when omitted.
        seed: RNG seed for the default classical phases.

    Returns
    -------
        A :class:`CoSimulationResult` with the classical-phase and quantum-moment
        trajectories, the quantum/classical/global order parameters, the
        all-classical baseline, and the partition record.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite value")
    if n_steps < 1:
        raise ValueError("n_steps must be a positive integer")

    if partition is None:
        partition = partition_knm(
            K,
            omega,
            max_quantum_nodes=max_quantum_nodes,
            coupling_threshold=coupling_threshold,
        )
    K_sym = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    omega = np.asarray(omega, dtype=np.float64)

    n_q = partition.n_quantum
    n_c = partition.n_classical
    cross = partition.cross_coupling  # (n_q, n_c)
    K_cc = np.ascontiguousarray(partition.classical_coupling)
    omega_c = partition.classical_omega

    if theta0_classical is None:
        rng = np.random.default_rng(seed)
        theta_c = rng.uniform(-np.pi, np.pi, size=n_c) if n_c else np.zeros(0)
    else:
        theta_c = np.asarray(theta0_classical, dtype=np.float64).copy()
        if theta_c.shape != (n_c,):
            raise ValueError(f"theta0_classical must have length {n_c}")

    state = _initial_quantum_state(n_q, quantum_state0)
    half = _internal_half_propagator(partition.quantum_coupling, partition.quantum_omega, dt)

    classical_phases = np.empty((n_steps + 1, n_c), dtype=np.float64)
    exp_x_traj = np.empty((n_steps + 1, n_q), dtype=np.float64)
    exp_y_traj = np.empty((n_steps + 1, n_q), dtype=np.float64)
    quantum_order = np.empty(n_steps + 1, dtype=np.float64)
    classical_order = np.empty(n_steps + 1, dtype=np.float64)
    global_order = np.empty(n_steps + 1, dtype=np.float64)

    def record(step: int) -> None:
        exp_x, exp_y = _expectation_xy(state, n_q)
        exp_x_traj[step] = exp_x
        exp_y_traj[step] = exp_y
        classical_phases[step] = theta_c
        quantum_vec = exp_x + 1j * exp_y
        classical_vec = np.exp(1j * theta_c) if n_c else np.zeros(0, dtype=complex)
        quantum_order[step] = float(np.abs(np.sum(quantum_vec)) / n_q) if n_q else 0.0
        classical_order[step] = _order_parameter(theta_c)
        total = np.sum(quantum_vec) + (np.sum(classical_vec) if n_c else 0.0)
        global_order[step] = float(np.abs(total) / (n_q + n_c))

    record(0)
    for step in range(n_steps):
        # Quantum second-order Trotter split with the classical mean field.
        state = half @ state
        if n_c:
            field_x = cross @ np.cos(theta_c)
            field_y = cross @ np.sin(theta_c)
            for i in range(n_q):
                gate = _drive_gate(field_x[i], field_y[i], dt)
                state = _apply_single_qubit(state, gate, i, n_q)
        state = half @ state

        # Classical Euler step with the quantum-core mean field.
        if n_c:
            exp_x, exp_y = _expectation_xy(state, n_q)
            drive_a = cross.T @ exp_y
            drive_b = cross.T @ exp_x
            theta_c = _classical_substep(theta_c, omega_c, K_cc, drive_a, drive_b, dt)
        record(step + 1)

    # All-classical baseline over the full network with matched initial phases.
    theta0_full = np.zeros(K_sym.shape[0], dtype=np.float64)
    exp_x0, exp_y0 = _expectation_xy(_initial_quantum_state(n_q, quantum_state0), n_q)
    theta0_full[list(partition.quantum_indices)] = np.arctan2(exp_y0, exp_x0)
    if n_c:
        theta0_full[list(partition.classical_indices)] = classical_phases[0]
    baseline_global = _full_classical_order(K_sym, omega, theta0_full, dt, n_steps)
    baseline_classical = (
        _full_classical_order(K_cc, omega_c, classical_phases[0], dt, n_steps)
        if n_c
        else np.zeros(n_steps + 1)
    )
    deviation = float(np.sqrt(np.mean((global_order - baseline_global) ** 2)))

    provenance = {
        "dt": dt,
        "n_steps": n_steps,
        "n_quantum": n_q,
        "n_classical": n_c,
        "quantum_integrator": "second_order_trotter_exact_internal",
        "classical_integrator": "explicit_euler",
        "cross_fraction": partition.conservation.cross_fraction,
        "engine": optional_rust_engine() is not None,
        "claim_boundary": (
            "mean-field embedding; baseline_global is the all-classical reference, "
            "deviation measures the quantum-core effect plus the decoupling error, "
            "not a certified error bound; not exact, not hardware"
        ),
    }

    return CoSimulationResult(
        times=np.arange(n_steps + 1, dtype=np.float64) * dt,
        classical_phases=classical_phases,
        quantum_expectation_x=exp_x_traj,
        quantum_expectation_y=exp_y_traj,
        quantum_order=quantum_order,
        classical_order=classical_order,
        global_order=global_order,
        baseline_classical_order=baseline_classical,
        baseline_global_order=baseline_global,
        baseline_deviation=deviation,
        partition=partition,
        provenance=provenance,
    )
