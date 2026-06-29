# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Avqds
"""McLachlan variational quantum real-time dynamics with a fixed ansatz.

The state is evolved in real time with McLachlan's time-dependent variational
principle (TDVP). For a parametrised ansatz A(θ)|0>, the parameters follow the
equation of motion that minimises the McLachlan distance
||d/dt|ψ> + iH|ψ>||²:

    M(θ) × dθ/dt = V(θ),
    M_ij = Re(<∂_i ψ|∂_j ψ>),   V_i = -Im(<∂_i ψ|H|ψ>),

where M is the real part of the quantum geometric tensor (the Fubini–Study
metric) and V is the force projected onto the tangent space. Integrating this
ODE propagates θ(t) so that A(θ(t))|0> tracks e^{-iHt}|ψ_0>.

Relationship to AVQDS (Yao et al., PRX Quantum 2, 030307 (2021))
----------------------------------------------------------------
The McLachlan equation of motion above is the dynamical core of AVQDS. The full
AVQDS algorithm additionally grows the ansatz *adaptively*: at each step it
draws from an operator pool and appends the operator that most reduces the
McLachlan distance, so the parameter count rises on demand. This module does
NOT perform that operator-pool growth — the ansatz is fixed at construction
(see :func:`scpn_quantum_control.bridge.knm_hamiltonian.knm_to_ansatz`: a
physics-informed hardware-efficient circuit of RY/RZ rotations with CZ
entanglers on K_nm-connected pairs), so the parameter count is constant for the
whole trajectory. It is therefore fixed-ansatz variational quantum real-time
evolution (VarQRTE) — the non-adaptive special case of AVQDS, retaining the
historical name and acronym for the cited lineage.

Two honesty caveats on fidelity to the publication:
    - M is built from central finite-difference parameter derivatives
      (``epsilon`` below), not the analytic quantum geometric tensor.
    - Circuit depth is set by the ansatz and is independent of ``t_total``; this
      follows from using any fixed-ansatz variational propagator, not from
      operator growth.

For the Kuramoto-XY system this tracks synchronisation dynamics in real time
with a depth fixed by the ansatz rather than by a Trotter step count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_ansatz, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@runtime_checkable
class _SparseMatrixLike(Protocol):
    """Matrix object with a dense conversion method."""

    def toarray(self) -> object:
        """Return a dense matrix representation."""


@dataclass
class AVQDSResult:
    """Result of a fixed-ansatz McLachlan variational real-time simulation.

    Attributes:
        times: sample times of the trajectory, shape ``(n_steps + 1,)``.
        energies: ``<ψ(t)|H|ψ(t)>`` at each sample time.
        fidelities: squared overlap ``|<ψ_exact(t)|ψ(t)>|²`` against the
            exact reference evolution at each sample time.
        parameters_history: ansatz parameter vectors θ(t) at each sample time.
            Every entry has the same length ``n_params`` — the ansatz is fixed,
            so the count never grows (no adaptive operator pool).
        n_params: number of variational parameters of the fixed ansatz.
        final_energy: energy at the final sample time.
        final_fidelity: fidelity against the exact reference at the final time.
    """

    times: FloatArray
    energies: FloatArray
    fidelities: FloatArray  # overlap with exact evolution
    parameters_history: list[FloatArray]
    n_params: int
    final_energy: float
    final_fidelity: float


def _qubits_from_state_length(state_length: int) -> int:
    """Infer qubit count from an exact statevector length."""
    if state_length < 1 or state_length & (state_length - 1):
        raise ValueError(f"statevector length must be a positive power of two, got {state_length}")
    return state_length.bit_length() - 1


def _dense_complex_matrix(matrix: object) -> ComplexArray:
    """Return a dense complex matrix from Qiskit or sparse-like matrix output."""
    if isinstance(matrix, _SparseMatrixLike):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.complex128)


def _mclachlan_matrices(
    ansatz: QuantumCircuit,
    params: FloatArray,
    H_op: SparsePauliOp,
    epsilon: float = 1e-4,
    *,
    max_dense_gib: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute McLachlan M matrix and V vector.

    M_ij = Re(<∂_i ψ|∂_j ψ>)
    V_i = -Im(<∂_i ψ|H|ψ>)
    """
    ansatz_qubits = getattr(ansatz, "num_qubits", None)
    if ansatz_qubits is not None:
        require_dense_allocation(
            int(ansatz_qubits),
            rank=2,
            max_gib=max_dense_gib,
            label="AVQDS dense Hamiltonian",
        )
    n_params = len(params)
    sv_0 = Statevector.from_instruction(ansatz.assign_parameters(params))
    psi_0 = sv_0.data
    if ansatz_qubits is None:
        require_dense_allocation(
            _qubits_from_state_length(len(psi_0)),
            rank=2,
            max_gib=max_dense_gib,
            label="AVQDS dense Hamiltonian",
        )

    # Compute parameter derivatives via finite differences
    dpsi: ComplexArray = np.zeros((n_params, len(psi_0)), dtype=np.complex128)
    for k in range(n_params):
        p_plus = params.copy()
        p_plus[k] += epsilon
        p_minus = params.copy()
        p_minus[k] -= epsilon
        psi_plus = Statevector.from_instruction(ansatz.assign_parameters(p_plus)).data
        psi_minus = Statevector.from_instruction(ansatz.assign_parameters(p_minus)).data
        dpsi[k] = (psi_plus - psi_minus) / (2.0 * epsilon)

    # M matrix
    M = np.zeros((n_params, n_params), dtype=np.float64)
    for i in range(n_params):
        for j in range(n_params):
            M[i, j] = float(np.real(np.dot(dpsi[i].conj(), dpsi[j])))

    # V vector
    H_mat = _dense_complex_matrix(H_op.to_matrix())
    H_psi = H_mat @ psi_0

    V = np.zeros(n_params, dtype=np.float64)
    for i in range(n_params):
        V[i] = -float(np.imag(np.dot(dpsi[i].conj(), H_psi)))

    return M, V


def avqds_simulate(
    K: FloatArray,
    omega: FloatArray,
    t_total: float = 1.0,
    n_steps: int = 20,
    ansatz_reps: int = 2,
    seed: int | None = None,
    *,
    max_dense_gib: float | None = None,
) -> AVQDSResult:
    """Evolve the Kuramoto-XY dynamics by fixed-ansatz McLachlan variational propagation.

    Builds the fixed physics-informed ansatz once, then time-steps its
    parameters by solving the McLachlan equation of motion
    ``M(θ) dθ/dt = V(θ)`` (regularised linear solve) while propagating an exact
    reference state for the fidelity comparison. The ansatz structure is never
    modified, so ``n_params`` is constant across the whole trajectory; this is
    not the adaptive operator-pool growth of full AVQDS.

    Args:
        K: coupling matrix.
        omega: natural frequencies.
        t_total: total simulation time.
        n_steps: number of variational time steps.
        ansatz_reps: ansatz circuit repetitions (sets the fixed parameter count).
        seed: random seed for initial parameters.
        max_dense_gib: dense exact-simulation budget for Hamiltonian,
            statevector, and propagator allocations.

    Returns:
        AVQDSResult with the time grid, energies, fidelities against the exact
        reference, the parameter history, and the final-step summaries.
    """
    from ..hardware.gpu_accel import expm

    n = K.shape[0]
    require_dense_allocation(
        n,
        rank=2,
        max_gib=max_dense_gib,
        label="AVQDS dense Hamiltonian",
    )
    H_op = knm_to_hamiltonian(K, omega)
    H_mat = _dense_complex_matrix(H_op.to_matrix())

    ansatz = knm_to_ansatz(K, reps=ansatz_reps)
    n_params = ansatz.num_parameters

    rng = np.random.default_rng(seed)
    params = rng.normal(0, 0.1, size=n_params)
    dt = t_total / n_steps

    times: FloatArray = np.linspace(0.0, t_total, n_steps + 1, dtype=np.float64)
    energies = np.zeros(n_steps + 1, dtype=np.float64)
    fidelities = np.zeros(n_steps + 1, dtype=np.float64)
    param_history: list[FloatArray] = [params.copy()]

    # Initial exact state for fidelity comparison
    psi_exact = Statevector.from_instruction(ansatz.assign_parameters(params)).data

    for step in range(n_steps + 1):
        sv = Statevector.from_instruction(ansatz.assign_parameters(params))
        energies[step] = float(sv.expectation_value(H_op).real)
        fidelities[step] = float(abs(np.dot(psi_exact.conj(), sv.data)) ** 2)

        if step == n_steps:
            break

        # McLachlan step
        M, V = _mclachlan_matrices(ansatz, params, H_op, max_dense_gib=max_dense_gib)

        # Regularised solve: dθ = M^{-1} V × dt
        reg = 1e-6 * np.eye(n_params)
        dtheta = np.linalg.solve(M + reg, V) * dt
        params = params + dtheta
        param_history.append(params.copy())

        # Evolve exact reference
        U_dt = expm(-1j * H_mat * dt)
        psi_exact = U_dt @ psi_exact

    return AVQDSResult(
        times=times,
        energies=energies,
        fidelities=fidelities,
        parameters_history=param_history,
        n_params=n_params,
        final_energy=float(energies[-1]),
        final_fidelity=float(fidelities[-1]),
    )
