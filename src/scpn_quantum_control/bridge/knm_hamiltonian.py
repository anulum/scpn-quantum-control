# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Knm Hamiltonian
"""Knm coupling matrix -> Pauli Hamiltonian compiler.

Translates the 16x16 Knm coupling matrix + 16 natural frequencies from
Paper 27 into a SparsePauliOp for quantum simulation.

Kuramoto <-> XY mapping:
  K[i,j]*sin(theta_j - theta_i)  <=>  -J_ij*(X_i X_j + Y_i Y_j)
  omega_i                         <=>  -h_i * Z_i
"""

from __future__ import annotations

from threading import Event
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from scipy import sparse

from .._constants import COUPLING_SPARSITY_EPS
from .._rust_accel import optional_rust_engine
from ..compile_budget import require_pauli_operator_budget
from ..dense_budget import require_dense_allocation
from ..execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from ..execution_reservations import reserve_execution_memory

KNM_SPARSITY_EPS = COUPLING_SPARSITY_EPS

# Paper 27, Table 1: canonical natural frequencies (rad/s)
OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ],
    dtype=np.float64,
)


def omega_for_oscillators(n_oscillators: int) -> NDArray[np.float64]:
    """Return deterministic natural frequencies for ``n_oscillators``.

    The first 16 entries are the canonical Paper 27 values in
    :data:`OMEGA_N_16`. Larger synthetic networks use a periodic extension of
    that measured table so scalable classical, co-simulation, and partitioned
    examples receive a full-length vector without fabricating new Paper 27 data.

    Parameters
    ----------
    n_oscillators:
        Number of oscillator frequencies to return; must be at least one.

    Returns
    -------
    numpy.ndarray
        A fresh ``float64`` vector of length ``n_oscillators``.

    Raises
    ------
    TypeError
        If ``n_oscillators`` is not an integer.
    ValueError
        If ``n_oscillators`` is below one.

    """
    if not isinstance(n_oscillators, int):
        raise TypeError("n_oscillators must be an integer")
    if n_oscillators < 1:
        raise ValueError("n_oscillators must be >= 1")
    if n_oscillators <= OMEGA_N_16.size:
        return OMEGA_N_16[:n_oscillators].copy()
    repeats = (n_oscillators + OMEGA_N_16.size - 1) // OMEGA_N_16.size
    return np.tile(OMEGA_N_16, repeats)[:n_oscillators].astype(np.float64, copy=True)


def build_knm_paper27(
    L: int = 16,
    K_base: float = 0.45,  # Paper 27, Eq. 3
    K_alpha: float = 0.3,  # Paper 27, Eq. 3
) -> NDArray[np.float64]:
    """Build the canonical Knm coupling matrix from Paper 27.

    K[i,j] = K_base * exp(-K_alpha * |i - j|)   (Paper 27, Eq. 3)
    with calibration anchors from Table 2 and cross-hierarchy boost constants.
    """
    idx = np.arange(L)
    K: NDArray[np.float64] = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))

    # Paper 27 Table 2 calibration anchors (only apply if indices in range)
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        if i < L and j < L:
            K[i, j] = K[j, i] = val

    # Paper 27 cross-hierarchy boosts
    if L > 15:
        K[0, 15] = K[15, 0] = max(K[0, 15], 0.05)  # L1-L16
    if L > 6:
        K[4, 6] = K[6, 4] = max(K[4, 6], 0.15)  # L5-L7

    return K


def build_kuramoto_ring(
    n: int,
    coupling: float = 1.0,
    omega: NDArray[np.float64] | None = None,
    rng_seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a nearest-neighbour ring coupling matrix for n Kuramoto oscillators.

    Returns (K, omega) ready for QuantumKuramotoSolver or knm_to_hamiltonian.
    If omega is None, draws from N(0,1) with the given seed.
    """
    K: NDArray[np.float64] = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        K[i, j] = K[j, i] = coupling
    if omega is None:
        rng = np.random.default_rng(rng_seed)
        omega = rng.standard_normal(n)
    return K, np.asarray(omega, dtype=np.float64)


def knm_to_xxz_hamiltonian(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    delta: float = 0.0,
) -> SparsePauliOp:
    """Convert Knm + frequencies to XXZ Hamiltonian with anisotropy Δ.

    H = -sum_{i<j} K[i,j] * (X_iX_j + Y_iY_j + Δ·Z_iZ_j) - sum_i omega_i * Z_i

    Δ = 0: XY model (standard Kuramoto mapping, in-plane S² dynamics)
    Δ = 1: isotropic Heisenberg (full S² dynamics, Kouchekian-Teodorescu 2025)

    The anisotropy parameter controls the off-plane spin coupling that
    the standard Kuramoto-XY mapping omits. The full Heisenberg model
    (Δ=1) corresponds to the variational S² spin formulation proven in
    arXiv:2601.00113 (Kouchekian & Teodorescu, 2025).

    At Δ=1, perturbations around equilibria connect to the semiclassical
    Gaudin model and the Richardson pairing mechanism.

    Parameters
    ----------
    K
        Real finite square coupling matrix, symmetrised on a copy.
    omega
        Real finite one-dimensional oscillator frequencies.
    delta
        Finite XXZ anisotropy; selected ZZ coefficients must remain finite.

    Returns
    -------
    SparsePauliOp
        Canonically selected XY/XXZ terms without further tolerance pruning.

    Raises
    ------
    ValueError
        Invalid shape, dtype, nonfinite input or overflowed ZZ coefficient.
    DenseAllocationError
        The declared Pauli representation exceeds the active memory budget.

    """
    n = len(omega)
    if K.ndim >= 1 and K.shape[0] != n:
        raise ValueError(f"K has {K.shape[0]} rows but omega has {n} elements")
    if omega.ndim != 1 or K.shape != (n, n):
        raise ValueError("Pauli Hamiltonian input shape must be K=(n,n), omega=(n,)")
    if K.dtype.kind not in "fi" or omega.dtype.kind not in "fi":
        raise ValueError("Pauli Hamiltonian input dtype must be real numeric")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)) or not np.isfinite(delta):
        raise ValueError("Pauli Hamiltonian inputs must be finite")

    require_pauli_operator_budget(
        n,
        include_zz=abs(delta) > KNM_SPARSITY_EPS,
        label="XY/XXZ Pauli Hamiltonian",
    )

    # Enforce symmetry (Finding #7: K Symmetry Broken by Gradient Training)
    K = 0.5 * K + 0.5 * K.T

    pauli_list = []

    for i in range(n):
        if abs(omega[i]) > KNM_SPARSITY_EPS:
            z_str = ["I"] * n
            z_str[i] = "Z"
            pauli_list.append(("".join(reversed(z_str)), -omega[i]))

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < KNM_SPARSITY_EPS:
                continue
            # XX term
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            pauli_list.append(("".join(reversed(xx)), -K[i, j]))
            # YY term
            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            pauli_list.append(("".join(reversed(yy)), -K[i, j]))
            # ZZ term (off-plane, controlled by delta)
            if abs(delta) > KNM_SPARSITY_EPS:
                zz = ["I"] * n
                zz[i] = "Z"
                zz[j] = "Z"
                with np.errstate(over="ignore", invalid="ignore"):
                    coefficient = -K[i, j] * delta
                if not np.isfinite(coefficient):
                    raise ValueError("XXZ coefficient is not finite")
                pauli_list.append(("".join(reversed(zz)), coefficient))

    if not pauli_list:
        return SparsePauliOp.from_list([("I" * n, 0.0)])
    labels, coeffs = zip(*pauli_list, strict=True)
    return SparsePauliOp(list(labels), list(coeffs)).simplify(atol=0.0, rtol=0.0)


def knm_to_hamiltonian(K: NDArray[np.float64], omega: NDArray[np.float64]) -> SparsePauliOp:
    """Convert Knm coupling matrix + natural frequencies to SparsePauliOp.

    H = -sum_{i<j} K[i,j] * (X_i X_j + Y_i Y_j) - sum_i omega_i * Z_i

    Uses Qiskit little-endian qubit ordering. Equivalent to
    ``knm_to_xxz_hamiltonian(K, omega, delta=0.0)``.
    """
    return knm_to_xxz_hamiltonian(K, omega, delta=0.0)


def knm_to_sparse_matrix(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    delta: float = 0.0,
    *,
    max_gib: float | None = None,
) -> sparse.csc_matrix:
    """Build sparse XY/XXZ Hamiltonian matrix in CSC format.

    ``SparsePauliOp.to_matrix`` materialises a ``2**n``-dimensional operator
    whose non-zero count scales as ``O(n * 2**n)``; this fails closed before
    that allocation for pathological ``n``. Pass ``max_gib`` to override the
    active dense budget for this call.
    """
    n = len(omega)
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=1,
        object_count=max(1, n),
        max_gib=max_gib,
        label="sparse XY Hamiltonian matrix (2^n nnz-scale)",
    )
    H_op = knm_to_xxz_hamiltonian(K, omega, delta)
    # to_matrix(sparse=True) returns a scipy.sparse.csr_matrix
    raw = H_op.to_matrix(sparse=True)
    return raw.tocsc()


def knm_to_dense_matrix(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    delta: float = 0.0,
    *,
    max_dense_gib: float | None = None,
    backend: Literal["auto", "python", "rust"] = "auto",
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> NDArray[np.complex128]:
    """Build dense XY Hamiltonian matrix, Rust fast path with Qiskit fallback.

    Returns complex ndarray of shape (2^n, 2^n). Admission includes the
    requested output, two matrix intermediates and input-conversion buffers.
    ``backend='rust'`` refuses missing native support or nonzero anisotropy;
    ``auto`` retains the optional Rust route with an explicit Python fallback.
    Snapshot admission does not reserve memory or bound undocumented backend
    workspaces.

    Parameters
    ----------
    K
        Real finite coupling array of shape ``(n, n)``; symmetrised on a copy.
    omega
        Real finite frequency vector of shape ``(n,)``.
    delta
        XXZ anisotropy; the native XY entry supports only zero anisotropy.
    max_dense_gib
        Optional whole declared-buffer cap constrained by live process headroom.
    backend
        Explicit Python or Rust selection, or optional native selection in auto mode.
    deadline_monotonic
        Optional absolute monotonic deadline, checked before and after dispatch.
    cancelled
        Optional caller cancellation event, checked before and after dispatch.

    Returns
    -------
    numpy.ndarray
        Complex128 dense operator in Qiskit little-endian convention.

    Raises
    ------
    ValueError
        Unsupported route, malformed input/output shape, or nonfinite numeric data.
    DenseAllocationError
        Declared execution memory or a requested native entry is unavailable.
    RuntimeError
        A previously admitted native entry disappears before dispatch.

    """
    if omega.ndim != 1:
        raise ValueError("dense Hamiltonian input shape must be K=(n,n), omega=(n,)")
    n = len(omega)
    if K.shape != (n, n):
        raise ValueError("dense Hamiltonian input shape must be K=(n,n), omega=(n,)")
    if K.dtype.kind not in "fi" or omega.dtype.kind not in "fi":
        raise ValueError("dense Hamiltonian input dtype must be real numeric")
    if not np.isfinite(delta):
        raise ValueError("dense Hamiltonian inputs must be finite")
    if backend not in ("auto", "python", "rust"):
        raise ValueError("unknown dense Hamiltonian backend")
    if backend == "rust" and delta != 0.0:
        raise ValueError("native dense Hamiltonian supports delta=0 only")
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer.hilbert("output", "dense_output", n, rank=2),
            ExecutionBuffer.hilbert("matrix_temporaries", "intermediate", n, rank=2, count=2),
            ExecutionBuffer("coupling_conversion", "intermediate", (n, n), "float64", 3),
            ExecutionBuffer("frequency_conversion", "intermediate", (n,), "float64"),
        )
    )
    with reserve_execution_memory(
        plan,
        max_gib=max_dense_gib,
        native_symbol="build_xy_hamiltonian_dense" if backend == "rust" else None,
        deadline_monotonic=deadline_monotonic,
        cancelled=cancelled,
    ) as reservation:
        reservation.checkpoint()
        # These sequential boolean masks fit within the declared conversion
        # workspace and must not be materialised before admission.
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)):
            raise ValueError("dense Hamiltonian inputs must be finite")
        reservation.checkpoint()
        # Enforce symmetry (Finding #7: K Symmetry Broken by Gradient Training)
        K = 0.5 * K + 0.5 * K.T

        # Rust engine only supports delta=0.0 for now
        if backend != "python" and delta == 0.0:
            _engine = optional_rust_engine()
            if _engine is not None and callable(
                getattr(_engine, "build_xy_hamiltonian_dense", None)
            ):
                reservation.checkpoint()
                h_flat = np.asarray(
                    _engine.build_xy_hamiltonian_dense(
                        K.ravel().astype(np.float64),
                        omega.astype(np.float64),
                        n,
                    )
                )
                reservation.checkpoint()
                if h_flat.ndim != 1 or h_flat.size != 1 << (2 * n):
                    raise ValueError("native dense Hamiltonian returned malformed output shape")
                if h_flat.dtype.kind not in "fi" or not np.all(np.isfinite(h_flat)):
                    raise ValueError("native dense Hamiltonian returned invalid numeric data")
                output = h_flat.reshape(2**n, 2**n).astype(complex)
                reservation.checkpoint()
                return output
            if backend == "rust":
                raise RuntimeError("requested native kernel became unavailable; fallback refused")

        H_op = knm_to_xxz_hamiltonian(K, omega, delta)
        reservation.checkpoint()
        H_raw = H_op.to_matrix()
        reservation.checkpoint()
        output = H_raw.toarray() if hasattr(H_raw, "toarray") else np.array(H_raw)
        if not np.all(np.isfinite(output)):
            raise ValueError("dense Hamiltonian returned invalid numeric data")
        reservation.checkpoint()
        return output


def knm_to_ansatz(
    K: NDArray[np.float64], reps: int = 2, threshold: float = 0.01
) -> QuantumCircuit:
    """Build physics-informed ansatz: CZ entanglement only between Knm-connected pairs.

    Pattern from QUANTUM_LAB script 16 (PhysicsInformedAnsatz).
    """
    n = K.shape[0]
    params = ParameterVector("p", n * 2 * reps)
    qc = QuantumCircuit(n)

    idx = 0
    for _ in range(reps):
        for q in range(n):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n):
            qc.rz(params[idx], q)
            idx += 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(K[i, j]) >= threshold:
                    qc.cz(i, j)

    return qc
