# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Magic Nonstabilizerness
"""Exact small-system stabilizer Rényi-2 diagnostics.

Stabilizer Rényi Entropy M_n measures how far a state is from the
set of stabilizer states (classically simulable via Clifford circuits).

M_2(|ψ⟩) = -log₂(Σ_P ⟨ψ|P|ψ⟩⁴ / 2^n) - n

where the sum is over all n-qubit Pauli strings P (4^n terms).

The implementation enumerates all ``4**n`` Pauli strings and is therefore a
bounded exact diagnostic. A maximum in a finite coupling scan is not, by
itself, a critical-point estimator, a fault-tolerant resource-cost certificate,
or evidence of classical hardness or quantum advantage.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation


@dataclass
class MagicResult:
    """Single-coupling pure-state stabilizer Rényi-2 result."""

    K_base: float
    sre_m2: float  # Stabilizer Rényi Entropy M_2
    xi_sum: float  # Σ_P ⟨P⟩⁴ (raw fourth-moment sum)
    n_qubits: int


@dataclass
class MagicScanResult:
    """Finite coupling scan of stabilizer Rényi-2 values.

    ``peak_K`` is the maximum on the supplied grid only; it is not a certified
    phase-transition location.
    """

    k_values: NDArray[np.float64]
    sre_m2: NDArray[np.float64]  # M_2 at each K
    peak_K: float | None  # K where M_2 is maximum
    peak_magic: float


def _compute_sre_m2(psi: NDArray[np.complex128], n: int) -> tuple[float, float]:
    """Compute Stabilizer Rényi Entropy M_2.

    M_2 = -log₂(Ξ) - n  where Ξ = Σ_P ⟨ψ|P|ψ⟩⁴ / 2^n

    Sum over all 4^n Pauli strings. Tractable for n ≤ 5.
    """
    sv = Statevector(np.ascontiguousarray(psi))
    paulis = ["I", "X", "Y", "Z"]

    xi_sum = 0.0
    for combo in iterproduct(paulis, repeat=n):
        label = "".join(combo)
        exp_val = float(sv.expectation_value(SparsePauliOp(label)).real)
        xi_sum += exp_val**4

    xi_normalized = xi_sum / (2**n)

    if xi_normalized < 1e-30:
        sre = float(n)  # maximum magic
    else:
        sre = -np.log2(xi_normalized)

    return float(sre), xi_sum


def magic_at_coupling(
    omega: NDArray[np.float64],
    K_topology: NDArray[np.float64],
    K_base: float,
    *,
    max_dense_gib: float | None = None,
) -> MagicResult:
    """Compute exact ground-state SRE ``M_2`` at one coupling.

    Parameters
    ----------
    omega, K_topology, K_base
        Frequency vector, topology matrix, and scalar coupling multiplier.
    max_dense_gib
        Optional fail-closed budget for dense eigensolver allocations.

    Returns
    -------
    MagicResult
        Exact small-system SRE value and raw Pauli fourth-moment sum.

    Notes
    -----
    Runtime grows exponentially through both dense diagonalization and Pauli
    enumeration. The result is a resource diagnostic, not a criticality or
    advantage certificate.

    """
    n = len(omega)
    K = K_base * K_topology
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=3,
        max_gib=max_dense_gib,
        label="magic dense eigensolver workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0 = eigenvectors[:, 0]

    sre, xi = _compute_sre_m2(psi0, n)

    return MagicResult(
        K_base=K_base,
        sre_m2=sre,
        xi_sum=xi,
        n_qubits=n,
    )


def magic_vs_coupling(
    omega: NDArray[np.float64],
    K_topology: NDArray[np.float64],
    k_range: NDArray[np.float64] | None = None,
    *,
    max_dense_gib: float | None = None,
) -> MagicScanResult:
    """Scan exact small-system non-stabilizerness on a finite coupling grid.

    ``peak_K`` reports only the grid argmax. Interpreting it as a critical point
    requires a separate preregistered finite-size and uncertainty study.
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15, dtype=np.float64)

    n_k = len(k_range)
    sre = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        result = magic_at_coupling(
            omega,
            K_topology,
            float(kb),
            max_dense_gib=max_dense_gib,
        )
        sre[idx] = result.sre_m2

    peak_idx = int(np.argmax(sre))
    peak_K = float(k_range[peak_idx]) if sre[peak_idx] > 0 else None

    return MagicScanResult(
        k_values=k_range,
        sre_m2=sre,
        peak_K=peak_K,
        peak_magic=float(sre[peak_idx]),
    )
