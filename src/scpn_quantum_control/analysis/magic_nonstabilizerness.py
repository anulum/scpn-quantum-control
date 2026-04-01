# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Magic Nonstabilizerness
"""Magic (non-stabilizerness) at the synchronization transition.

Stabilizer Rényi Entropy M_n measures how far a state is from the
set of stabilizer states (classically simulable via Clifford circuits).

M_2(|ψ⟩) = -log₂(Σ_P ⟨ψ|P|ψ⟩⁴ / 2^n) - n

where the sum is over all n-qubit Pauli strings P (4^n terms).

At a QPT: magic typically peaks at criticality (the critical state
is maximally non-classical). For BKT (infinite-order): the scaling
of magic is unknown — the infinite-order character may produce
different behavior from the power-law peaks seen at 2nd-order QPTs.

Prior art: Tarabunga et al. 2024 (magic at QPTs in XXZ chain, but
Ising-type transitions, not BKT). Hoshino et al. 2025 (SRE + CFT
for c=1/2 Ising). Nobody for c=1 BKT or Kuramoto.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as iterproduct

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian


@dataclass
class MagicResult:
    """Magic / non-stabilizerness result."""

    K_base: float
    sre_m2: float  # Stabilizer Rényi Entropy M_2
    xi_sum: float  # Σ_P ⟨P⟩⁴ (raw fourth-moment sum)
    n_qubits: int


@dataclass
class MagicScanResult:
    """Magic scan across coupling strength."""

    k_values: np.ndarray
    sre_m2: np.ndarray  # M_2 at each K
    peak_K: float | None  # K where M_2 is maximum
    peak_magic: float


def _compute_sre_m2(psi: np.ndarray, n: int) -> tuple[float, float]:
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
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
) -> MagicResult:
    """Compute SRE M_2 of ground state at given coupling."""
    n = len(omega)
    K = K_base * K_topology
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega)
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
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
) -> MagicScanResult:
    """Scan magic across coupling to find non-stabilizerness peak.

    At K_c: magic should peak (maximally non-classical critical state).
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 15)

    n_k = len(k_range)
    sre = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        result = magic_at_coupling(omega, K_topology, float(kb))
        sre[idx] = result.sre_m2

    peak_idx = int(np.argmax(sre))
    peak_K = float(k_range[peak_idx]) if sre[peak_idx] > 0 else None

    return MagicScanResult(
        k_values=k_range,
        sre_m2=sre,
        peak_K=peak_K,
        peak_magic=float(sre[peak_idx]),
    )
