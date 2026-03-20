# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Quantum integrated information (Φ) from density matrix.

Integrated Information Theory (IIT, Tononi 2004) quantifies how much
a system's whole exceeds the sum of its parts. The quantum extension
(Zanardi et al. 2018, PRE 97 042112) defines quantum Φ via the
distance between the full density matrix and the product of its
reduced subsystem density matrices.

For a bipartition (A, B) of n qubits:
    Φ(A, B) = S(ρ_AB || ρ_A ⊗ ρ_B)
            = S(ρ_A) + S(ρ_B) - S(ρ_AB)

where S is the von Neumann entropy. This equals the quantum mutual
information I(A:B).

Quantum Φ (minimum over all bipartitions):
    Φ_Q = min_{(A,B)} I(A:B)

This is the "minimum information partition" — the bipartition
where the system is most independent. A high Φ_Q means no way
to split the system without losing information.

Connection to SCPN: Φ_Q at the synchronization transition
should peak (maximum integration at criticality).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ..hardware.classical import classical_exact_diag


@dataclass
class PhiResult:
    """Quantum integrated information result."""

    phi_quantum: float  # minimum mutual information over bipartitions
    phi_max: float  # maximum mutual information
    n_qubits: int
    n_bipartitions: int
    mip_partition: tuple[list[int], list[int]]  # minimum information partition
    mutual_info_per_partition: list[float]
    total_entropy: float


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Uses eigenvalue decomposition to avoid log(0).
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def partial_trace(rho: np.ndarray, keep: list[int], n_qubits: int) -> np.ndarray:
    """Trace out all qubits NOT in `keep` list.

    Args:
        rho: 2^n × 2^n density matrix
        keep: list of qubit indices to keep (0-indexed)
        n_qubits: total number of qubits
    """
    dims = [2] * n_qubits
    rho_tensor = rho.reshape(dims + dims)

    trace_out = sorted(set(range(n_qubits)) - set(keep))

    # Trace out qubits from highest index to lowest
    for q in reversed(trace_out):
        # Contract axis q with axis q + n_remaining
        n_remaining = rho_tensor.ndim // 2
        rho_tensor = np.trace(rho_tensor, axis1=q, axis2=q + n_remaining)

    n_keep = len(keep)
    d = 2**n_keep
    return rho_tensor.reshape(d, d)


def mutual_information(
    rho: np.ndarray,
    subsystem_a: list[int],
    subsystem_b: list[int],
    n_qubits: int,
) -> float:
    """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

    Returns mutual information in bits (log base 2).
    """
    rho_a = partial_trace(rho, subsystem_a, n_qubits)
    rho_b = partial_trace(rho, subsystem_b, n_qubits)

    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    s_ab = von_neumann_entropy(rho)

    return float(s_a + s_b - s_ab)


def _all_bipartitions(n: int) -> list[tuple[list[int], list[int]]]:
    """Generate all non-trivial bipartitions of n qubits.

    Each bipartition splits {0,...,n-1} into two non-empty subsets.
    Only generates partitions where min(|A|) <= n//2 to avoid duplicates.
    """
    qubits = list(range(n))
    partitions: list[tuple[list[int], list[int]]] = []
    for k in range(1, n // 2 + 1):
        for combo in combinations(qubits, k):
            a = list(combo)
            b = [q for q in qubits if q not in combo]
            if k == n // 2 and a > b:
                continue  # avoid duplicate (A,B) = (B,A) when |A|=|B|
            partitions.append((a, b))
    return partitions


def compute_quantum_phi(
    K: np.ndarray,
    omega: np.ndarray,
) -> PhiResult:
    """Compute quantum Φ from ground state of K_nm Hamiltonian.

    Finds the minimum information partition (MIP) — the bipartition
    where the system is most separable.
    """
    n = K.shape[0]
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]

    rho = np.outer(psi, psi.conj())
    s_total = von_neumann_entropy(rho)

    partitions = _all_bipartitions(n)
    mi_values: list[float] = []

    for a, b in partitions:
        mi = mutual_information(rho, a, b, n)
        mi_values.append(mi)

    phi_min = min(mi_values) if mi_values else 0.0
    phi_max = max(mi_values) if mi_values else 0.0
    mip_idx = mi_values.index(phi_min) if mi_values else 0
    mip = partitions[mip_idx] if partitions else ([], [])

    return PhiResult(
        phi_quantum=phi_min,
        phi_max=phi_max,
        n_qubits=n,
        n_bipartitions=len(partitions),
        mip_partition=mip,
        mutual_info_per_partition=mi_values,
        total_entropy=s_total,
    )


def phi_vs_coupling_scan(
    omega: np.ndarray,
    k_base_values: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Scan Φ_Q vs coupling strength to find the criticality peak."""
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_base_values is None:
        k_base_values = np.linspace(0.01, 2.0, 20)

    n = len(omega)
    results: dict[str, list[float]] = {
        "k_base": [],
        "phi_quantum": [],
        "phi_max": [],
        "total_entropy": [],
    }

    for kb in k_base_values:
        K = build_knm_paper27(L=n, K_base=float(kb))
        phi = compute_quantum_phi(K, omega)
        results["k_base"].append(float(kb))
        results["phi_quantum"].append(phi.phi_quantum)
        results["phi_max"].append(phi.phi_max)
        results["total_entropy"].append(phi.total_entropy)

    return results
