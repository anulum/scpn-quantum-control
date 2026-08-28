# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Quantum Phi
"""Minimum bipartite quantum mutual information from a density matrix.

This legacy module was named ``quantum_phi``. The implemented quantity is
quantum mutual information minimized over non-trivial bipartitions:

For a bipartition (A, B) of n qubits:
    Φ(A, B) = S(ρ_AB || ρ_A ⊗ ρ_B)
            = S(ρ_A) + S(ρ_B) - S(ρ_AB)

where S is the von Neumann entropy. This equals the quantum mutual
information I(A:B).

Minimum bipartite QMI:
    I_min = min_{(A,B)} I(A:B)

The result is a finite-state correlation diagnostic. It is not Integrated
Information Theory Φ: no causal model, intervention repertoire, cause-effect
structure, or IIT exclusion/composition calculation is implemented. It must
not be used as evidence about consciousness, sentience, cognition, or clinical
state. Legacy ``phi_*`` field and function names remain for serialization and
import compatibility only.

No peak or criticality claim is made. Such a claim requires a preregistered
finite-size study with a named null model and uncertainty analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from ..hardware.classical import classical_exact_diag


@dataclass
class PhiResult:
    """Legacy-named minimum bipartite mutual-information result.

    Attributes
    ----------
    phi_quantum
        Minimum bipartite QMI in bits. The field name is compatibility-only and
        does not denote IIT Φ.
    phi_max
        Maximum bipartite QMI in bits across enumerated partitions.
    n_qubits
        Number of qubits in the exact ground state.
    n_bipartitions
        Number of unique non-trivial bipartitions evaluated.
    mip_partition
        Partition attaining the minimum QMI.
    mutual_info_per_partition
        QMI value in bits for each enumerated partition.
    total_entropy
        Von Neumann entropy of the full ground-state density matrix.

    """

    phi_quantum: float  # minimum mutual information over bipartitions
    phi_max: float  # maximum mutual information
    n_qubits: int
    n_bipartitions: int
    mip_partition: tuple[list[int], list[int]]  # minimum information partition
    mutual_info_per_partition: list[float]
    total_entropy: float


def von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
    """Von Neumann entropy S(ρ) = -Tr(ρ log ρ).

    Parameters
    ----------
    rho
        Density matrix. The caller is responsible for physical-state
        validation.

    Returns
    -------
    float
        Base-2 entropy in bits. Eigenvalues at or below ``1e-15`` are omitted.

    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def partial_trace(
    rho: NDArray[np.complex128], keep: list[int], n_qubits: int
) -> NDArray[np.complex128]:
    """Trace out every qubit not listed in ``keep``.

    Parameters
    ----------
    rho
        ``2**n_qubits × 2**n_qubits`` density matrix.
    keep
        Zero-indexed qubits retained in the reduced state.
    n_qubits
        Total number of qubits.

    Returns
    -------
    numpy.ndarray
        Reduced density matrix in the retained subsystem ordering.

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
    rho: NDArray[np.complex128],
    subsystem_a: list[int],
    subsystem_b: list[int],
    n_qubits: int,
) -> float:
    """Compute ``I(A:B) = S(A) + S(B) - S(AB)``.

    Parameters
    ----------
    rho
        Full density matrix.
    subsystem_a, subsystem_b
        Qubit indices defining the bipartition.
    n_qubits
        Total number of qubits represented by ``rho``.

    Returns
    -------
    float
        Quantum mutual information in bits.

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
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
) -> PhiResult:
    """Compute legacy-named minimum bipartite QMI for an exact ground state.

    Parameters
    ----------
    K, omega
        Coupling matrix and natural-frequency vector passed to the dense exact
        Kuramoto-XY diagonalizer.

    Returns
    -------
    PhiResult
        Minimum/maximum bipartite QMI and partition metadata. Despite the
        compatibility type name, the result is not IIT Φ.

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
    omega: NDArray[np.float64],
    k_base_values: NDArray[np.float64] | None = None,
) -> dict[str, list[float]]:
    """Scan the legacy-named minimum-QMI diagnostic over coupling.

    The finite scan does not locate or certify a critical point. Returned
    ``phi_*`` keys are compatibility names for bipartite QMI values.
    """
    from ..bridge.knm_hamiltonian import build_knm_paper27

    if k_base_values is None:
        k_base_values = np.linspace(0.01, 2.0, 20, dtype=np.float64)

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
