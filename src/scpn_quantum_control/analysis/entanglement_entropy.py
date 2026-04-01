# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Entanglement Entropy
"""Entanglement entropy and Schmidt gap at the synchronization transition.

Two probes of bipartite entanglement structure:

1. Von Neumann entanglement entropy:
   S(A) = -Tr(ρ_A log₂ ρ_A)
   At a c=1 CFT critical point (BKT): S(l) = (1/3)log₂(l) + const
   (Calabrese & Cardy 2004). Does c=1 survive heterogeneous frequencies?

2. Schmidt gap:
   Δ_S = λ₁ - λ₂ (two largest Schmidt values)
   Closes at QPTs (De Chiara et al. 2012). For BKT, the closing
   follows the essential singularity of the correlation length.

Prior art:
- Nature Comms 2025: c=1 measured on IBM for homogeneous XXZ at BKT
- Tirrito et al. 2022: entanglement spectrum + ML detects BKT
- De Chiara 2012: Schmidt gap as QPT order parameter
- Nobody for heterogeneous-frequency Kuramoto-XY
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


@dataclass
class EntanglementResult:
    """Entanglement entropy + Schmidt gap at single coupling."""

    K_base: float
    entropy: float  # S(A) in bits
    schmidt_gap: float  # λ₁ - λ₂
    schmidt_values: np.ndarray  # sorted descending
    spectral_gap: float


@dataclass
class EntanglementScanResult:
    """Scan across coupling strength."""

    k_values: np.ndarray
    entropy: np.ndarray
    schmidt_gap: np.ndarray
    spectral_gap: np.ndarray
    entropy_peak_K: float | None  # K where entropy peaks
    schmidt_gap_min_K: float | None  # K where Schmidt gap is smallest


def _bipartite_entropy_and_schmidt(
    psi: np.ndarray, n: int, n_A: int
) -> tuple[float, float, np.ndarray]:
    """Compute entanglement entropy and Schmidt gap for bipartition A|B.

    A = first n_A qubits, B = remaining n-n_A qubits.
    """
    dim_A = 2**n_A
    dim_B = 2 ** (n - n_A)

    # Reshape state as matrix (dim_A × dim_B)
    psi_mat = psi.reshape(dim_A, dim_B)

    # SVD → Schmidt decomposition
    svd_vals = np.linalg.svd(psi_mat, compute_uv=False)
    svd_vals = svd_vals[svd_vals > 1e-15]  # remove numerical zeros
    svd_sq = svd_vals**2

    # Von Neumann entropy: S = -Σ λ² log₂(λ²)
    entropy = 0.0
    for p in svd_sq:
        if p > 1e-30:
            entropy -= p * np.log2(p)

    # Schmidt gap
    sorted_vals = np.sort(svd_vals)[::-1]
    if len(sorted_vals) >= 2:
        schmidt_gap = float(sorted_vals[0] - sorted_vals[1])
    else:
        schmidt_gap = float(sorted_vals[0]) if len(sorted_vals) > 0 else 0.0

    return float(entropy), schmidt_gap, sorted_vals


def entanglement_at_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    K_base: float,
) -> EntanglementResult:
    """Compute entanglement entropy and Schmidt gap at given coupling.

    Uses half-chain bipartition (n_A = n//2).
    """
    n = len(omega)
    n_A = n // 2
    if n_A == 0:
        n_A = 1

    K = K_base * K_topology
    H_mat = knm_to_dense_matrix(K, omega)
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0 = eigenvectors[:, 0]
    gap = float(eigenvalues[1] - eigenvalues[0])

    S, sg, sv = _bipartite_entropy_and_schmidt(psi0, n, n_A)

    return EntanglementResult(
        K_base=K_base,
        entropy=S,
        schmidt_gap=sg,
        schmidt_values=sv,
        spectral_gap=gap,
    )


def entanglement_vs_coupling(
    omega: np.ndarray,
    K_topology: np.ndarray,
    k_range: np.ndarray | None = None,
) -> EntanglementScanResult:
    """Scan entanglement entropy and Schmidt gap across coupling.

    At K_c: entropy should peak (log scaling), Schmidt gap should close.
    JAX GPU fast path when available (vectorised scan via jax.vmap).
    """
    if k_range is None:
        k_range = np.linspace(0.5, 5.0, 20)

    # JAX GPU fast path: entire scan as one GPU kernel
    try:
        from ..hardware.jax_accel import entanglement_scan_jax, is_jax_gpu_available

        if is_jax_gpu_available():
            jax_result = entanglement_scan_jax(K_topology, omega, k_range)
            entropy = jax_result["entropy"]
            schmidt_gap = jax_result["schmidt_gap"]
            spec_gap = jax_result["spectral_gap"]
            entropy_peak = float(k_range[int(np.argmax(entropy))]) if np.max(entropy) > 0 else None
            sg_min = float(k_range[int(np.argmin(schmidt_gap))])
            return EntanglementScanResult(
                k_values=k_range,
                entropy=entropy,
                schmidt_gap=schmidt_gap,
                spectral_gap=spec_gap,
                entropy_peak_K=entropy_peak,
                schmidt_gap_min_K=sg_min,
            )
    except (ImportError, RuntimeError):
        pass

    n_k = len(k_range)
    entropy = np.zeros(n_k)
    schmidt_gap = np.zeros(n_k)
    spec_gap = np.zeros(n_k)

    for idx, kb in enumerate(k_range):
        result = entanglement_at_coupling(omega, K_topology, float(kb))
        entropy[idx] = result.entropy
        schmidt_gap[idx] = result.schmidt_gap
        spec_gap[idx] = result.spectral_gap

    entropy_peak = float(k_range[int(np.argmax(entropy))]) if np.max(entropy) > 0 else None
    sg_min = float(k_range[int(np.argmin(schmidt_gap))])

    return EntanglementScanResult(
        k_values=k_range,
        entropy=entropy,
        schmidt_gap=schmidt_gap,
        spectral_gap=spec_gap,
        entropy_peak_K=entropy_peak,
        schmidt_gap_min_K=sg_min,
    )
