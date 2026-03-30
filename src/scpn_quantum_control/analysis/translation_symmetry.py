# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Translation Symmetry for Periodic Chains
"""Translation symmetry exploitation for homogeneous Kuramoto-XY chains.

When ω_i = ω for all i AND K is translation-invariant (K_ij = K(|i-j| mod N)),
the Hamiltonian commutes with the cyclic shift operator T: |b_0 b_1...b_{N-1}⟩ → |b_{N-1} b_0...b_{N-2}⟩.

Eigenstates of T have definite crystal momentum k = 2πm/N (m = 0,...,N-1).
Combined with U(1), the Hilbert space splits into N × (N+1) sectors.

For a homogeneous N=16 chain:
  Full: 65,536 states
  U(1) M=0: 12,870 states
  U(1) M=0 + momentum k=0: ~805 states → 160× reduction

This module is only applicable when frequencies are homogeneous.
For heterogeneous ω (the SCPN case), translation is broken.

Inspired by QuSpin (Weinberg & Bukov, SciPost 2017).
"""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


def _cyclic_shift(k: int, n: int) -> int:
    """Apply cyclic left shift T to basis state |k⟩: bit 0 → bit N-1."""
    msb = (k >> (n - 1)) & 1
    return ((k << 1) & ((1 << n) - 1)) | msb


def is_translation_invariant(K: np.ndarray, omega: np.ndarray, tol: float = 1e-10) -> bool:
    """Check whether system has cyclic translation symmetry.

    Requires: K_ij = K(|i-j| mod N) and ω_i = ω for all i.
    """
    n = K.shape[0]

    # Check homogeneous frequencies
    if np.std(omega) > tol:
        return False

    # Check K is circulant
    for i in range(n):
        for j in range(n):
            d1 = (j - i) % n
            if abs(K[i, j] - K[0, d1]) > tol:
                return False
    return True


def momentum_sectors(n: int) -> dict[int, list[int]]:
    """Find orbits of the cyclic shift operator and assign momentum labels.

    Returns dict mapping momentum index m → list of representative basis states.
    Each orbit of T has length dividing N.
    """
    dim = 2**n
    visited = set()
    orbits: dict[int, list[list[int]]] = {m: [] for m in range(n)}

    for k in range(dim):
        if k in visited:
            continue
        orbit = []
        state = k
        for _ in range(n):
            orbit.append(state)
            visited.add(state)
            state = _cyclic_shift(state, n)
            if state == k:
                break

        # Orbit length divides N
        L = len(orbit)
        # This orbit contributes to momenta k = 2πm/N where m satisfies T^L = 1
        # i.e., m must be a multiple of N/L
        step = n // L
        for m in range(0, n, step):
            orbits[m].append(orbit)

    # Count states per momentum sector
    result: dict[int, list[int]] = {}
    for m, orbit_list in orbits.items():
        representatives = [orb[0] for orb in orbit_list]
        result[m] = representatives

    return result


def momentum_sector_dimensions(n: int) -> dict[int, int]:
    """Return dimension of each momentum sector."""
    sectors = momentum_sectors(n)
    return {m: len(reps) for m, reps in sectors.items()}


def eigh_with_translation(
    K: np.ndarray,
    omega: np.ndarray,
    momentum: int = 0,
) -> dict:
    """Diagonalise in a specific momentum sector using Bloch's theorem.

    For the k=0 (totally symmetric) sector, the projected Hamiltonian
    acts on symmetrised basis states |φ_α⟩ = (1/√L) Σ_{r=0}^{L-1} T^r |α⟩.

    Parameters
    ----------
    K, omega : coupling and frequencies (must be translation-invariant)
    momentum : momentum quantum number m ∈ {0, 1, ..., N-1}

    Returns
    -------
    dict with: eigvals, dim, momentum, is_ti
    """
    n = K.shape[0]

    if not is_translation_invariant(K, omega):
        raise ValueError(
            "System is not translation-invariant. "
            "Translation symmetry requires homogeneous ω and circulant K."
        )

    H_full = knm_to_dense_matrix(K, omega)
    dim = 2**n
    phase = np.exp(2j * np.pi * momentum / n)

    # Build projection matrix: rows are symmetrised basis states
    sectors = momentum_sectors(n)
    reps = sectors.get(momentum, [])

    if not reps:
        return {"eigvals": np.array([]), "dim": 0, "momentum": momentum, "is_ti": True}

    # For each orbit representative, build the Bloch state
    proj_rows = []
    for alpha in reps:
        orbit = []
        state = alpha
        for _ in range(n):
            orbit.append(state)
            state = _cyclic_shift(state, n)
            if state == alpha:
                break
        L = len(orbit)
        row = np.zeros(dim, dtype=np.complex128)
        for r, s in enumerate(orbit):
            row[s] = phase**r / np.sqrt(L)
        proj_rows.append(row)

    P = np.array(proj_rows)  # shape (dim_sector, dim_full)
    H_sector = P @ H_full @ P.conj().T

    # H_sector should be Hermitian
    H_sector = (H_sector + H_sector.conj().T) / 2

    eigvals = np.linalg.eigvalsh(H_sector.real if np.allclose(H_sector.imag, 0) else H_sector)

    return {
        "eigvals": eigvals,
        "dim": len(reps),
        "momentum": momentum,
        "is_ti": True,
    }
