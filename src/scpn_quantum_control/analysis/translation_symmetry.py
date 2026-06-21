# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Translation Symmetry for Periodic Chains
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

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ..bridge.sparse_hamiltonian import build_sparse_hamiltonian
from ..dense_budget import GIB, DenseAllocationError, dense_budget_bytes


def _cyclic_shift(k: int, n: int) -> int:
    """Apply cyclic left shift T to basis state |k⟩: bit 0 → bit N-1."""
    msb = (k >> (n - 1)) & 1
    return ((k << 1) & ((1 << n) - 1)) | msb


def is_translation_invariant(
    K: NDArray[np.float64], omega: NDArray[np.float64], tol: float = 1e-10
) -> bool:
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


def _require_momentum_sector_budget(
    sector_dim: int,
    *,
    max_dense_gib: float | None,
    object_count: int = 2,
) -> None:
    """Guard dense sector eigensolver workspace for a projected momentum block."""
    if sector_dim < 1:
        return
    bytes_required = sector_dim * sector_dim * np.dtype(np.complex128).itemsize * object_count
    budget = dense_budget_bytes(max_dense_gib)
    if bytes_required > budget:
        raise DenseAllocationError(
            "translation momentum sector dense eigensolver workspace "
            f"for dim={sector_dim} requires {bytes_required / GIB:.2f} GiB "
            f"for {object_count} objects of shape ({sector_dim}, {sector_dim}) "
            f"(complex128), above the active dense budget {budget / GIB:.2f} GiB. "
            "Use a sparse or matrix-free eigensolver for this momentum sector."
        )


def _bloch_projector(n: int, reps: list[int], phase: complex) -> sparse.csr_matrix:
    """Build a sparse row projector onto translation-momentum Bloch states."""
    dim = 2**n
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    for row_idx, alpha in enumerate(reps):
        orbit = []
        state = alpha
        for _ in range(n):
            orbit.append(state)
            state = _cyclic_shift(state, n)
            if state == alpha:
                break
        norm = np.sqrt(len(orbit))
        for r, basis_state in enumerate(orbit):
            rows.append(row_idx)
            cols.append(basis_state)
            data.append(phase**r / norm)

    return sparse.csr_matrix((data, (rows, cols)), shape=(len(reps), dim), dtype=np.complex128)


def eigh_with_translation(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    momentum: int = 0,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Diagonalise in a specific momentum sector using Bloch's theorem.

    For the k=0 (totally symmetric) sector, the projected Hamiltonian
    acts on symmetrised basis states |φ_α⟩ = (1/√L) Σ_{r=0}^{L-1} T^r |α⟩.

    Parameters
    ----------
    K, omega : coupling and frequencies (must be translation-invariant)
    momentum : momentum quantum number m ∈ {0, 1, ..., N-1}
    max_dense_gib : optional GiB budget for the projected dense eigensolver

    Returns
    -------
    dict with: eigvals, dim, momentum, is_ti
    """
    n = K.shape[0]
    if isinstance(momentum, bool) or not isinstance(momentum, int):
        raise ValueError("momentum must be an integer in the range 0 <= momentum < n.")
    if momentum < 0 or momentum >= n:
        raise ValueError(f"momentum must satisfy 0 <= momentum < {n}; got {momentum}.")

    if not is_translation_invariant(K, omega):
        raise ValueError(
            "System is not translation-invariant. "
            "Translation symmetry requires homogeneous ω and circulant K."
        )

    phase = np.exp(2j * np.pi * momentum / n)

    sectors = momentum_sectors(n)
    reps = sectors.get(momentum, [])

    if not reps:
        return {"eigvals": np.array([]), "dim": 0, "momentum": momentum, "is_ti": True}

    _require_momentum_sector_budget(len(reps), max_dense_gib=max_dense_gib)
    H_sparse = build_sparse_hamiltonian(K, omega)
    P = _bloch_projector(n, reps, phase)
    H_sector = (P @ H_sparse @ P.conj().T).toarray()

    # H_sector should be Hermitian
    H_sector = (H_sector + H_sector.conj().T) / 2

    eigvals = np.linalg.eigvalsh(H_sector.real if np.allclose(H_sector.imag, 0) else H_sector)

    return {
        "eigvals": eigvals,
        "dim": len(reps),
        "momentum": momentum,
        "is_ti": True,
    }
