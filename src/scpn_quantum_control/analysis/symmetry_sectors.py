# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Z2 Parity Sector Decomposition
"""Symmetry-aware exact diagonalisation for the XY Hamiltonian.

The heterogeneous Kuramoto-XY Hamiltonian H = -Σ K_ij(X_iX_j + Y_iY_j) - Σ ω_i Z_i
commutes with the global Z2 parity operator P = Z_1 ⊗ Z_2 ⊗ ... ⊗ Z_N.
This means every eigenstate has definite parity (even or odd number of excitations).

Exploiting this halves the Hilbert space dimension for ED:
  Full space: 2^N states
  Each sector: 2^(N-1) states

For N=16: full ED needs 65536x65536 matrix (32 GB).
With parity: two 32768x32768 matrices (8 GB each) — fits in 32 GB RAM.

Inspired by QuSpin's symmetry handling (Weinberg & Bukov, SciPost 2017).
"""

from __future__ import annotations

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


def _parity(k: int, n: int) -> int:
    """Parity of basis state |k>: 0 if even number of 1-bits, 1 if odd."""
    return bin(k).count("1") % 2


def basis_indices_by_parity(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Split computational basis into even and odd parity sectors.

    Returns (even_indices, odd_indices) where each is a sorted array
    of basis state indices.
    """
    dim = 2**n
    even = []
    odd = []
    for k in range(dim):
        if _parity(k, n) == 0:
            even.append(k)
        else:
            odd.append(k)
    return np.array(even, dtype=np.intp), np.array(odd, dtype=np.intp)


def project_hamiltonian(H: np.ndarray, sector_indices: np.ndarray) -> np.ndarray:
    """Project full Hamiltonian onto a parity sector.

    H_sector[i,j] = H[sector_indices[i], sector_indices[j]]
    """
    return np.asarray(H[np.ix_(sector_indices, sector_indices)])


def build_sector_hamiltonian(
    K: np.ndarray, omega: np.ndarray, parity: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build the XY Hamiltonian projected onto a parity sector.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.
    parity : int
        0 for even sector, 1 for odd sector.

    Returns
    -------
    H_sector : array (dim/2, dim/2)
        Hamiltonian in the parity sector.
    sector_indices : array
        Mapping from sector index to full basis index.
    """
    n = K.shape[0]
    even_idx, odd_idx = basis_indices_by_parity(n)
    sector_indices = even_idx if parity == 0 else odd_idx

    H_full = knm_to_dense_matrix(K, omega)
    H_sector = project_hamiltonian(H_full, sector_indices)
    return H_sector, sector_indices


def eigh_by_sector(K: np.ndarray, omega: np.ndarray) -> dict:
    """Diagonalise both parity sectors separately.

    Returns dict with keys:
        eigvals_even, eigvecs_even, indices_even,
        eigvals_odd, eigvecs_odd, indices_odd,
        eigvals_all (sorted), ground_energy, ground_parity
    """
    n = K.shape[0]
    even_idx, odd_idx = basis_indices_by_parity(n)
    H_full = knm_to_dense_matrix(K, omega)

    H_even = project_hamiltonian(H_full, even_idx)
    H_odd = project_hamiltonian(H_full, odd_idx)

    vals_e, vecs_e = np.linalg.eigh(H_even)
    vals_o, vecs_o = np.linalg.eigh(H_odd)

    all_vals = np.sort(np.concatenate([vals_e, vals_o]))

    ground_parity = 0 if vals_e[0] <= vals_o[0] else 1
    ground_energy = min(vals_e[0], vals_o[0])

    return {
        "eigvals_even": vals_e,
        "eigvecs_even": vecs_e,
        "indices_even": even_idx,
        "eigvals_odd": vals_o,
        "eigvecs_odd": vecs_o,
        "indices_odd": odd_idx,
        "eigvals_all": all_vals,
        "ground_energy": float(ground_energy),
        "ground_parity": ground_parity,
    }


def level_spacing_by_sector(K: np.ndarray, omega: np.ndarray) -> dict:
    """Level-spacing ratio r̄ computed within each parity sector.

    This avoids mixing even/odd spectra which would artificially
    give Poisson statistics (two independent spectra overlaid always
    look integrable).
    """
    result = eigh_by_sector(K, omega)

    def _r_bar(eigvals: np.ndarray) -> float:
        gaps = np.diff(eigvals)
        gaps = gaps[gaps > 1e-14]  # skip degeneracies
        if len(gaps) < 2:
            return float("nan")
        ratios = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
        return float(np.mean(ratios))

    r_even = _r_bar(result["eigvals_even"])
    r_odd = _r_bar(result["eigvals_odd"])

    return {
        "r_bar_even": r_even,
        "r_bar_odd": r_odd,
        "r_bar_combined": (r_even + r_odd) / 2
        if not (np.isnan(r_even) or np.isnan(r_odd))
        else float("nan"),
        "ground_energy": result["ground_energy"],
        "ground_parity": result["ground_parity"],
        "dim_per_sector": len(result["indices_even"]),
    }


def memory_estimate_mb(n: int, use_sectors: bool = True) -> float:
    """Estimate memory for ED in MB (float64 complex)."""
    if use_sectors:
        dim = 2 ** (n - 1)
    else:
        dim = 2**n
    return float(dim * dim * 16 / 1e6)  # complex128 = 16 bytes
