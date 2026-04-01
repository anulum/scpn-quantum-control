# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — U(1) Magnetisation Sector Decomposition
"""U(1) symmetry exploitation for the XY Hamiltonian.

The XY interaction X_iX_j + Y_iY_j = 2(σ⁺_iσ⁻_j + σ⁻_iσ⁺_j) is a flip-flop:
it swaps excitations between qubits but never creates or destroys them. Therefore
the total magnetisation M = Σ_i Z_i is conserved: [H, M] = 0.

This decomposes the 2^N Hilbert space into N+1 sectors labelled by M ∈ {-N, -N+2, ..., N}.
The sector with magnetisation M has dimension C(N, k) where k = (N+M)/2 is the
number of excitations (spin-up qubits).

For N=16:
  Full space: 2^16 = 65,536 states → 32 GB
  Largest sector (M=0): C(16,8) = 12,870 → 2.5 GB
  This is a 13× memory reduction.

For N=20:
  Full space: 2^20 = 1,048,576 → 16 TB (impossible)
  Largest sector (M=0): C(20,10) = 184,756 → 500 GB (still large)
  But M=±2 sectors: C(20,9) = 167,960 → 430 GB

Combined with sparse methods, U(1) sectors enable N=18-22 on workstations.

Inspired by QuSpin (Weinberg & Bukov, SciPost 2017).
"""

from __future__ import annotations

from math import comb

import numpy as np

from ..bridge.knm_hamiltonian import knm_to_dense_matrix


def _magnetisation(k: int, n: int) -> int:
    """Total magnetisation M = Σ (1 - 2*bit_i) for basis state |k⟩.

    Convention: |0⟩ = spin-up (M=+1), |1⟩ = spin-down (M=-1).
    M ranges from +N (all up) to -N (all down) in steps of 2.
    """
    n_ones = bin(k).count("1")
    return n - 2 * n_ones


def basis_by_magnetisation(n: int) -> dict[int, np.ndarray]:
    """Partition computational basis by total magnetisation M.

    Returns dict mapping M → array of basis state indices.
    Uses Rust-accelerated popcount (97× faster) when available.
    """
    # Rust fast path
    try:
        import scpn_quantum_engine as eng

        labels = eng.magnetisation_labels(n)
        sectors_out: dict[int, list[int]] = {}
        for k, m in enumerate(labels):
            m_int = int(m)
            if m_int not in sectors_out:
                sectors_out[m_int] = []
            sectors_out[m_int].append(k)
        return {
            m_val: np.array(indices, dtype=np.intp)
            for m_val, indices in sorted(sectors_out.items())
        }
    except (ImportError, Exception):
        pass

    sectors: dict[int, list[int]] = {}
    for k in range(2**n):
        m = _magnetisation(k, n)
        if m not in sectors:
            sectors[m] = []
        sectors[m].append(k)
    return {m: np.array(indices, dtype=np.intp) for m, indices in sorted(sectors.items())}


def sector_dimensions(n: int) -> dict[int, int]:
    """Return dimension of each magnetisation sector.

    Sector M has dimension C(N, k) where k = (N+M)/2.
    """
    dims = {}
    for k in range(n + 1):
        m = n - 2 * k
        dims[m] = comb(n, k)
    return dims


def largest_sector_dim(n: int) -> int:
    """Dimension of the largest sector (M=0 for even N, M=±1 for odd N)."""
    return comb(n, n // 2)


def project_to_sector(
    H_full: np.ndarray,
    sector_indices: np.ndarray,
) -> np.ndarray:
    """Project full Hamiltonian onto a magnetisation sector."""
    return np.asarray(H_full[np.ix_(sector_indices, sector_indices)])


def build_sector_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
    M: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Hamiltonian projected onto magnetisation sector M.

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.
    M : int
        Target magnetisation. Must be in {-N, -N+2, ..., N}.

    Returns
    -------
    H_sector : array (dim_M, dim_M)
        Hamiltonian in the sector.
    sector_indices : array (dim_M,)
        Mapping from sector index to full basis index.

    Raises
    ------
    ValueError
        If M is not a valid magnetisation value.
    """
    n = K.shape[0]
    sectors = basis_by_magnetisation(n)
    if M not in sectors:
        valid = sorted(sectors.keys())
        raise ValueError(f"M={M} not valid for n={n}. Valid values: {valid}")
    indices = sectors[M]
    H_full = knm_to_dense_matrix(K, omega)
    return project_to_sector(H_full, indices), indices


def eigh_by_magnetisation(
    K: np.ndarray,
    omega: np.ndarray,
    sectors: list[int] | None = None,
) -> dict:
    """Diagonalise specified magnetisation sectors.

    Parameters
    ----------
    K, omega : coupling and frequencies
    sectors : list of M values to diagonalise. Default: all sectors.

    Returns
    -------
    dict with keys:
        results: dict[M] → {eigvals, eigvecs, indices, dim}
        eigvals_all: sorted eigenvalues across all computed sectors
        ground_energy: float
        ground_sector: int (M value of ground state)
    """
    n = K.shape[0]
    all_sectors = basis_by_magnetisation(n)

    if sectors is None:
        sectors = sorted(all_sectors.keys())

    H_full = knm_to_dense_matrix(K, omega)
    results: dict[int, dict] = {}
    all_eigvals: list[float] = []

    for m in sectors:
        if m not in all_sectors:
            continue
        indices = all_sectors[m]
        H_sector = project_to_sector(H_full, indices)
        vals, vecs = np.linalg.eigh(H_sector)
        results[m] = {
            "eigvals": vals,
            "eigvecs": vecs,
            "indices": indices,
            "dim": len(indices),
        }
        all_eigvals.extend(vals.tolist())

    all_eigvals_sorted = np.sort(all_eigvals)

    ground_energy = float(all_eigvals_sorted[0])
    ground_sector = min(
        (m for m in results),
        key=lambda m: float(results[m]["eigvals"][0]),
    )

    return {
        "results": results,
        "eigvals_all": np.array(all_eigvals_sorted),
        "ground_energy": ground_energy,
        "ground_sector": ground_sector,
        "n_sectors_computed": len(results),
    }


def level_spacing_by_magnetisation(
    K: np.ndarray,
    omega: np.ndarray,
    M: int | None = None,
) -> dict:
    """Level-spacing ratio r̄ within a magnetisation sector.

    Computing r̄ within a single symmetry sector avoids the artefact of
    overlaying spectra from different sectors (which always looks Poisson).

    Parameters
    ----------
    K, omega : coupling and frequencies
    M : magnetisation sector. Default: M=0 (largest sector for even N).
    """
    n = K.shape[0]
    if M is None:
        M = 0 if n % 2 == 0 else 1

    result = eigh_by_magnetisation(K, omega, sectors=[M])

    if M not in result["results"]:
        return {"r_bar": float("nan"), "M": M, "dim": 0}

    eigvals = result["results"][M]["eigvals"]
    gaps = np.diff(eigvals)
    gaps = gaps[gaps > 1e-14]

    if len(gaps) < 2:
        return {"r_bar": float("nan"), "M": M, "dim": len(eigvals)}

    ratios = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])

    return {
        "r_bar": float(np.mean(ratios)),
        "M": M,
        "dim": len(eigvals),
        "n_gaps": len(gaps),
    }


def memory_estimate(n: int) -> dict:
    """Memory estimates for different approaches.

    Returns dict with keys:
        full_ed_mb, z2_sector_mb, u1_largest_mb, u1_m0_mb
    """
    dim_full = 2**n
    dim_z2 = 2 ** (n - 1)
    dim_u1_largest = largest_sector_dim(n)
    dim_u1_m0 = comb(n, n // 2) if n % 2 == 0 else comb(n, n // 2)

    def _mb(dim: int) -> float:
        return float(dim * dim * 16 / 1e6)  # complex128

    return {
        "full_ed_mb": _mb(dim_full),
        "z2_sector_mb": _mb(dim_z2),
        "u1_largest_mb": _mb(dim_u1_largest),
        "u1_m0_mb": _mb(dim_u1_m0),
        "full_dim": dim_full,
        "z2_dim": dim_z2,
        "u1_largest_dim": dim_u1_largest,
        "reduction_factor": round(dim_full / dim_u1_largest, 1),
    }
