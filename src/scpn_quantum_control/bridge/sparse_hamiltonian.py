# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sparse XY Hamiltonian Construction
"""Sparse CSC/CSR Hamiltonian for large-N Kuramoto-XY systems.

The XY Hamiltonian H = -Σ K_ij(X_iX_j + Y_iY_j) - Σ ω_i Z_i has O(n²·2^n)
non-zero elements in a 2^n × 2^n matrix. For n=16 the matrix is 65536×65536
but has only ~8M non-zeros (0.2% fill). Sparse storage + sparse eigensolver
(ARPACK via scipy.sparse.linalg.eigsh) enables:

  n=16: 8M non-zeros × 24 bytes ≈ 200 MB (vs 32 GB dense)
  n=18: 34M non-zeros ≈ 800 MB
  n=20: 130M non-zeros ≈ 3 GB

Combined with U(1) magnetisation sectors, sparse eigsh enables n=22+ on 32 GB RAM.

Inspired by QuSpin's sparse Hamiltonian construction (Weinberg & Bukov, SciPost 2017).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def _try_rust_sparse(K: np.ndarray, omega: np.ndarray, n: int) -> sparse.csc_matrix | None:
    """Try Rust-accelerated sparse construction (80× faster)."""
    try:
        import scpn_quantum_engine as eng

        rows, cols, vals = eng.build_sparse_xy_hamiltonian(K.ravel(), omega, n)
        return sparse.csc_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(2**n, 2**n),
        )
    except (ImportError, Exception):
        return None


def build_sparse_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
) -> sparse.csc_matrix:
    """Build the XY Hamiltonian as a sparse CSC matrix.

    H_{k, k⊕mask_ij} = -2K_ij  when b_i(k) ≠ b_j(k)
    H_{kk} = -Σ_i ω_i(1 - 2b_i(k))

    Parameters
    ----------
    K : array (n, n)
        Coupling matrix.
    omega : array (n,)
        Natural frequencies.

    Returns
    -------
    scipy.sparse.csc_matrix
        Sparse Hamiltonian, shape (2^n, 2^n).
    """
    n = K.shape[0]
    dim = 2**n

    # Rust fast path (80× faster at n=8)
    rust_result = _try_rust_sparse(K, omega, n)
    if rust_result is not None:
        return rust_result

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Diagonal: -Σ ω_i (1 - 2·b_i(k))
    for k in range(dim):
        diag = 0.0
        for i in range(n):
            bi = (k >> i) & 1
            diag -= omega[i] * (1 - 2 * bi)
        rows.append(k)
        cols.append(k)
        vals.append(diag)

    # Off-diagonal: XY coupling (flip-flop)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < 1e-15:
                continue
            mask = (1 << i) | (1 << j)
            val = -2.0 * K[i, j]
            for k in range(dim):
                bi = (k >> i) & 1
                bj = (k >> j) & 1
                if bi != bj:
                    k_flip = k ^ mask
                    rows.append(k)
                    cols.append(k_flip)
                    vals.append(val)

    H = sparse.csc_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(dim, dim),
    )
    return H


def build_sparse_sector_hamiltonian(
    K: np.ndarray,
    omega: np.ndarray,
    M: int,
) -> tuple[sparse.csc_matrix, np.ndarray]:
    """Build sparse Hamiltonian within a U(1) magnetisation sector.

    Combines sparse construction with U(1) symmetry for maximum reduction.

    Returns (H_sparse, sector_indices).
    """
    from ..analysis.magnetisation_sectors import basis_by_magnetisation

    n = K.shape[0]
    sectors = basis_by_magnetisation(n)
    if M not in sectors:
        valid = sorted(sectors.keys())
        raise ValueError(f"M={M} not valid. Valid: {valid}")

    indices = sectors[M]
    dim_sector = len(indices)

    # Build inverse map: full index → sector index
    inv_map = {int(idx): i for i, idx in enumerate(indices)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for si, k in enumerate(indices):
        # Diagonal
        diag = 0.0
        for i in range(n):
            bi = (int(k) >> i) & 1
            diag -= omega[i] * (1 - 2 * bi)
        rows.append(si)
        cols.append(si)
        vals.append(diag)

        # Off-diagonal
        for i in range(n):
            for j in range(i + 1, n):
                if abs(K[i, j]) < 1e-15:
                    continue
                bi = (int(k) >> i) & 1
                bj = (int(k) >> j) & 1
                if bi != bj:
                    k_flip = int(k) ^ ((1 << i) | (1 << j))
                    if k_flip in inv_map:
                        sj = inv_map[k_flip]
                        rows.append(si)
                        cols.append(sj)
                        vals.append(-2.0 * K[i, j])

    H = sparse.csc_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(dim_sector, dim_sector),
    )
    return H, indices


def sparse_eigsh(
    K: np.ndarray,
    omega: np.ndarray,
    k: int = 10,
    which: str = "SA",
    M: int | None = None,
) -> dict:
    """Sparse eigenvalue computation for the XY Hamiltonian.

    Uses ARPACK (scipy.sparse.linalg.eigsh) for the k smallest eigenvalues.

    Parameters
    ----------
    K, omega : coupling and frequencies
    k : number of eigenvalues to compute
    which : "SA" (smallest algebraic), "LA" (largest), "SM" (smallest magnitude)
    M : if specified, compute within magnetisation sector M

    Returns
    -------
    dict with keys: eigvals, eigvecs, nnz, dim, sector
    """
    if M is not None:
        H, indices = build_sparse_sector_hamiltonian(K, omega, M)
        sector_info = {"M": M, "sector_dim": H.shape[0]}
    else:
        H = build_sparse_hamiltonian(K, omega)
        sector_info = {"M": None, "sector_dim": H.shape[0]}

    # Ensure k < dim
    actual_k = min(k, H.shape[0] - 2)
    if actual_k < 1:
        # Matrix too small for sparse solver — use dense
        H_dense = H.toarray()
        vals, vecs = np.linalg.eigh(H_dense)
        return {
            "eigvals": vals[:k],
            "eigvecs": vecs[:, :k],
            "nnz": H.nnz,
            "dim": H.shape[0],
            "method": "dense_fallback",
            **sector_info,
        }

    vals, vecs = eigsh(H, k=actual_k, which=which)
    idx = np.argsort(vals)
    return {
        "eigvals": vals[idx],
        "eigvecs": vecs[:, idx],
        "nnz": H.nnz,
        "dim": H.shape[0],
        "method": "sparse_arpack",
        **sector_info,
    }


def sparsity_stats(n: int, K: np.ndarray) -> dict:
    """Estimate sparsity statistics without building the full matrix.

    Returns dict with: dim, nnz_estimate, fill_pct, memory_sparse_mb, memory_dense_mb
    """
    dim = 2**n
    n_couplings = sum(1 for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > 1e-15)
    # Each coupling contributes 2^n non-zeros (each basis state with bi≠bj)
    # On average, half the basis states have bi≠bj for a given (i,j) pair
    nnz_offdiag = n_couplings * dim  # upper bound (exact: n_couplings × 2^(n-1))
    nnz_diag = dim
    nnz_total = nnz_offdiag + nnz_diag

    # Sparse CSC: 3 arrays (data, indices, indptr)
    # data: nnz × 8 bytes, indices: nnz × 4 bytes, indptr: (dim+1) × 4 bytes
    mem_sparse_mb = (nnz_total * 12 + (dim + 1) * 4) / 1e6
    mem_dense_mb = dim * dim * 8 / 1e6  # float64

    return {
        "dim": dim,
        "nnz_estimate": nnz_total,
        "fill_pct": round(100 * nnz_total / (dim * dim), 3),
        "memory_sparse_mb": round(mem_sparse_mb, 1),
        "memory_dense_mb": round(mem_dense_mb, 1),
        "reduction_factor": round(mem_dense_mb / mem_sparse_mb, 1)
        if mem_sparse_mb > 0
        else float("inf"),
    }
