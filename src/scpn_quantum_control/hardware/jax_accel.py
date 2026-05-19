# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX GPU Acceleration
"""JAX-accelerated exact dense quantum analysis.

Provides vectorised coupling scans via jax.vmap. These routines still use
dense Hilbert-space Hamiltonians and exact eigensolvers; acceleration comes
from XLA/GPU batching, not from sparse, tensor-network, or sector reduction.
Callers must keep problem sizes within the active dense-allocation budget.

Requires: pip install jax[cuda12] (Linux/WSL2) or jax[cpu] (fallback).
CUDA GPU strongly recommended — XLA compilation saturates CPU.

Usage::

    from scpn_quantum_control.hardware.jax_accel import (
        is_jax_gpu_available,
        entanglement_scan_jax,
        eigensolve_batch_jax,
    )

    if is_jax_gpu_available():
        result = entanglement_scan_jax(K_topo, omega, k_range)
"""

from __future__ import annotations

import os as _os
from typing import Any

import numpy as np

from ..dense_budget import require_dense_allocation

_JAX_AVAILABLE = False
_JAX_GPU = False
_jnp: Any | None = None


def _detect_jax_accelerator() -> tuple[bool, bool, Any | None]:
    """Detect JAX without hiding present-but-broken runtimes."""
    try:
        import jax
        import jax.numpy as _jnp_module
    except ImportError:
        return False, False, None

    return True, any(d.platform == "gpu" for d in jax.devices()), _jnp_module


if _os.environ.get("SCPN_JAX_DISABLE", "0") != "1":
    _JAX_AVAILABLE, _JAX_GPU, _jnp = _detect_jax_accelerator()


def is_jax_available() -> bool:
    """Return whether JAX was importable during accelerator detection."""
    return _JAX_AVAILABLE


def is_jax_gpu_available() -> bool:
    """Return whether a JAX GPU device was detected."""
    return _JAX_GPU


def jax_device_name() -> str:
    """Return the first JAX device name or ``unavailable``."""
    if _JAX_AVAILABLE:
        import jax

        return str(jax.devices()[0])
    return "unavailable"


def _build_xy_hamiltonian_jax(K, omega, n: int):
    """Build XY Hamiltonian on JAX device. Same bitwise flip-flop as Rust."""
    dim = 1 << n
    if _jnp is None:
        raise RuntimeError("JAX NumPy backend is unavailable")
    jnp = _jnp

    h = jnp.zeros((dim, dim))

    # Diagonal: -ω_i Z_i
    for idx in range(dim):
        diag = 0.0
        for i in range(n):
            bit = (idx >> i) & 1
            diag -= omega[i] * (1.0 - 2.0 * bit)
        h = h.at[idx, idx].set(diag)

    # Off-diagonal: -K[i,j](XX+YY) flip-flop
    for i in range(n):
        for j in range(i + 1, n):
            mask = (1 << i) | (1 << j)
            for idx in range(dim):
                bi = (idx >> i) & 1
                bj = (idx >> j) & 1
                if bi != bj:
                    flipped = idx ^ mask
                    h = h.at[idx, flipped].add(-2.0 * K[i, j])

    return h


def eigensolve_batch_jax(
    K_topo: np.ndarray,
    omega: np.ndarray,
    k_range: np.ndarray,
    *,
    max_dense_gib: float | None = None,
) -> dict:
    """Batch eigendecomposition across coupling values on GPU.

    Returns dict with eigenvalues and ground states for each K_base.
    """
    if not _JAX_AVAILABLE or _jnp is None:
        raise RuntimeError("JAX not available")

    import jax

    jnp = _jnp
    n = len(omega)
    require_dense_allocation(
        n,
        dtype=np.float64,
        rank=2,
        object_count=max(len(k_range), 1),
        max_gib=max_dense_gib,
        label="JAX eigensolve dense batch",
    )
    K_topo_j = jnp.array(K_topo)
    omega_j = jnp.array(omega)
    k_range_j = jnp.array(k_range)

    @jax.jit
    def _eigvals_at_k(kb):
        K = kb * K_topo_j
        H = _build_xy_hamiltonian_jax(K, omega_j, n)
        return jnp.linalg.eigvalsh(H)

    # vmap over k_range — all K values in one GPU launch
    all_eigvals = jax.vmap(_eigvals_at_k)(k_range_j)

    return {
        "k_values": np.asarray(k_range),
        "eigenvalues": np.asarray(all_eigvals),
        "spectral_gaps": np.asarray(all_eigvals[:, 1] - all_eigvals[:, 0]),
        "ground_energies": np.asarray(all_eigvals[:, 0]),
    }


def entanglement_scan_jax(
    K_topo: np.ndarray,
    omega: np.ndarray,
    k_range: np.ndarray,
    *,
    max_dense_gib: float | None = None,
) -> dict:
    """Entanglement entropy + Schmidt gap scan on GPU via JAX.

    Builds dense Hamiltonians in numpy/Rust, transfers the batch to GPU,
    then runs dense eigh + SVD via jax.vmap. This is an exact dense path,
    not a sparse or tensor-network large-n solver.
    """
    if not _JAX_AVAILABLE or _jnp is None:
        raise RuntimeError("JAX not available")

    jnp = _jnp
    n = len(omega)
    n_A = n // 2 or 1
    dim_A = 1 << n_A
    dim_B = 1 << (n - n_A)
    dim = 1 << n
    require_dense_allocation(
        n,
        dtype=np.float64,
        rank=2,
        object_count=max(2 * len(k_range), 1),
        max_gib=max_dense_gib,
        label="JAX entanglement dense batch",
    )

    # Build all Hamiltonians in numpy/Rust (fast) then batch-transfer to GPU
    from ..bridge.knm_hamiltonian import knm_to_dense_matrix

    H_batch = np.zeros((len(k_range), dim, dim))
    for idx, kb in enumerate(k_range):
        K = float(kb) * K_topo
        H_batch[idx] = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib).real

    import jax

    H_batch_j = jnp.array(H_batch)

    @jax.jit
    def _entropy_from_H(H):
        eigvals, eigvecs = jnp.linalg.eigh(H)
        psi = eigvecs[:, 0]
        psi_mat = psi.reshape(dim_A, dim_B)
        svd_vals = jnp.linalg.svd(psi_mat, compute_uv=False)
        svd_sq = svd_vals**2
        svd_sq = jnp.where(svd_sq > 1e-30, svd_sq, 1e-30)
        entropy = -jnp.sum(svd_sq * jnp.log2(svd_sq))
        sorted_vals = jnp.sort(svd_vals)[::-1]
        schmidt_gap = sorted_vals[0] - jnp.where(sorted_vals.shape[0] > 1, sorted_vals[1], 0.0)
        spectral_gap = eigvals[1] - eigvals[0]
        return entropy, schmidt_gap, spectral_gap

    entropies, gaps, spec_gaps = jax.vmap(_entropy_from_H)(H_batch_j)

    return {
        "k_values": np.asarray(k_range),
        "entropy": np.asarray(entropies),
        "schmidt_gap": np.asarray(gaps),
        "spectral_gap": np.asarray(spec_gaps),
    }
