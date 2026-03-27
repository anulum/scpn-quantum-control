# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — JAX GPU Acceleration
"""JAX-accelerated quantum analysis for large systems (n=16-25 qubits).

Provides vectorised coupling scans via jax.vmap — runs the full scan
(Hamiltonian construction + eigendecomposition + analysis) as a single
GPU kernel instead of a Python loop.

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

import numpy as np

_JAX_AVAILABLE = False
_JAX_GPU = False
_jnp = None

if _os.environ.get("SCPN_JAX_DISABLE", "0") != "1":
    try:
        import jax
        import jax.numpy as _jnp_module

        _jnp = _jnp_module
        _JAX_AVAILABLE = True
        _JAX_GPU = any(d.platform == "gpu" for d in jax.devices())
    except (ImportError, Exception):
        pass


def is_jax_available() -> bool:
    return _JAX_AVAILABLE


def is_jax_gpu_available() -> bool:
    return _JAX_GPU


def jax_device_name() -> str:
    if _JAX_AVAILABLE:
        import jax

        return str(jax.devices()[0])
    return "unavailable"


def _build_xy_hamiltonian_jax(K: jax.Array, omega: jax.Array, n: int) -> jax.Array:
    """Build XY Hamiltonian on JAX device. Same bitwise flip-flop as Rust."""
    dim = 1 << n
    assert _jnp is not None  # caller checks _JAX_AVAILABLE
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
) -> dict:
    """Batch eigendecomposition across coupling values on GPU.

    Returns dict with eigenvalues and ground states for each K_base.
    """
    if not _JAX_AVAILABLE or _jnp is None:
        raise RuntimeError("JAX not available")

    import jax

    jnp = _jnp
    n = len(omega)
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
) -> dict:
    """Entanglement entropy + Schmidt gap scan, fully on GPU via JAX.

    For each K_base: build H → eigh → SVD of ground state → entropy.
    Vectorised with jax.vmap for GPU parallelism.
    """
    if not _JAX_AVAILABLE or _jnp is None:
        raise RuntimeError("JAX not available")

    import jax

    jnp = _jnp
    n = len(omega)
    n_A = n // 2 or 1
    dim_A = 1 << n_A
    dim_B = 1 << (n - n_A)
    K_topo_j = jnp.array(K_topo)
    omega_j = jnp.array(omega)

    @jax.jit
    def _entropy_at_k(kb):
        K = kb * K_topo_j
        H = _build_xy_hamiltonian_jax(K, omega_j, n)
        eigvals, eigvecs = jnp.linalg.eigh(H)
        psi = eigvecs[:, 0]

        # SVD of reshaped ground state
        psi_mat = psi.reshape(dim_A, dim_B)
        svd_vals = jnp.linalg.svd(psi_mat, compute_uv=False)
        svd_sq = svd_vals**2
        svd_sq = jnp.where(svd_sq > 1e-30, svd_sq, 1e-30)
        entropy = -jnp.sum(svd_sq * jnp.log2(svd_sq))

        sorted_vals = jnp.sort(svd_vals)[::-1]
        schmidt_gap = sorted_vals[0] - jnp.where(sorted_vals.shape[0] > 1, sorted_vals[1], 0.0)
        spectral_gap = eigvals[1] - eigvals[0]

        return entropy, schmidt_gap, spectral_gap

    k_range_j = jnp.array(k_range)
    entropies, gaps, spec_gaps = jax.vmap(_entropy_at_k)(k_range_j)

    return {
        "k_values": np.asarray(k_range),
        "entropy": np.asarray(entropies),
        "schmidt_gap": np.asarray(gaps),
        "spectral_gap": np.asarray(spec_gaps),
    }
