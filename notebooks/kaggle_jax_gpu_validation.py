# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kaggle JAX GPU Validation Notebook
import json
import subprocess
import sys
import time

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "jax[cuda12]",
        "numpy",
        "scipy",
        "qiskit",
        "qiskit-aer",
    ]
)

import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"GPU available: {any(d.platform == 'gpu' for d in jax.devices())}")


# ============================================================
# Cell 3: Build XY Hamiltonian (JAX implementation)
# ============================================================
def build_xy_hamiltonian_jax(K, omega, n):
    """XY Hamiltonian via bitwise flip-flop on JAX device."""
    dim = 1 << n
    h = jnp.zeros((dim, dim))

    for idx in range(dim):
        diag = 0.0
        for i in range(n):
            bit = (idx >> i) & 1
            diag -= omega[i] * (1.0 - 2.0 * bit)
        h = h.at[idx, idx].set(diag)

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


def entanglement_scan_jax(K_topo, omega, k_range):
    """Vectorised entanglement scan via jax.vmap."""
    n = len(omega)
    n_A = n // 2 or 1
    dim_A = 1 << n_A
    dim_B = 1 << (n - n_A)
    K_topo_j = jnp.array(K_topo)
    omega_j = jnp.array(omega)

    @jax.jit
    def _entropy_at_k(kb):
        K = kb * K_topo_j
        H = build_xy_hamiltonian_jax(K, omega_j, n)
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

    k_range_j = jnp.array(k_range)
    return jax.vmap(_entropy_at_k)(k_range_j)


# ============================================================
# Cell 4: Reference (numpy, no JAX)
# ============================================================
def entanglement_scan_numpy(K_topo, omega, k_range):
    """Reference scan using pure numpy."""
    n = len(omega)
    n_A = n // 2 or 1
    results = {"entropy": [], "schmidt_gap": [], "spectral_gap": []}

    for kb in k_range:
        K = float(kb) * K_topo
        dim = 2**n
        # Build Hamiltonian
        h = np.zeros((dim, dim))
        for idx in range(dim):
            diag = 0.0
            for i in range(n):
                bit = (idx >> i) & 1
                diag -= omega[i] * (1.0 - 2.0 * bit)
            h[idx, idx] = diag
        for i in range(n):
            for j in range(i + 1, n):
                mask = (1 << i) | (1 << j)
                for idx in range(dim):
                    bi = (idx >> i) & 1
                    bj = (idx >> j) & 1
                    if bi != bj:
                        flipped = idx ^ mask
                        h[idx, flipped] -= 2.0 * K[i, j]

        eigvals, eigvecs = np.linalg.eigh(h)
        psi = eigvecs[:, 0]
        dim_A = 2**n_A
        dim_B = 2 ** (n - n_A)
        psi_mat = psi.reshape(dim_A, dim_B)
        svd_vals = np.linalg.svd(psi_mat, compute_uv=False)
        svd_sq = svd_vals**2
        entropy = -sum(p * np.log2(p) for p in svd_sq if p > 1e-30)
        sorted_v = sorted(svd_vals, reverse=True)
        gap = sorted_v[0] - sorted_v[1] if len(sorted_v) >= 2 else sorted_v[0]

        results["entropy"].append(float(entropy))
        results["schmidt_gap"].append(float(gap))
        results["spectral_gap"].append(float(eigvals[1] - eigvals[0]))

    return results


# ============================================================
# Cell 5: Run benchmarks
# ============================================================

# Paper 27 canonical frequencies
OMEGA_N_16 = np.array(
    [
        1.329,
        2.610,
        0.844,
        1.520,
        0.710,
        3.780,
        1.055,
        0.625,
        2.210,
        1.740,
        0.480,
        3.210,
        0.915,
        1.410,
        2.830,
        0.991,
    ]
)


def build_knm(L, K_base=0.45, K_alpha=0.3):
    idx = np.arange(L)
    K = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        if i < L and j < L:
            K[i, j] = K[j, i] = val
    return K


all_results = {}

for n in [4, 6, 8]:
    print(f"\n{'=' * 50}")
    print(f"n={n} ({2**n} dim)")
    print(f"{'=' * 50}")

    K = build_knm(n)
    K_norm = K / max(np.max(K), 1e-10)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 6.0, 30)

    # Numpy reference
    t0 = time.perf_counter()
    ref = entanglement_scan_numpy(K_norm, omega, k_range)
    t_numpy = time.perf_counter() - t0
    print(f"  numpy:  {t_numpy:.3f}s")

    # JAX (first call includes JIT compilation)
    t0 = time.perf_counter()
    jax_ent, jax_gap, jax_spec = entanglement_scan_jax(K_norm, omega, k_range)
    jax.block_until_ready(jax_ent)
    t_jax_cold = time.perf_counter() - t0
    print(f"  JAX (cold): {t_jax_cold:.3f}s")

    # JAX warm (JIT cached)
    t0 = time.perf_counter()
    jax_ent, jax_gap, jax_spec = entanglement_scan_jax(K_norm, omega, k_range)
    jax.block_until_ready(jax_ent)
    t_jax_warm = time.perf_counter() - t0
    print(f"  JAX (warm): {t_jax_warm:.3f}s")

    # Parity check
    max_entropy_err = float(np.max(np.abs(np.array(jax_ent) - np.array(ref["entropy"]))))
    max_gap_err = float(np.max(np.abs(np.array(jax_gap) - np.array(ref["schmidt_gap"]))))
    print(f"  Parity: entropy err={max_entropy_err:.2e}, gap err={max_gap_err:.2e}")
    print(f"  Speedup (warm): {t_numpy / t_jax_warm:.1f}x")

    all_results[f"n{n}"] = {
        "dim": 2**n,
        "numpy_s": round(t_numpy, 4),
        "jax_cold_s": round(t_jax_cold, 4),
        "jax_warm_s": round(t_jax_warm, 4),
        "speedup_warm": round(t_numpy / t_jax_warm, 1),
        "max_entropy_err": max_entropy_err,
        "max_gap_err": max_gap_err,
    }

# Save results
print("\n" + json.dumps(all_results, indent=2))
print("\nDone. Copy the JSON above for results/jax_benchmarks.json")
