# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — BKT Universality Tests on Kaggle
#
# Tests two predictions:
# 1. CFT central charge c=1 from entanglement scaling S ~ (c/3)ln(L)
# 2. Spectral gap BKT essential singularity Delta ~ exp(-b/sqrt(K-K_c))
# Plus: non-ergodicity (level spacing + eigenstate entanglement) at n=10,12

import json
import subprocess
import sys
import time

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy"])

import numpy as np

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


def build_hamiltonian(K, omega, n):
    dim = 1 << n
    h = np.zeros((dim, dim))
    for idx in range(dim):
        diag = 0.0
        for i in range(n):
            bit = (idx >> i) & 1
            diag -= omega[i] * (1.0 - 2.0 * bit)
        h[idx, idx] = diag
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < 1e-15:
                continue
            mask = (1 << i) | (1 << j)
            for idx in range(dim):
                bi = (idx >> i) & 1
                bj = (idx >> j) & 1
                if bi != bj:
                    h[idx, idx ^ mask] -= 2.0 * K[i, j]
    return h


def level_spacing_ratio(eigenvalues):
    spacings = np.diff(np.sort(eigenvalues))
    spacings = spacings[spacings > 1e-15]
    if len(spacings) < 2:
        return 0.0
    ratios = np.minimum(spacings[:-1], spacings[1:]) / np.maximum(spacings[:-1], spacings[1:])
    return float(np.mean(ratios))


results = {}

# ============================================================
# TEST 1: CFT Central Charge
# ============================================================
print("=" * 60)
print("TEST 1: CFT CENTRAL CHARGE c FROM ENTANGLEMENT SCALING")
print("BKT prediction: S(l) = (c/3) ln(l) + const, c = 1")
print("=" * 60)

K_c_est = 3.0

for n in [6, 8, 10, 12]:
    print(f"\nn={n} ({2**n} dim)...")
    t0 = time.perf_counter()
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    H = build_hamiltonian(K_c_est * K_norm, omega, n)
    _, eigvecs = np.linalg.eigh(H)
    psi = eigvecs[:, 0]

    entropies = []
    l_values = []
    for l in range(1, n // 2 + 1):
        dim_A = 2**l
        dim_B = 2 ** (n - l)
        svd_sq = np.linalg.svd(psi.reshape(dim_A, dim_B), compute_uv=False) ** 2
        svd_sq = svd_sq[svd_sq > 1e-30]
        S = -np.sum(svd_sq * np.log2(svd_sq))
        entropies.append(float(S))
        l_values.append(l)

    x = np.log(np.array(l_values))
    y = np.array(entropies)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        c_periodic = 3 * slope
        c_open = 6 * slope
    else:
        c_periodic = c_open = float("nan")

    dt = time.perf_counter() - t0
    print(f"  S(l) = {[round(s, 3) for s in entropies]}")
    print(f"  c_periodic = {c_periodic:.3f}, c_open = {c_open:.3f} (BKT: 1.000)")
    print(f"  Time: {dt:.1f}s")

    results[f"cft_n{n}"] = {
        "n": n,
        "K_c": K_c_est,
        "l_values": l_values,
        "entropies": entropies,
        "c_periodic": round(c_periodic, 4),
        "c_open": round(c_open, 4),
        "time_s": round(dt, 1),
    }

# ============================================================
# TEST 2: BKT Gap Fit
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: SPECTRAL GAP BKT ESSENTIAL SINGULARITY")
print("Delta ~ exp(-b / sqrt(K - K_c))")
print("=" * 60)

for n in [4, 6, 8, 10]:
    print(f"\nn={n}...")
    t0 = time.perf_counter()
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    k_range = np.linspace(0.5, 8.0, 30)
    gaps = []
    for kb in k_range:
        H = build_hamiltonian(kb * K_norm, omega, n)
        evals = np.linalg.eigvalsh(H)
        gaps.append(evals[1] - evals[0])
    gaps = np.array(gaps)

    min_idx = np.argmin(gaps)
    K_c_local = k_range[min_idx]

    mask = (k_range > K_c_local + 0.3) & (gaps > 1e-6)
    if np.sum(mask) >= 3:
        x_fit = -1.0 / np.sqrt(k_range[mask] - K_c_local)
        y_fit = np.log(gaps[mask])
        slope_bkt, intercept_bkt = np.polyfit(x_fit, y_fit, 1)
        y_pred = slope_bkt * x_fit + intercept_bkt
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        verdict = "BKT CONFIRMED" if R2 > 0.8 else "BKT REJECTED"
    else:
        slope_bkt = R2 = 0
        verdict = "insufficient data"

    dt = time.perf_counter() - t0
    print(f"  K_c = {K_c_local:.2f}, b = {slope_bkt:.3f}, R2 = {R2:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  Time: {dt:.1f}s")

    results[f"bkt_n{n}"] = {
        "n": n,
        "K_c": round(K_c_local, 3),
        "b": round(slope_bkt, 3),
        "R2": round(R2, 4),
        "verdict": verdict,
        "time_s": round(dt, 1),
    }

# ============================================================
# TEST 3: Non-Ergodicity at Large N
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: NON-ERGODICITY AT n=10, 12")
print("Poisson r_bar = 0.386, GOE r_bar = 0.530")
print("=" * 60)

for n in [10, 12]:
    print(f"\nn={n} ({2**n} dim)...")
    t0 = time.perf_counter()
    K_topo = build_knm(n)
    K_norm = K_topo / max(np.max(K_topo), 1e-10)
    omega = OMEGA_N_16[:n]
    H = build_hamiltonian(3.0 * K_norm, omega, n)
    evals, evecs = np.linalg.eigh(H)
    r_bar = level_spacing_ratio(evals)

    # Eigenstate entanglement (middle quarter)
    dim = 2**n
    n_A = n // 2
    dim_A = 2**n_A
    dim_B = 2 ** (n - n_A)
    quarter = dim // 4
    S_excited = []
    for idx in range(quarter, min(3 * quarter, dim)):
        psi = evecs[:, idx]
        svd_sq = np.linalg.svd(psi.reshape(dim_A, dim_B), compute_uv=False) ** 2
        svd_sq = svd_sq[svd_sq > 1e-30]
        S_excited.append(-np.sum(svd_sq * np.log2(svd_sq)))
    S_mean = np.mean(S_excited)
    S_ratio = S_mean / n_A
    S_thermal = (n_A - 0.72) / n_A

    dt = time.perf_counter() - t0
    print(f"  r_bar = {r_bar:.4f} (Poisson=0.386, GOE=0.530)")
    print(f"  S_excited/S_max = {S_ratio:.4f} (thermal={S_thermal:.4f})")
    print(f"  Deficit: {(1 - S_ratio / S_thermal) * 100:.1f}% below thermal")
    print(f"  Time: {dt:.1f}s")

    results[f"ergodicity_n{n}"] = {
        "n": n,
        "r_bar": round(r_bar, 4),
        "S_ratio": round(S_ratio, 4),
        "S_thermal": round(S_thermal, 4),
        "deficit_pct": round((1 - S_ratio / S_thermal) * 100, 1),
        "time_s": round(dt, 1),
    }

print("\n" + "=" * 60)
print("RESULTS JSON")
print("=" * 60)
print(json.dumps(results, indent=2))
print("\nDone.")
