# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Fractal Self-Similarity of K_nm Across Scales
import json

import numpy as np
from scipy import stats

print("=" * 70)
print("FRACTAL SELF-SIMILARITY OF K_nm ACROSS SCALES")
print("=" * 70)

# SCPN K_nm from Paper 27 (8x8)
K_nm_8 = np.array(
    [
        [0.000, 0.951, 0.588, 0.309, 0.191, 0.118, 0.073, 0.045],
        [0.951, 0.000, 0.951, 0.588, 0.309, 0.191, 0.118, 0.073],
        [0.588, 0.951, 0.000, 0.951, 0.588, 0.309, 0.191, 0.118],
        [0.309, 0.588, 0.951, 0.000, 0.951, 0.588, 0.309, 0.191],
        [0.191, 0.309, 0.588, 0.951, 0.000, 0.951, 0.588, 0.309],
        [0.118, 0.191, 0.309, 0.588, 0.951, 0.000, 0.951, 0.588],
        [0.073, 0.118, 0.191, 0.309, 0.588, 0.951, 0.000, 0.951],
        [0.045, 0.073, 0.118, 0.191, 0.309, 0.588, 0.951, 0.000],
    ]
)

# TEST 1: Coarse-graining self-similarity
print("\n" + "=" * 70)
print("TEST 1: COARSE-GRAINING (block averaging)")
print("=" * 70)


# Coarse-grain 8x8 -> 4x4 by averaging 2x2 blocks
def coarse_grain(K, block_size=2):
    N = K.shape[0]
    N_new = N // block_size
    K_cg = np.zeros((N_new, N_new))
    for i in range(N_new):
        for j in range(N_new):
            block = K[i * block_size : (i + 1) * block_size, j * block_size : (j + 1) * block_size]
            K_cg[i, j] = np.mean(block)
    return K_cg


K_4 = coarse_grain(K_nm_8, 2)
K_2 = coarse_grain(K_4, 2)

print("8x8 K_nm (original):")
for row in K_nm_8:
    print("  " + " ".join(f"{v:.3f}" for v in row))

print("\n4x4 K_nm (coarse-grained):")
for row in K_4:
    print("  " + " ".join(f"{v:.3f}" for v in row))

print("\n2x2 K_nm (coarse-grained):")
for row in K_2:
    print("  " + " ".join(f"{v:.3f}" for v in row))


# Self-similarity test: normalise each matrix and compare structure
def normalise_offdiag(K):
    mask = ~np.eye(K.shape[0], dtype=bool)
    vals = K[mask]
    if np.max(vals) > 0:
        K_norm = K / np.max(vals)
    else:
        K_norm = K.copy()
    np.fill_diagonal(K_norm, 0)
    return K_norm


K8_norm = normalise_offdiag(K_nm_8)
K4_norm = normalise_offdiag(K_4)
K2_norm = normalise_offdiag(K_2)


# Extract decay profiles (coupling vs distance)
def decay_profile(K):
    N = K.shape[0]
    max_d = N - 1
    profile = []
    for d in range(1, max_d + 1):
        vals = []
        for i in range(N):
            j = i + d
            if j < N:
                vals.append(K[i, j])
        profile.append(np.mean(vals))
    return np.array(profile)


prof8 = decay_profile(K8_norm)
prof4 = decay_profile(K4_norm)
prof2 = decay_profile(K2_norm)

print("\nNormalised decay profiles:")
print(f"  8x8: {' '.join(f'{v:.3f}' for v in prof8)}")
print(f"  4x4: {' '.join(f'{v:.3f}' for v in prof4)}")
print(f"  2x2: {' '.join(f'{v:.3f}' for v in prof2)}")

# Compare 8x8 and 4x4 profiles (truncated to same length)
min_len = min(len(prof8), len(prof4))
if min_len >= 3:
    r_self, p_self = stats.pearsonr(prof8[:min_len], prof4[:min_len])
    print(f"\nSelf-similarity (8x8 vs 4x4 profile): r={r_self:.4f}, p={p_self:.4f}")
else:
    r_self = np.nan
    p_self = np.nan
    print("\nToo few points for correlation")


# TEST 2: Eigenvalue spectrum
print("\n" + "=" * 70)
print("TEST 2: EIGENVALUE SPECTRUM")
print("=" * 70)

eigenvalues = np.sort(np.linalg.eigvalsh(K_nm_8))[::-1]
print("Eigenvalues of K_nm (8x8):")
for i, ev in enumerate(eigenvalues):
    print(f"  lambda_{i} = {ev:.6f}")

# Eigenvalue ratios
print("\nEigenvalue ratios (consecutive):")
ev_ratios = []
for i in range(len(eigenvalues) - 1):
    if abs(eigenvalues[i + 1]) > 1e-10:
        ratio = eigenvalues[i] / eigenvalues[i + 1]
        ev_ratios.append(ratio)
        print(f"  lambda_{i}/lambda_{i + 1} = {ratio:.4f}")

# Power law test: lambda_k ~ k^(-beta)
k_vals = np.arange(1, len(eigenvalues) + 1)
pos_ev = eigenvalues[eigenvalues > 0]
k_pos = np.arange(1, len(pos_ev) + 1)

if len(pos_ev) >= 3:
    log_k = np.log(k_pos)
    log_ev = np.log(pos_ev)
    slope_ev, intercept_ev, r_ev, p_ev, _ = stats.linregress(log_k, log_ev)
    print(f"\nPower law fit (positive eigenvalues): lambda ~ k^({slope_ev:.3f})")
    print(f"R^2 = {r_ev**2:.4f}, p = {p_ev:.4f}")
    print(f"Fractal dimension estimate: D_s ~ {-slope_ev:.3f}")
else:
    slope_ev = np.nan
    r_ev = np.nan

# Comparison to known systems
print("\nComparison to known spectral exponents:")
print("  Random matrix (GOE):     lambda ~ k^(-1/2)")
print("  1D chain:                lambda ~ k^(-2)")
print("  Sierpinski gasket:       lambda ~ k^(-0.68)")
print(f"  SCPN K_nm:               lambda ~ k^({slope_ev:.3f})")


# TEST 3: Multifractal analysis
print("\n" + "=" * 70)
print("TEST 3: MULTIFRACTAL ANALYSIS OF K_nm ELEMENTS")
print("=" * 70)

# Extract upper triangle elements
mask = np.triu(np.ones_like(K_nm_8, dtype=bool), k=1)
elements = K_nm_8[mask]

# Box-counting at different scales
print(f"\nK_nm element distribution ({len(elements)} values):")
print(f"  Min: {np.min(elements):.4f}")
print(f"  Max: {np.max(elements):.4f}")
print(f"  Mean: {np.mean(elements):.4f}")
print(f"  Std: {np.std(elements):.4f}")

# Generalised dimensions D_q
# For a probability distribution p_i, D_q = lim(log(sum(p_i^q)) / ((q-1) * log(eps)))
# We compute for the K_nm element distribution

# Normalise to probabilities
p = elements / np.sum(elements)

q_vals = np.array([-5, -3, -1, 0, 1, 2, 3, 5, 10])
print("\nGeneralised dimensions D_q:")
for q in q_vals:
    if q == 1:
        # Shannon entropy dimension
        D_q = -np.sum(p * np.log(p + 1e-15)) / np.log(len(p))
    else:
        D_q = np.log(np.sum(p**q)) / ((q - 1) * np.log(len(p)))
    print(f"  D_{q:+3.0f} = {D_q:.4f}")

# Multifractal width
D_inf_pos = np.log(np.sum(p**10)) / (9 * np.log(len(p)))
D_inf_neg = np.log(np.sum(p ** (-5))) / ((-6) * np.log(len(p)))
mf_width = abs(D_inf_neg - D_inf_pos)
print(f"\nMultifractal width: Delta_D = {mf_width:.4f}")
print("(0 = monofractal, >0.1 = multifractal)")
is_multifractal = mf_width > 0.05


# TEST 4: Renormalisation group flow of K_c
print("\n" + "=" * 70)
print("TEST 4: RENORMALISATION GROUP FLOW")
print("=" * 70)

# RG: coarse-grain, then measure K_c at each scale
# If self-similar, K_c should scale predictably

omega_8 = np.array([0.062, 0.191, 0.382, 0.618, 0.809, 0.927, 0.981, 1.000])
omega_4 = np.array([np.mean(omega_8[i : i + 2]) for i in range(0, 8, 2)])
omega_2 = np.array([np.mean(omega_4[i : i + 2]) for i in range(0, 4, 2)])


def find_K_c(N, K_nm, omega, n_trials=15, T=200, dt=0.01):
    """Binary search for K_c where R crosses 0.5."""
    K_lo, K_hi = 0.1, 20.0
    for _ in range(15):
        K_mid = (K_lo + K_hi) / 2
        n_steps = int(T / dt)
        R_trials = []
        for _ in range(n_trials):
            theta = np.random.uniform(0, 2 * np.pi, N)
            for _s in range(n_steps):
                dtheta = omega.copy()
                for i in range(N):
                    coupling = 0.0
                    for j in range(N):
                        coupling += K_mid * K_nm[i, j] * np.sin(theta[j] - theta[i])
                    dtheta[i] += coupling / N
                theta += dtheta * dt
            z = np.mean(np.exp(1j * theta))
            R_trials.append(abs(z))
        R = np.mean(R_trials)
        if R > 0.5:
            K_hi = K_mid
        else:
            K_lo = K_mid
    return K_mid


print("Computing K_c at each RG scale...")
K_c_8 = find_K_c(8, K_nm_8, omega_8, n_trials=10)
print(f"  N=8: K_c = {K_c_8:.3f}")

K_c_4 = find_K_c(4, K_4, omega_4, n_trials=10)
print(f"  N=4: K_c = {K_c_4:.3f}")

K_c_2 = find_K_c(2, K_2, omega_2, n_trials=10)
print(f"  N=2: K_c = {K_c_2:.3f}")

# RG flow: how does K_c scale with N?
Ns = np.array([2, 4, 8])
K_cs = np.array([K_c_2, K_c_4, K_c_8])
log_N = np.log(Ns)
log_Kc = np.log(K_cs)
slope_rg, intercept_rg, r_rg, p_rg, _ = stats.linregress(log_N, log_Kc)
print(f"\nRG scaling: K_c ~ N^({slope_rg:.3f}), R^2 = {r_rg**2:.4f}")
print(f"Prediction: K_c(N=16) = {np.exp(intercept_rg + slope_rg * np.log(16)):.3f}")
print(f"Prediction: K_c(N=32) = {np.exp(intercept_rg + slope_rg * np.log(32)):.3f}")
print(f"Prediction: K_c(N=64) = {np.exp(intercept_rg + slope_rg * np.log(64)):.3f}")


# TEST 5: Comparison to known fractal networks
print("\n" + "=" * 70)
print("TEST 5: COMPARISON TO FRACTAL NETWORKS")
print("=" * 70)


# Build known fractal coupling matrices for comparison
# Hierarchical: K_ij = 2^(-level(i,j))
def hierarchical_K(N):
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # Level = position of highest differing bit
                xor = i ^ j
                level = int(np.log2(xor)) + 1 if xor > 0 else 0
                K[i, j] = 2.0 ** (-level)
    return K


# Small-world: exponential decay + random long-range
def small_world_K(N, alpha=0.3, p_long=0.1, seed=42):
    rng = np.random.RandomState(seed)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                d = abs(i - j)
                K[i, j] = np.exp(-alpha * d)
                if rng.random() < p_long:
                    K[i, j] = max(K[i, j], 0.5)
    return K


networks = {
    "SCPN": K_nm_8,
    "hierarchical": hierarchical_K(8),
    "small_world": small_world_K(8, 0.3, 0.1),
}

for name, K in networks.items():
    evs = np.sort(np.linalg.eigvalsh(K))[::-1]
    pos_evs = evs[evs > 1e-10]
    prof = decay_profile(normalise_offdiag(K))

    if len(pos_evs) >= 3:
        k_idx = np.arange(1, len(pos_evs) + 1)
        sl, _, rv, _, _ = stats.linregress(np.log(k_idx), np.log(pos_evs))
        spec_exp = sl
    else:
        spec_exp = np.nan

    mask_ut = np.triu(np.ones_like(K, dtype=bool), k=1)
    elems = K[mask_ut]
    entropy = -np.sum((elems / np.sum(elems)) * np.log(elems / np.sum(elems) + 1e-15))

    print(f"\n{name}:")
    print(f"  Spectral exponent: {spec_exp:.3f}")
    print(f"  Decay profile: {' '.join(f'{v:.3f}' for v in prof[:5])}")
    print(f"  Element entropy: {entropy:.3f}")
    print(f"  Max eigenvalue: {evs[0]:.3f}")

    # Correlation with SCPN decay profile
    if name != "SCPN":
        min_l = min(len(prof), len(decay_profile(normalise_offdiag(K_nm_8))))
        p_scpn = decay_profile(normalise_offdiag(K_nm_8))[:min_l]
        p_other = prof[:min_l]
        if min_l >= 3:
            r_comp, p_comp = stats.pearsonr(p_scpn, p_other)
            print(f"  Correlation with SCPN: r={r_comp:.4f}, p={p_comp:.4f}")


# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: FRACTAL STRUCTURE OF K_nm")
print("=" * 70)

print(
    f"\n1. Coarse-graining self-similarity: r={r_self:.4f}"
    + (f", p={p_self:.4f}" if not np.isnan(p_self) else "")
)
if not np.isnan(r_self) and r_self > 0.9:
    print("   SELF-SIMILAR: decay profile preserved under coarse-graining")
elif not np.isnan(r_self):
    print("   PARTIALLY self-similar")
else:
    print("   Insufficient data")

print(f"2. Spectral exponent: {slope_ev:.3f}")
print("   (Sierpinski: -0.68, 1D chain: -2.0, random: -0.5)")

print(f"3. Multifractal: {'YES' if is_multifractal else 'NO'} (width={mf_width:.4f})")

print(f"4. RG flow: K_c ~ N^({slope_rg:.3f})")
if slope_rg > 0:
    print("   K_c INCREASES with system size -> harder to sync larger systems")
else:
    print("   K_c DECREASES with system size -> easier to sync larger systems")

print("5. Closest known network: ", end="")
# Determine closest match based on spectral exponent

# JSON output
results = {
    "self_similarity_r": round(float(r_self), 4) if not np.isnan(r_self) else None,
    "self_similarity_p": round(float(p_self), 4) if not np.isnan(p_self) else None,
    "spectral_exponent": round(float(slope_ev), 4) if not np.isnan(slope_ev) else None,
    "spectral_R2": round(float(r_ev**2), 4) if not np.isnan(r_ev) else None,
    "multifractal": is_multifractal,
    "multifractal_width": round(float(mf_width), 4),
    "rg_K_c_exponent": round(float(slope_rg), 4),
    "rg_R2": round(float(r_rg**2), 4),
    "K_c_N8": round(float(K_c_8), 3),
    "K_c_N4": round(float(K_c_4), 3),
    "K_c_N2": round(float(K_c_2), 3),
    "K_c_N16_predicted": round(float(np.exp(intercept_rg + slope_rg * np.log(16))), 3),
}

print("\n\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
