# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FMO Photosynthesis as Kuramoto System
import json

import numpy as np
from scipy import linalg, stats

print("=" * 70)
print("FMO PHOTOSYNTHESIS COMPLEX AS KURAMOTO SYSTEM")
print("=" * 70)

# =====================================================================
# FMO Hamiltonian (cm^-1) from Adolphs & Renger 2006
# 7-site model, widely used in quantum biology literature
# =====================================================================

# Site energies (diagonal) in cm^-1
E_fmo = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])

# Coupling matrix (off-diagonal) in cm^-1
# From Adolphs & Renger, Biophys J 2006
J_fmo = np.array(
    [
        [0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
        [-87.7, 0, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, 0, -53.5, -2.2, -9.6, 6.0],
        [-5.9, 8.2, -53.5, 0, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 0, 81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0, 81.1, 0, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 0],
    ]
)

N_fmo = 7

# Full Hamiltonian
H_fmo = np.diag(E_fmo) + J_fmo

print("FMO site energies (cm^-1):")
for i, e in enumerate(E_fmo):
    print(f"  BChl {i + 1}: {e:.0f}")

print("\nFMO coupling matrix |J_ij| (cm^-1):")
for i in range(N_fmo):
    row = " ".join(f"{abs(J_fmo[i, j]):6.1f}" for j in range(N_fmo))
    print(f"  {row}")

# =====================================================================
# SCPN K_nm (first 7 of 8 oscillators for comparison)
# =====================================================================
K_nm_scpn = np.array(
    [
        [0.000, 0.951, 0.588, 0.309, 0.191, 0.118, 0.073],
        [0.951, 0.000, 0.951, 0.588, 0.309, 0.191, 0.118],
        [0.588, 0.951, 0.000, 0.951, 0.588, 0.309, 0.191],
        [0.309, 0.588, 0.951, 0.000, 0.951, 0.588, 0.309],
        [0.191, 0.309, 0.588, 0.951, 0.000, 0.951, 0.588],
        [0.118, 0.191, 0.309, 0.588, 0.951, 0.000, 0.951],
        [0.073, 0.118, 0.191, 0.309, 0.588, 0.951, 0.000],
    ]
)

# TEST 1: Structural comparison
print("\n" + "=" * 70)
print("TEST 1: FMO vs SCPN COUPLING STRUCTURE")
print("=" * 70)

# Normalise both matrices
J_abs = np.abs(J_fmo)
np.fill_diagonal(J_abs, 0)
J_norm = J_abs / np.max(J_abs)

K_norm = K_nm_scpn / np.max(K_nm_scpn)

# Extract upper triangles
mask = np.triu(np.ones((N_fmo, N_fmo), dtype=bool), k=1)
fmo_vals = J_norm[mask]
scpn_vals = K_norm[mask]

r_struct, p_struct = stats.pearsonr(fmo_vals, scpn_vals)
print(f"Structural correlation (normalised |J| vs K_nm): r={r_struct:.4f}, p={p_struct:.4f}")


# Decay profiles
def decay_profile(K, N):
    profile = []
    for d in range(1, N):
        vals = []
        for i in range(N):
            j = i + d
            if j < N:
                vals.append(K[i, j])
        profile.append(np.mean(vals))
    return np.array(profile)


prof_fmo = decay_profile(J_norm, N_fmo)
prof_scpn = decay_profile(K_norm, N_fmo)

print("\nDecay profiles (normalised):")
print(f"  FMO:  {' '.join(f'{v:.3f}' for v in prof_fmo)}")
print(f"  SCPN: {' '.join(f'{v:.3f}' for v in prof_scpn)}")

r_decay, p_decay = stats.pearsonr(prof_fmo, prof_scpn)
print(f"Decay profile correlation: r={r_decay:.4f}, p={p_decay:.4f}")

# FMO topology characterisation
print("\nFMO topology:")
print(f"  Strongest coupling: BChl 5-6 ({abs(J_fmo[4, 5]):.1f} cm^-1)")
print(f"  Next: BChl 1-2 ({abs(J_fmo[0, 1]):.1f} cm^-1)")
print("  FMO is NOT a simple chain — it has specific long-range couplings")

# Clustering coefficient
threshold = 0.1  # normalised
adj_fmo = (J_norm > threshold).astype(float)
adj_scpn = (K_norm > threshold).astype(float)
print(f"\n  FMO adjacency (>{threshold}): {np.sum(adj_fmo) / 2:.0f} edges")
print(f"  SCPN adjacency (>{threshold}): {np.sum(adj_scpn) / 2:.0f} edges")


# TEST 2: Eigenvalue spectrum comparison
print("\n" + "=" * 70)
print("TEST 2: EIGENVALUE SPECTRA")
print("=" * 70)

ev_fmo = np.sort(linalg.eigvalsh(H_fmo))[::-1]
ev_scpn = np.sort(linalg.eigvalsh(K_nm_scpn))[::-1]

# Normalise to [0,1] range
ev_fmo_norm = (ev_fmo - ev_fmo[-1]) / (ev_fmo[0] - ev_fmo[-1])
ev_scpn_norm = (ev_scpn - ev_scpn[-1]) / (ev_scpn[0] - ev_scpn[-1])

print("Normalised eigenvalues:")
for i in range(N_fmo):
    print(f"  {i}: FMO={ev_fmo_norm[i]:.4f}, SCPN={ev_scpn_norm[i]:.4f}")

r_ev, p_ev = stats.pearsonr(ev_fmo_norm, ev_scpn_norm)
print(f"\nEigenvalue correlation: r={r_ev:.4f}, p={p_ev:.4f}")

# Level spacing
fmo_spacing = np.diff(ev_fmo_norm)
scpn_spacing = np.diff(ev_scpn_norm)
print("\nLevel spacings:")
print(f"  FMO:  {' '.join(f'{s:.4f}' for s in fmo_spacing)}")
print(f"  SCPN: {' '.join(f'{s:.4f}' for s in scpn_spacing)}")


# TEST 3: Kuramoto simulation of FMO
print("\n" + "=" * 70)
print("TEST 3: KURAMOTO SIMULATION OF FMO")
print("=" * 70)

# Map FMO to Kuramoto:
# omega_i = E_i (site energies as natural frequencies)
# K_nm = |J_ij| (couplings)
omega_fmo_norm = (E_fmo - np.mean(E_fmo)) / np.std(E_fmo)
K_fmo_kuramoto = J_abs / np.max(J_abs)


def simulate_fmo_kuramoto(K_scale, dt=0.01, T=300, n_trials=15):
    n_steps = int(T / dt)
    R_trials = []
    for _ in range(n_trials):
        theta = np.random.uniform(0, 2 * np.pi, N_fmo)
        for _s in range(n_steps):
            dtheta = omega_fmo_norm.copy()
            for i in range(N_fmo):
                for j in range(N_fmo):
                    dtheta[i] += (
                        K_scale * K_fmo_kuramoto[i, j] * np.sin(theta[j] - theta[i]) / N_fmo
                    )
            theta += dtheta * dt
        z = np.mean(np.exp(1j * theta))
        R_trials.append(abs(z))
    return np.mean(R_trials), np.std(R_trials)


# Scan K to find K_c for FMO topology
K_scan = np.linspace(0.5, 8.0, 20)
R_fmo_scan = []
for K in K_scan:
    R, _ = simulate_fmo_kuramoto(K, n_trials=10)
    R_fmo_scan.append(R)
    print(f"K={K:.2f}: R={R:.3f}")

R_arr = np.array(R_fmo_scan)
idx_kc = np.argmin(np.abs(R_arr - 0.5))
K_c_fmo = K_scan[idx_kc]
print(f"\nK_c for FMO topology: {K_c_fmo:.2f}")
print("K_c for SCPN topology: ~2.7")
print(f"Ratio: {K_c_fmo / 2.7:.2f}")


# TEST 4: Transport efficiency (ENAQT)
print("\n" + "=" * 70)
print("TEST 4: NOISE-ASSISTED TRANSPORT (ENAQT)")
print("=" * 70)

# In FMO: excitation enters at BChl 1, exits at BChl 3 (trap)
# Efficiency = how much phase/energy reaches site 3
noise_scan = np.linspace(0.0, 5.0, 20)
transport_eff = []

for gamma in noise_scan:
    dt = 0.01
    T = 200
    n_steps = int(T / dt)
    n_trials = 20
    eff_trials = []

    for _ in range(n_trials):
        theta = np.zeros(N_fmo)
        theta[0] = np.pi  # excitation at BChl 1

        for _s in range(n_steps):
            dtheta = omega_fmo_norm.copy()
            for i in range(N_fmo):
                for j in range(N_fmo):
                    dtheta[i] += 3.0 * K_fmo_kuramoto[i, j] * np.sin(theta[j] - theta[i]) / N_fmo
            noise = gamma * np.random.randn(N_fmo) * np.sqrt(dt)
            theta += dtheta * dt + noise

        # Efficiency: phase transfer to site 3 (trap)
        eff = abs(np.sin(theta[2]))  # BChl 3 is index 2
        eff_trials.append(eff)

    transport_eff.append(np.mean(eff_trials))

transport_arr = np.array(transport_eff)
peak_idx = np.argmax(transport_arr)
gamma_opt = noise_scan[peak_idx]

print("Transport efficiency vs noise:")
for i in range(0, len(noise_scan), 4):
    print(f"  gamma={noise_scan[i]:.2f}: efficiency={transport_arr[i]:.3f}")

print(f"\nOptimal noise: gamma={gamma_opt:.2f}")
print(f"Zero-noise efficiency: {transport_arr[0]:.3f}")
print(f"Peak efficiency: {transport_arr[peak_idx]:.3f}")
enaqt = transport_arr[peak_idx] > transport_arr[0] * 1.05
print(f"ENAQT confirmed (>5% enhancement): {enaqt}")
if enaqt:
    enhancement = transport_arr[peak_idx] / max(transport_arr[0], 0.001)
    print(f"Enhancement factor: {enhancement:.2f}x")


# TEST 5: FMO vs random vs SCPN topology efficiency
print("\n" + "=" * 70)
print("TEST 5: TOPOLOGY COMPARISON FOR TRANSPORT")
print("=" * 70)

topologies = {
    "FMO (biological)": K_fmo_kuramoto,
    "SCPN (Paper 27)": K_nm_scpn / np.max(K_nm_scpn),
    "chain": np.diag(np.ones(N_fmo - 1), 1) + np.diag(np.ones(N_fmo - 1), -1),
    "all-to-all": np.ones((N_fmo, N_fmo)) - np.eye(N_fmo),
}

for name, K_top in topologies.items():
    dt = 0.01
    T = 200
    n_steps = int(T / dt)
    n_trials = 20
    eff_trials = []

    for _ in range(n_trials):
        theta = np.zeros(N_fmo)
        theta[0] = np.pi

        for _s in range(n_steps):
            dtheta = omega_fmo_norm.copy()
            for i in range(N_fmo):
                for j in range(N_fmo):
                    dtheta[i] += 3.0 * K_top[i, j] * np.sin(theta[j] - theta[i]) / N_fmo
            noise = gamma_opt * np.random.randn(N_fmo) * np.sqrt(dt)
            theta += dtheta * dt + noise

        eff = abs(np.sin(theta[2]))
        eff_trials.append(eff)

    print(f"  {name:25s}: efficiency = {np.mean(eff_trials):.3f}")


# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: FMO AS KURAMOTO SYSTEM")
print("=" * 70)

print(f"""
1. Structural match FMO vs SCPN: r={r_struct:.3f}, p={p_struct:.4f}
   {"CORRELATED" if p_struct < 0.05 else "NOT SIGNIFICANT"}

2. Decay profile match: r={r_decay:.3f}, p={p_decay:.4f}
   FMO has SPECIFIC long-range couplings (not simple exponential)

3. Eigenvalue correlation: r={r_ev:.3f}, p={p_ev:.4f}

4. K_c: FMO={K_c_fmo:.2f}, SCPN=2.7
   FMO topology {"easier" if K_c_fmo < 2.7 else "harder"} to synchronise

5. ENAQT: {"CONFIRMED" if enaqt else "NOT DETECTED"}
   Optimal noise gamma={gamma_opt:.2f}
   This connects directly to our stochastic resonance results

6. FMO topology evolved for TRANSPORT, not just SYNC
   The coupling matrix is shaped by ~3 billion years of evolution
   to maximise excitation transfer to the reaction centre
""")

results = {
    "structural_r": round(float(r_struct), 4),
    "structural_p": round(float(p_struct), 4),
    "decay_r": round(float(r_decay), 4),
    "decay_p": round(float(p_decay), 4),
    "eigenvalue_r": round(float(r_ev), 4),
    "K_c_fmo": round(float(K_c_fmo), 3),
    "K_c_scpn": 2.7,
    "enaqt_confirmed": bool(enaqt),
    "optimal_noise": round(float(gamma_opt), 3),
    "peak_efficiency": round(float(transport_arr[peak_idx]), 3),
}

print("\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
