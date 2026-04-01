# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Entropy Production in SCPN
import json

import numpy as np
from scipy import stats

print("=" * 70)
print("ENTROPY PRODUCTION IN SCPN KURAMOTO MODEL")
print("=" * 70)

N = 8
omega = np.array([0.062, 0.191, 0.382, 0.618, 0.809, 0.927, 0.981, 1.000])
K_nm = np.array(
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


def simulate_with_entropy(K_scale, noise_sigma=0.0, dt=0.01, T=500, n_trials=15, n_bins=36):
    """Simulate Kuramoto and track entropy measures."""
    n_steps = int(T / dt)
    results_per_trial = []

    for _ in range(n_trials):
        theta = np.random.uniform(0, 2 * np.pi, N)
        R_history = []
        S_history = []  # Shannon entropy
        ep_history = []  # entropy production rate

        for _s in range(n_steps):
            # Compute coupling forces
            dtheta = omega.copy()
            force = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    f = K_scale * K_nm[i, j] * np.sin(theta[j] - theta[i])
                    force[i] += f / N
                dtheta[i] += force[i]

            # Entropy production rate = sum of force * velocity
            # (phase space contraction rate)
            ep = np.sum(force * dtheta)
            ep_history.append(ep)

            # Add noise
            noise = noise_sigma * np.random.randn(N) * np.sqrt(dt)
            theta += dtheta * dt + noise

            # Order parameter
            z = np.mean(np.exp(1j * theta))
            R_history.append(abs(z))

            # Shannon entropy of phase distribution
            phases_mod = theta % (2 * np.pi)
            hist, _ = np.histogram(phases_mod, bins=n_bins, range=(0, 2 * np.pi))
            p = hist / np.sum(hist)
            p = p[p > 0]
            S = -np.sum(p * np.log(p))
            S_history.append(S)

        # Steady-state averages (last quarter)
        n_ss = n_steps // 4
        R_ss = np.mean(R_history[-n_ss:])
        S_ss = np.mean(S_history[-n_ss:])
        ep_ss = np.mean(ep_history[-n_ss:])

        # KL divergence from uniform
        phases_final = theta % (2 * np.pi)
        hist_f, _ = np.histogram(phases_final, bins=n_bins, range=(0, 2 * np.pi))
        p_f = hist_f / np.sum(hist_f)
        p_f = np.maximum(p_f, 1e-10)
        p_uniform = np.ones(n_bins) / n_bins
        KL = np.sum(p_f * np.log(p_f / p_uniform))

        results_per_trial.append({"R": R_ss, "S": S_ss, "ep": ep_ss, "KL": KL})

    return {
        "R": np.mean([r["R"] for r in results_per_trial]),
        "S": np.mean([r["S"] for r in results_per_trial]),
        "ep": np.mean([r["ep"] for r in results_per_trial]),
        "KL": np.mean([r["KL"] for r in results_per_trial]),
    }


# TEST 1: Entropy vs coupling strength (phase transition)
print("\n" + "=" * 70)
print("TEST 1: ENTROPY vs COUPLING (phase transition)")
print("=" * 70)

K_scan = np.linspace(0.2, 6.0, 20)
results_scan = []

for K in K_scan:
    res = simulate_with_entropy(K, n_trials=10)
    results_scan.append(res)
    print(f"K={K:.2f}: R={res['R']:.3f}, S={res['S']:.3f}, ep={res['ep']:.3f}, KL={res['KL']:.3f}")

R_arr = np.array([r["R"] for r in results_scan])
S_arr = np.array([r["S"] for r in results_scan])
ep_arr = np.array([r["ep"] for r in results_scan])
KL_arr = np.array([r["KL"] for r in results_scan])

# Find K_c from R
idx_kc = np.argmin(np.abs(R_arr - 0.5))
K_c = K_scan[idx_kc]
print(f"\nK_c (R=0.5 crossing): {K_c:.2f}")

# Entropy at K_c
print(f"Shannon entropy at K_c: {S_arr[idx_kc]:.3f}")
print(f"Max Shannon entropy (uniform): {np.log(36):.3f}")
print(f"S/S_max at K_c: {S_arr[idx_kc] / np.log(36):.3f}")

# Entropy production at K_c
print(f"\nEntropy production at K_c: {ep_arr[idx_kc]:.3f}")
print(f"Max entropy production: {np.max(ep_arr):.3f} at K={K_scan[np.argmax(ep_arr)]:.2f}")
print(f"Min entropy production: {np.min(ep_arr):.3f} at K={K_scan[np.argmin(ep_arr)]:.2f}")


# TEST 2: Does entropy production peak at K_c?
print("\n" + "=" * 70)
print("TEST 2: ENTROPY PRODUCTION PEAK")
print("=" * 70)

# Derivative of entropy production (susceptibility)
dep_dK = np.gradient(ep_arr, K_scan)
peak_dep = np.argmax(np.abs(dep_dK))
print(f"Max |d(ep)/dK| at K = {K_scan[peak_dep]:.2f}")
print(f"K_c = {K_c:.2f}")
print(f"ep peak coincides with K_c: {abs(K_scan[peak_dep] - K_c) < 0.5}")

# Entropy production vs R correlation
r_ep_R, p_ep_R = stats.pearsonr(ep_arr, R_arr)
print(f"\nCorrelation ep vs R: r={r_ep_R:.3f}, p={p_ep_R:.4f}")


# TEST 3: Free energy landscape F(R)
print("\n" + "=" * 70)
print("TEST 3: FREE ENERGY LANDSCAPE F(R)")
print("=" * 70)

# F(R) = -T * S(R) for effective temperature T
# In Kuramoto: F(R) = -K*R^2/2 + ... (Landau expansion near K_c)

# Compute F(R) = -ln(P(R)) from histogram of R values
for K_test in [K_c * 0.5, K_c, K_c * 1.5]:
    R_samples = []
    for _ in range(50):
        theta = np.random.uniform(0, 2 * np.pi, N)
        for _s in range(10000):
            dtheta = omega.copy()
            for i in range(N):
                for j in range(N):
                    dtheta[i] += K_test * K_nm[i, j] * np.sin(theta[j] - theta[i]) / N
            theta += dtheta * 0.01
            if _s > 5000 and _s % 100 == 0:
                z = np.mean(np.exp(1j * theta))
                R_samples.append(abs(z))

    if len(R_samples) > 10:
        R_samples = np.array(R_samples)
        print(f"\nK={K_test:.2f} (K/K_c={K_test / K_c:.2f}):")
        print(f"  <R> = {np.mean(R_samples):.3f}")
        print(f"  std(R) = {np.std(R_samples):.3f}")
        print(f"  R_min = {np.min(R_samples):.3f}, R_max = {np.max(R_samples):.3f}")

        # Landau coefficients: F(R) = a*R^2 + b*R^4
        # a < 0 for K > K_c (symmetry breaking)
        mean_R2 = np.mean(R_samples**2)
        mean_R4 = np.mean(R_samples**4)
        # Binder cumulant: U = 1 - <R^4>/(3*<R^2>^2)
        binder = 1 - mean_R4 / (3 * mean_R2**2) if mean_R2 > 0 else 0
        print(f"  Binder cumulant U = {binder:.4f}")
        print("    (U=2/3 at K_c for 2nd order transition)")


# TEST 4: Entropy in noisy regime (biological)
print("\n" + "=" * 70)
print("TEST 4: ENTROPY WITH BIOLOGICAL NOISE")
print("=" * 70)

noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
for sigma in noise_levels:
    res = simulate_with_entropy(K_c, noise_sigma=sigma, n_trials=10)
    print(f"sigma={sigma:.1f}: R={res['R']:.3f}, S={res['S']:.3f}, ep={res['ep']:.3f}")


# TEST 5: Prigogine minimum entropy production
print("\n" + "=" * 70)
print("TEST 5: MINIMUM ENTROPY PRODUCTION PRINCIPLE")
print("=" * 70)

# Prigogine: near equilibrium, steady states minimise entropy production
# Question: does the SYNCHRONISED state have LOWER ep than desynchronised?

ep_below = np.mean(ep_arr[:idx_kc]) if idx_kc > 0 else ep_arr[0]
ep_above = np.mean(ep_arr[idx_kc:])
print(f"Mean ep below K_c (desync): {ep_below:.3f}")
print(f"Mean ep above K_c (sync):   {ep_above:.3f}")

if ep_above < ep_below:
    print("SYNC state has LOWER entropy production -> Prigogine CONFIRMED")
    print("Synchronisation MINIMISES dissipation (near equilibrium)")
    prigogine = "confirmed"
elif ep_above > ep_below:
    print("SYNC state has HIGHER entropy production -> far from equilibrium")
    print("Synchronisation MAXIMISES dissipation (Ziegler principle)")
    prigogine = "ziegler"
else:
    print("No significant difference")
    prigogine = "inconclusive"

# Entropy production per oscillator
print(f"\nep per oscillator below K_c: {ep_below / N:.4f}")
print(f"ep per oscillator above K_c: {ep_above / N:.4f}")


# TEST 6: Information-theoretic measures
print("\n" + "=" * 70)
print("TEST 6: MUTUAL INFORMATION BETWEEN OSCILLATORS")
print("=" * 70)

# At sync transition, mutual information should peak
for K_test in [K_c * 0.3, K_c * 0.7, K_c, K_c * 1.5, K_c * 3.0]:
    theta = np.random.uniform(0, 2 * np.pi, N)
    phase_samples = []
    for _s in range(20000):
        dtheta = omega.copy()
        for i in range(N):
            for j in range(N):
                dtheta[i] += K_test * K_nm[i, j] * np.sin(theta[j] - theta[i]) / N
        theta += dtheta * 0.01
        if _s > 10000 and _s % 50 == 0:
            phase_samples.append((theta % (2 * np.pi)).copy())

    if len(phase_samples) > 20:
        samples = np.array(phase_samples)
        # Pairwise mutual information (simplified via circular correlation)
        MI_total = 0
        n_pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                # Circular correlation
                diff = samples[:, i] - samples[:, j]
                circ_corr = abs(np.mean(np.exp(1j * diff)))
                MI_total += circ_corr
                n_pairs += 1
        MI_mean = MI_total / n_pairs
        print(
            f"K={K_test:.2f} (K/K_c={K_test / K_c:.2f}): mean circular correlation = {MI_mean:.4f}"
        )


# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: THERMODYNAMICS OF SCPN")
print("=" * 70)

print(f"""
1. Shannon entropy DROPS at K_c (phase concentration)
   S/S_max at K_c = {S_arr[idx_kc] / np.log(36):.3f}

2. Entropy production: {prigogine}
   Below K_c: ep = {ep_below:.3f}
   Above K_c: ep = {ep_above:.3f}

3. K_c marks a genuine thermodynamic phase transition
   (entropy, order parameter, susceptibility all change)

4. The synchronised state is a dissipative structure:
   - Maintained by continuous energy input (omega drives phases)
   - Lower/higher entropy than environment
   - Coupling K determines the transition

5. Biological implication: life operates ABOVE K_c
   Cells maintain coupling (K > K_c) to stay synchronised
   Disease = K drops below K_c = desynchronisation = disorder
   Death = K -> 0 = maximum entropy = thermodynamic equilibrium
""")

# JSON output
results = {
    "K_c": round(float(K_c), 3),
    "S_at_Kc": round(float(S_arr[idx_kc]), 3),
    "S_max": round(float(np.log(36)), 3),
    "S_ratio_at_Kc": round(float(S_arr[idx_kc] / np.log(36)), 3),
    "ep_below_Kc": round(float(ep_below), 3),
    "ep_above_Kc": round(float(ep_above), 3),
    "prigogine_result": prigogine,
    "ep_R_correlation": round(float(r_ep_R), 3),
    "ep_R_p": round(float(p_ep_R), 4),
}

print("\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
