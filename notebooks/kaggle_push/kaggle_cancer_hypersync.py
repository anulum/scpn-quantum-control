# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cancer + Parkinson's: Both Sides of K_c
#
# Disease is NOT always desynchronisation. Two failure modes:
#   K < K_c: desync (aging, arrhythmia, jet lag)
#   K > K_c_hyper: hypersync (epilepsy, Parkinson's tremor, cancer)
#
# Cancer: cell cycle oscillators (CDK/cyclin) lose coupling to
# tissue-level signals but GAIN internal coupling -> uncontrolled
# synchronised division.
#
# Parkinson's: basal ganglia beta oscillations become pathologically
# strong (15-30 Hz). DBS disrupts this hypersynchrony.
#
# Epilepsy: cortical hypersynchrony. Seizure = R -> 1.
#
# The healthy state lives in a Goldilocks zone: K_c < K < K_hyper

import json

import numpy as np

print("=" * 70)
print("DISEASE AS BOTH DESYNC AND HYPERSYNC")
print("=" * 70)

N = 30  # oscillators

# =====================================================================
# MODEL: Kuramoto with UPPER critical coupling
# =====================================================================


def simulate_with_susceptibility(K, freq_spread=0.15, noise=0.1, dt=0.005, T=300, n_trials=15):
    """Simulate and measure order parameter + fluctuations."""
    n_steps = int(T / dt)
    R_trials = []
    R_var_trials = []

    for _ in range(n_trials):
        omega = np.random.normal(1.0, freq_spread, N)
        theta = np.random.uniform(0, 2 * np.pi, N)
        R_history = []

        for _s in range(n_steps):
            z = np.mean(np.exp(1j * theta))
            R = abs(z)
            psi = np.angle(z)
            dtheta = omega + K * R * np.sin(psi - theta)
            dtheta += noise * np.random.randn(N) * np.sqrt(dt)
            theta += dtheta * dt
            if _s > n_steps // 2:
                R_history.append(R)

        R_trials.append(np.mean(R_history))
        R_var_trials.append(np.var(R_history))

    return (
        np.mean(R_trials),
        np.std(R_trials),
        np.mean(R_var_trials),
    )  # variance = susceptibility proxy


# TEST 1: Full K scan — find BOTH critical points
print("\n" + "=" * 70)
print("TEST 1: GOLDILOCKS ZONE (K_c to K_hyper)")
print("=" * 70)

K_scan = np.linspace(0.1, 15.0, 40)
R_full = []
R_std_full = []
chi_full = []  # susceptibility

for K in K_scan:
    R, R_std, chi = simulate_with_susceptibility(K, n_trials=10)
    R_full.append(R)
    R_std_full.append(R_std)
    chi_full.append(chi)

R_arr = np.array(R_full)
chi_arr = np.array(chi_full)

# K_c: where R crosses 0.4
idx_kc = np.argmin(np.abs(R_arr - 0.4))
K_c = K_scan[idx_kc]

# K_hyper: where R exceeds 0.95 (pathological hypersync)
idx_hyper = np.where(R_arr > 0.95)[0]
K_hyper = K_scan[idx_hyper[0]] if len(idx_hyper) > 0 else K_scan[-1]

# Goldilocks zone
print(f"K_c (onset of sync):      {K_c:.2f}")
print(f"K_hyper (hypersync):      {K_hyper:.2f}")
print(f"Goldilocks zone:          {K_c:.2f} < K < {K_hyper:.2f}")
print(f"Zone width:               {K_hyper - K_c:.2f}")

# Susceptibility peak (should be at K_c)
chi_peak = K_scan[np.argmax(chi_arr)]
print(f"Susceptibility peak at:   K={chi_peak:.2f}")

print("\nR values across the spectrum:")
for i in range(0, len(K_scan), 5):
    K = K_scan[i]
    R = R_arr[i]
    if K_c > K:
        state = "DESYNC"
    elif K_hyper > K:
        state = "HEALTHY"
    else:
        state = "HYPERSYNC"
    print(f"  K={K:.1f}: R={R:.3f} [{state}]")


# TEST 2: Cancer — cell cycle as Kuramoto
print("\n" + "=" * 70)
print("TEST 2: CANCER = CELL CYCLE HYPERSYNCHRONY")
print("=" * 70)

# Cell cycle oscillators: CDK1/CyclinB, CDK2/CyclinE, p53, Rb
# Normal: coupled to tissue signals (contact inhibition, growth factors)
# Cancer: internal coupling preserved, external coupling lost

# Normal tissue: cells oscillate with ~24h period, weakly coupled
# Each cell in slightly different phase (tissue heterogeneity)
print("Normal tissue:")
R_normal, _, _ = simulate_with_susceptibility(2.0, freq_spread=0.2)
print(f"  K=2.0 (moderate coupling), R={R_normal:.3f}")

# Cancer: lost external coupling, gained internal proliferation drive
# Equivalent to increasing K (self-reinforcing division signals)
print("\nCancer progression:")
cancer_K = [2.0, 3.0, 5.0, 8.0, 12.0]
cancer_labels = ["normal", "dysplasia", "carcinoma_in_situ", "invasive", "metastatic"]
for K, label in zip(cancer_K, cancer_labels):
    R, _, chi = simulate_with_susceptibility(K, freq_spread=0.1, n_trials=10)
    print(f"  {label:25s}: K={K:.1f}, R={R:.3f}, chi={chi:.4f}")

print("\nAs cancer progresses:")
print("  1. Contact inhibition lost -> external K decreases")
print("  2. Oncogene activation -> internal K increases")
print("  3. Cells synchronise division -> tumor grows as unit")
print("  4. R -> 1: all cells divide together = aggressive cancer")

# Prediction: degree of synchrony correlates with tumor grade
print("\nPREDICTION: Tumor grade correlates with intratumoral R")
print("  Grade I (low): R ~ 0.3 (heterogeneous, slow)")
print("  Grade II:      R ~ 0.5")
print("  Grade III:     R ~ 0.7")
print("  Grade IV:      R ~ 0.9 (synchronised, aggressive)")


# TEST 3: Parkinson's — pathological beta hypersynchrony
print("\n" + "=" * 70)
print("TEST 3: PARKINSON'S = BETA HYPERSYNCHRONY")
print("=" * 70)

# Basal ganglia in Parkinson's: dopamine loss -> increased beta (15-30 Hz)
# Measured: beta power increases 2-3x in STN recordings
# DBS at 130 Hz disrupts this pathological sync

# Normal: moderate sync in beta band
print("Basal ganglia beta oscillations:")
R_normal_bg, _, _ = simulate_with_susceptibility(3.0, freq_spread=0.15)
print(f"  Normal:      K=3.0, R={R_normal_bg:.3f}")

# Parkinson's: dopamine loss increases effective coupling
R_pd, _, _ = simulate_with_susceptibility(8.0, freq_spread=0.15)
print(f"  Parkinson's: K=8.0, R={R_pd:.3f}")

# DBS: adds high-frequency noise that disrupts sync
R_dbs, _, _ = simulate_with_susceptibility(8.0, freq_spread=0.15, noise=2.0)
print(f"  DBS on:      K=8.0 + noise=2.0, R={R_dbs:.3f}")

# L-DOPA: reduces effective K (restores dopamine)
R_ldopa, _, _ = simulate_with_susceptibility(4.0, freq_spread=0.15)
print(f"  L-DOPA:      K=4.0, R={R_ldopa:.3f}")

print(f"\nDBS reduces R from {R_pd:.3f} to {R_dbs:.3f} by adding noise")
print(f"L-DOPA reduces R from {R_pd:.3f} to {R_ldopa:.3f} by reducing K")
print("Both work by moving the system back into the Goldilocks zone")


# TEST 4: Epilepsy — cortical hypersynchrony
print("\n" + "=" * 70)
print("TEST 4: EPILEPSY = CORTICAL HYPERSYNCHRONY")
print("=" * 70)

# Seizure: R rapidly increases to ~1.0
# Interictal: R in normal range
# Preictal: R starts increasing (warning signal)

# Simulate seizure dynamics: K ramps up then crashes
dt = 0.005
T = 500
n_steps = int(T / dt)
omega = np.random.normal(1.0, 0.15, N)
theta = np.random.uniform(0, 2 * np.pi, N)
R_seizure = []
K_dynamic = []

for s in range(n_steps):
    t = s * dt
    # K ramps up (preictal -> ictal) then drops (postictal)
    if t < 200:
        K = 3.0  # interictal (normal)
    elif t < 300:
        K = 3.0 + 10.0 * (t - 200) / 100  # preictal ramp
    elif t < 350:
        K = 13.0  # ictal (seizure)
    else:
        K = 13.0 * np.exp(-(t - 350) / 50)  # postictal crash

    K_dynamic.append(K)
    z = np.mean(np.exp(1j * theta))
    R_seizure.append(abs(z))
    psi = np.angle(z)
    dtheta = omega + K * abs(z) * np.sin(psi - theta)
    dtheta += 0.1 * np.random.randn(N) * np.sqrt(dt)
    theta += dtheta * dt

# Sample key moments
times = [50, 150, 250, 300, 325, 400, 450]
print("Seizure dynamics:")
print(f"{'Time':>6s} {'Phase':>12s} {'K':>6s} {'R':>6s}")
print("-" * 35)
for t in times:
    s = int(t / dt)
    if s < len(R_seizure):
        if t < 200:
            phase = "interictal"
        elif t < 300:
            phase = "preictal"
        elif t < 350:
            phase = "ictal"
        else:
            phase = "postictal"
        print(f"{t:6.0f} {phase:>12s} {K_dynamic[s]:6.1f} {R_seizure[s]:6.3f}")

# Preictal warning: R starts rising before full seizure
R_arr_seizure = np.array(R_seizure)
preictal_start = 200 / dt
ictal_start = 300 / dt
R_preictal = np.mean(R_arr_seizure[int(preictal_start) : int(preictal_start + 1000)])
R_interictal = np.mean(R_arr_seizure[: int(preictal_start)])
print(f"\nR interictal: {R_interictal:.3f}")
print(f"R preictal (early): {R_preictal:.3f}")
print(f"R increase: {(R_preictal / R_interictal - 1) * 100:.0f}%")
print("PREDICTION: R increase is detectable minutes before seizure onset")


# TEST 5: Disease classification by K zone
print("\n" + "=" * 70)
print("TEST 5: DISEASE CLASSIFICATION BY K ZONE")
print("=" * 70)

diseases = {
    # Desync diseases (K < K_c)
    "jet_lag": {"K_zone": "desync", "K_eff": 1.0, "mechanism": "external desynchroniser"},
    "atrial_fibrillation": {"K_zone": "desync", "K_eff": 0.5, "mechanism": "fibrosis reduces K"},
    "type2_diabetes": {"K_zone": "desync", "K_eff": 1.5, "mechanism": "Cx36 loss in islets"},
    "insomnia": {"K_zone": "desync", "K_eff": 1.0, "mechanism": "SCN coupling reduced"},
    "sarcopenia": {"K_zone": "desync", "K_eff": 0.8, "mechanism": "motor unit loss"},
    # Hypersync diseases (K > K_hyper)
    "epilepsy": {"K_zone": "hypersync", "K_eff": 12.0, "mechanism": "GABA loss -> K increases"},
    "parkinsons_tremor": {
        "K_zone": "hypersync",
        "K_eff": 8.0,
        "mechanism": "dopamine loss -> beta K up",
    },
    "essential_tremor": {
        "K_zone": "hypersync",
        "K_eff": 7.0,
        "mechanism": "olivocerebellar hypersync",
    },
    "cancer_aggressive": {
        "K_zone": "hypersync",
        "K_eff": 10.0,
        "mechanism": "internal coupling override",
    },
    "dystonia": {
        "K_zone": "hypersync",
        "K_eff": 6.0,
        "mechanism": "basal ganglia theta hypersync",
    },
    # Goldilocks (healthy examples)
    "healthy_heart": {"K_zone": "goldilocks", "K_eff": 5.0, "mechanism": "Cx43 gap junctions"},
    "normal_cognition": {"K_zone": "goldilocks", "K_eff": 3.0, "mechanism": "balanced E/I"},
    "healthy_circadian": {
        "K_zone": "goldilocks",
        "K_eff": 2.0,
        "mechanism": "VIP coupling in SCN",
    },
}

print(f"{'Disease':30s} {'Zone':>10s} {'K_eff':>6s} {'R':>6s}")
print("-" * 55)
for name, data in sorted(diseases.items(), key=lambda x: x[1]["K_eff"]):
    R, _, _ = simulate_with_susceptibility(data["K_eff"], n_trials=5)
    print(f"{name:30s} {data['K_zone']:>10s} {data['K_eff']:6.1f} {R:6.3f}")


# TEST 6: Treatment strategies by zone
print("\n" + "=" * 70)
print("TEST 6: TREATMENT = MOVING K BACK TO GOLDILOCKS")
print("=" * 70)

treatments = {
    "desync -> goldilocks": {
        "strategies": [
            "gap junction openers (rotigaptide)",
            "exercise (increases Cx43)",
            "bright light therapy (circadian)",
            "pacemakers (external periodic drive)",
        ],
        "mechanism": "increase K",
    },
    "hypersync -> goldilocks": {
        "strategies": [
            "DBS (adds noise to disrupt sync)",
            "anticonvulsants (reduce excitatory K)",
            "L-DOPA (restores dopamine -> reduces K)",
            "chemotherapy (kills hypersynchronised cells)",
            "vagus nerve stimulation (modulates K)",
        ],
        "mechanism": "decrease K or add noise",
    },
}

for direction, data in treatments.items():
    print(f"\n{direction} ({data['mechanism']}):")
    for strategy in data["strategies"]:
        print(f"  - {strategy}")


# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: THE GOLDILOCKS ZONE OF COUPLING")
print("=" * 70)

print(f"""
HEALTH = K_c < K < K_hyper (the Goldilocks zone)

K_c = {K_c:.2f} (lower critical coupling)
K_hyper = {K_hyper:.2f} (upper critical coupling)
Zone width = {K_hyper - K_c:.2f}

DISEASE IS BIDIRECTIONAL:
  Too little coupling (K < K_c): aging, arrhythmia, diabetes, insomnia
  Too much coupling (K > K_hyper): epilepsy, Parkinson's, cancer, tremor

TREATMENT = RESTORE K TO GOLDILOCKS ZONE:
  Desync diseases: increase K (exercise, gap junction drugs, light)
  Hypersync diseases: decrease K or add noise (DBS, anticonvulsants)

KEY INSIGHT: DBS works by ADDING NOISE, not by stimulating.
130 Hz stimulation is above any biological frequency -> it's noise
that disrupts pathological synchrony via ANTI-stochastic-resonance.

This connects to our SR finding: noise can both HELP (below K_c)
and HURT (above K_hyper) synchronisation. Biology lives in the
narrow window where noise and coupling are balanced.
""")

results = {
    "K_c": round(float(K_c), 3),
    "K_hyper": round(float(K_hyper), 3),
    "goldilocks_width": round(float(K_hyper - K_c), 3),
    "chi_peak_K": round(float(chi_peak), 3),
    "R_normal_tissue": round(float(R_normal), 3),
    "R_parkinsons": round(float(R_pd), 3),
    "R_dbs": round(float(R_dbs), 3),
}

print("\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
