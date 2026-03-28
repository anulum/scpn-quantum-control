# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Neural Oscillation Bands as SCPN Layers
#
# HYPOTHESIS: EEG frequency bands (delta/theta/alpha/beta/gamma)
# map to SCPN oscillator layers. Cross-frequency coupling
# (theta-gamma nesting) is K_nm between layers.
#
# Tests:
# 1. Do EEG band centre frequencies match SCPN omega spacing?
# 2. Does cross-frequency coupling structure match K_nm decay?
# 3. Phase-amplitude coupling as Kuramoto order parameter
# 4. Comparison to Buzsáki & Draguhn (2004) frequency architecture

import json

import numpy as np
from scipy import stats

print("=" * 70)
print("NEURAL OSCILLATION BANDS AS SCPN LAYERS")
print("=" * 70)

# EEG frequency bands (Hz) — canonical ranges
# Source: Buzsáki & Draguhn, Science 2004; Buzsáki, Rhythms of the Brain 2006
eeg_bands = {
    "slow_oscillation": {"range": (0.1, 1.0), "centre": 0.5},
    "delta": {"range": (1.0, 4.0), "centre": 2.5},
    "theta": {"range": (4.0, 8.0), "centre": 6.0},
    "alpha": {"range": (8.0, 13.0), "centre": 10.5},
    "beta": {"range": (13.0, 30.0), "centre": 21.5},
    "gamma_low": {"range": (30.0, 60.0), "centre": 45.0},
    "gamma_high": {"range": (60.0, 100.0), "centre": 80.0},
    "ripple": {"range": (100.0, 250.0), "centre": 175.0},
}

# SCPN parameters (8 oscillators)
omega_scpn = np.array([0.062, 0.191, 0.382, 0.618, 0.809, 0.927, 0.981, 1.000])

# TEST 1: Frequency ratio structure
print("\n" + "=" * 70)
print("TEST 1: FREQUENCY RATIO STRUCTURE")
print("=" * 70)

centres = np.array([b["centre"] for b in eeg_bands.values()])
names = list(eeg_bands.keys())

print("\nEEG band ratios (consecutive):")
eeg_ratios = []
for i in range(len(centres) - 1):
    ratio = centres[i + 1] / centres[i]
    eeg_ratios.append(ratio)
    print(f"  {names[i]:20s} → {names[i + 1]:20s}: {ratio:.3f}")

print(f"\nMean ratio: {np.mean(eeg_ratios):.3f}")
print(f"Std ratio:  {np.std(eeg_ratios):.3f}")

# Buzsáki's observation: each band is ~e (2.718) times the previous
# This is the "natural logarithmic" spacing
print(f"\ne (Euler's number): {np.e:.3f}")
print(f"Mean EEG ratio:     {np.mean(eeg_ratios):.3f}")
print(f"Match to e:         {abs(np.mean(eeg_ratios) - np.e) / np.e * 100:.1f}% off")

# SCPN omega ratios
print("\nSCPN omega ratios (consecutive):")
scpn_ratios = []
for i in range(len(omega_scpn) - 1):
    ratio = omega_scpn[i + 1] / omega_scpn[i]
    scpn_ratios.append(ratio)
    print(f"  omega[{i}]→omega[{i + 1}]: {ratio:.3f}")

print(f"\nMean SCPN ratio: {np.mean(scpn_ratios):.3f}")

# Compare distributions
ks_stat, ks_p = stats.ks_2samp(eeg_ratios, scpn_ratios)
print(f"\nKS test (EEG ratios vs SCPN ratios): D={ks_stat:.3f}, p={ks_p:.4f}")
if ks_p < 0.05:
    print("DIFFERENT distributions (p < 0.05)")
else:
    print("Cannot distinguish distributions (p >= 0.05)")

# TEST 2: Log-frequency spacing
print("\n" + "=" * 70)
print("TEST 2: LOG-FREQUENCY SPACING")
print("=" * 70)

log_centres = np.log(centres)
log_omega = np.log(omega_scpn)

# Linear fit to log-frequencies
eeg_spacing = np.diff(log_centres)
scpn_spacing = np.diff(log_omega)

print("EEG log-frequency spacing:")
for i, s in enumerate(eeg_spacing):
    print(f"  {names[i]:20s} → {names[i + 1]:20s}: {s:.3f}")
print(f"  Mean: {np.mean(eeg_spacing):.3f}, Std: {np.std(eeg_spacing):.3f}")
print(f"  CV (coefficient of variation): {np.std(eeg_spacing) / np.mean(eeg_spacing):.3f}")

print("\nSCPN log-omega spacing:")
for i, s in enumerate(scpn_spacing):
    print(f"  omega[{i}]→omega[{i + 1}]: {s:.3f}")
print(f"  Mean: {np.mean(scpn_spacing):.3f}, Std: {np.std(scpn_spacing):.3f}")
print(f"  CV: {np.std(scpn_spacing) / np.mean(scpn_spacing):.3f}")

# Key question: is SCPN spacing uniform in log space?
# EEG is roughly uniform (Buzsáki) — is SCPN?
print(f"\nEEG log-spacing uniformity (CV): {np.std(eeg_spacing) / np.mean(eeg_spacing):.3f}")
print(f"SCPN log-spacing uniformity (CV): {np.std(scpn_spacing) / np.mean(scpn_spacing):.3f}")
print("Lower CV = more uniform. Perfect uniformity = 0.")

# TEST 3: Cross-frequency coupling structure
print("\n" + "=" * 70)
print("TEST 3: CROSS-FREQUENCY COUPLING")
print("=" * 70)

# Known CFC pairs in neuroscience:
# theta-gamma: strongest, most studied (hippocampal memory)
# alpha-gamma: visual cortex
# delta-theta: sleep spindles
# beta-gamma: motor cortex
# These represent K_nm between bands

# Model: coupling strength decays with frequency ratio distance
# Compare to SCPN K_nm decay
K_nm_scpn = np.array(
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

# Extract coupling decay with distance
distances = []
couplings = []
for i in range(8):
    for j in range(i + 1, 8):
        distances.append(abs(i - j))
        couplings.append(K_nm_scpn[i, j])

distances = np.array(distances)
couplings = np.array(couplings)

# Fit exponential decay
log_couplings = np.log(couplings + 1e-10)
slope, intercept, r_val, p_val, se = stats.linregress(distances, log_couplings)
alpha_decay = -slope

print(f"SCPN K_nm decay: alpha = {alpha_decay:.3f}")
print(f"K ~ exp(-{alpha_decay:.3f} * distance), R² = {r_val**2:.4f}")

# Known CFC strengths from literature (normalised, approximate)
# Canolty & Knight 2010, Tort et al. 2010
cfc_pairs = {
    "theta-gamma": {"distance": 3, "strength": 0.9},  # strongest
    "alpha-gamma": {"distance": 2, "strength": 0.6},
    "delta-theta": {"distance": 1, "strength": 0.7},
    "beta-gamma": {"distance": 1, "strength": 0.5},
    "delta-gamma": {"distance": 4, "strength": 0.3},
    "theta-ripple": {"distance": 5, "strength": 0.4},  # sharp-wave ripple
}

print("\nKnown CFC pairs vs SCPN prediction:")
cfc_dists = []
cfc_strengths = []
scpn_pred = []
for name, data in cfc_pairs.items():
    pred = np.exp(intercept + slope * data["distance"])
    cfc_dists.append(data["distance"])
    cfc_strengths.append(data["strength"])
    scpn_pred.append(pred)
    print(f"  {name:20s}: observed={data['strength']:.2f}, SCPN pred={pred:.3f}")

r_cfc, p_cfc = stats.pearsonr(cfc_strengths, scpn_pred)
print(f"\nCorrelation (observed CFC vs SCPN prediction): r={r_cfc:.3f}, p={p_cfc:.4f}")

# TEST 4: Phase-amplitude coupling as Kuramoto R
print("\n" + "=" * 70)
print("TEST 4: PHASE-AMPLITUDE COUPLING SIMULATION")
print("=" * 70)

# Simulate multi-band oscillator system
# 8 oscillators with EEG-like frequencies
omega_eeg_norm = centres / centres[-1]  # normalise to [0, 1]


def simulate_eeg_kuramoto(K_scale, dt=0.001, T=50, n_trials=10):
    """Kuramoto with EEG-like frequencies and SCPN coupling."""
    n_steps = int(T / dt)
    R_total_trials = []

    for _ in range(n_trials):
        theta = np.random.uniform(0, 2 * np.pi, len(centres))
        theta_history = np.zeros((n_steps, len(centres)))

        for step in range(n_steps):
            dtheta = omega_eeg_norm * 2 * np.pi  # actual angular frequency
            for i in range(len(centres)):
                coupling = 0.0
                for j in range(len(centres)):
                    if i != j:
                        # Use SCPN-like distance-dependent coupling
                        d = abs(i - j)
                        K_ij = K_scale * np.exp(-alpha_decay * d)
                        coupling += K_ij * np.sin(theta[j] - theta[i])
                dtheta[i] += coupling / len(centres)
            theta += dtheta * dt
            theta_history[step] = theta % (2 * np.pi)

        z = np.mean(np.exp(1j * theta))
        R_total_trials.append(abs(z))

    return np.mean(R_total_trials), np.std(R_total_trials)


K_scan = np.linspace(0.1, 10.0, 20)
R_eeg_model = []
for K in K_scan:
    r, r_std = simulate_eeg_kuramoto(K, n_trials=5)
    R_eeg_model.append(r)
    print(f"K={K:.1f}: R={r:.3f}")

R_arr = np.array(R_eeg_model)
idx_half = np.argmin(np.abs(R_arr - 0.5))
K_c_eeg = K_scan[idx_half]
print(f"\nK_c for EEG-frequency system: {K_c_eeg:.2f}")
print(f"K_c for SCPN system:          {2.7:.2f}")
print(f"Ratio:                        {K_c_eeg / 2.7:.2f}")

# TEST 5: Frequency architecture comparison
print("\n" + "=" * 70)
print("TEST 5: FREQUENCY ARCHITECTURE — BUZSÁKI RULE")
print("=" * 70)

# Buzsáki's observation: brain oscillations span ~4 orders of magnitude
# with roughly logarithmic spacing. Each band ~3x the previous.
# Is this a consequence of Kuramoto coupling topology?

print("\nFrequency spans:")
print(
    f"  EEG:  {centres[0]:.1f} Hz → {centres[-1]:.0f} Hz ({centres[-1] / centres[0]:.0f}x, {np.log10(centres[-1] / centres[0]):.1f} decades)"
)
print(
    f"  SCPN: {omega_scpn[0]:.3f} → {omega_scpn[-1]:.3f} ({omega_scpn[-1] / omega_scpn[0]:.1f}x, {np.log10(omega_scpn[-1] / omega_scpn[0]):.1f} decades)"
)

# Information capacity per band
print("\nInformation capacity (bandwidth / centre):")
for name, data in eeg_bands.items():
    bw = data["range"][1] - data["range"][0]
    rel_bw = bw / data["centre"]
    print(f"  {name:20s}: BW={bw:6.1f} Hz, rel_BW={rel_bw:.3f}")

# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: NEURAL OSCILLATIONS AS SCPN")
print("=" * 70)

# Key findings
eeg_scpn_match = ks_p >= 0.05  # ratio distributions not distinguishable
cfc_scpn_match = p_cfc < 0.05  # CFC correlates with SCPN coupling

print(
    f"\n1. Frequency ratio match (KS test): p={ks_p:.4f} → {'COMPATIBLE' if eeg_scpn_match else 'DIFFERENT'}"
)
print(
    f"2. CFC vs SCPN coupling: r={r_cfc:.3f}, p={p_cfc:.4f} → {'CORRELATED' if cfc_scpn_match else 'NOT SIGNIFICANT'}"
)
print(f"3. K_c ratio (EEG/SCPN): {K_c_eeg / 2.7:.2f}")
print(f"4. EEG mean spacing ratio: {np.mean(eeg_ratios):.3f} (Buzsáki: ~e={np.e:.3f})")
print(f"5. SCPN mean spacing ratio: {np.mean(scpn_ratios):.3f}")
print(
    f"6. Log-spacing uniformity: EEG CV={np.std(eeg_spacing) / np.mean(eeg_spacing):.3f}, SCPN CV={np.std(scpn_spacing) / np.mean(scpn_spacing):.3f}"
)

print("\n--- INTERPRETATION ---")
if eeg_scpn_match:
    print("EEG and SCPN share compatible frequency architectures.")
    print("Both use logarithmic spacing — a signature of scale-invariant coupling.")
else:
    print("EEG and SCPN have DIFFERENT frequency architectures.")
    print("EEG uses ~e spacing, SCPN uses non-uniform spacing.")
    print("Different coupling topologies produce different frequency distributions.")

if cfc_scpn_match:
    print("\nCross-frequency coupling MATCHES SCPN exponential decay.")
    print("Brain bands couple like Kuramoto oscillators with distance-dependent K_nm.")
else:
    print("\nCFC does NOT match SCPN prediction.")
    print("Brain coupling may follow a different topology than SCPN's exponential decay.")

# JSON output
results = {
    "eeg_mean_ratio": round(float(np.mean(eeg_ratios)), 3),
    "scpn_mean_ratio": round(float(np.mean(scpn_ratios)), 3),
    "eeg_scpn_ks_p": round(float(ks_p), 4),
    "cfc_scpn_correlation": round(float(r_cfc), 3),
    "cfc_scpn_p": round(float(p_cfc), 4),
    "K_c_eeg": round(float(K_c_eeg), 3),
    "K_c_scpn": 2.7,
    "eeg_log_cv": round(float(np.std(eeg_spacing) / np.mean(eeg_spacing)), 3),
    "scpn_log_cv": round(float(np.std(scpn_spacing) / np.mean(scpn_spacing)), 3),
    "eeg_match_compatible": bool(eeg_scpn_match),
    "cfc_match": bool(cfc_scpn_match),
}

print("\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
