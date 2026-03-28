# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Biochemical Validation of K_nm
#
# Tests whether the SCPN coupling matrix K_nm and natural frequencies
# omega_i correspond to measurable biochemical rate constants.
#
# The hypothesis: each K_nm coupling and omega_i frequency maps to a
# specific biological oscillatory process with a known timescale.

import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy"])

import numpy as np

OMEGA_N_16 = np.array([
    1.329, 2.610, 0.844, 1.520, 0.710, 3.780, 1.055, 0.625,
    2.210, 1.740, 0.480, 3.210, 0.915, 1.410, 2.830, 0.991,
])

# ============================================================
# 1. FREQUENCY-TO-BIOLOGY MAPPING
# ============================================================
print("=" * 70)
print("TEST 1: DO SCPN FREQUENCIES MATCH BIOLOGICAL TIMESCALES?")
print("=" * 70)
print()

# Known biological oscillation frequencies (rad/s for direct comparison)
bio_oscillations = {
    "L1 (Quantum bio)": {
        "radical_pair_lifetime_inv": 1e6,  # ~1 us lifetime -> 1 MHz
        "enzyme_tunnelling_rate": 1e3,     # ms timescale
        "microtubule_oscillation": 1e1,    # ~10 Hz (Hameroff)
        "SCPN_omega": OMEGA_N_16[0],
        "note": "SCPN omega_1 = 1.329 rad/s is macroscopic, not quantum"
    },
    "L2 (Neurochemical)": {
        "dopamine_release_rate": 5.0,       # ~5 Hz burst firing
        "serotonin_turnover": 0.1,          # ~0.1 Hz metabolic cycle
        "GABA_decay_constant": 100.0,       # ~100 Hz synaptic
        "SCPN_omega": OMEGA_N_16[1],
        "note": "omega_2 = 2.610 rad/s ~ 0.42 Hz, in metabolic range"
    },
    "L3 (Genomic)": {
        "gene_expression_oscillation": 0.001,  # ~mHz (circadian: 0.0000116 Hz)
        "epigenetic_modification_rate": 0.01,   # hours timescale
        "cell_cycle_frequency": 0.00001,        # ~24h period
        "SCPN_omega": OMEGA_N_16[2],
        "note": "omega_3 = 0.844 rad/s ~ 0.13 Hz, faster than genomic"
    },
    "L4 (Cellular sync)": {
        "gap_junction_conductance_Hz": 10.0,    # ~10 Hz voltage oscillations
        "calcium_wave_frequency": 0.1,          # ~0.1 Hz Ca2+ waves
        "cardiac_pacemaker": 1.2,               # ~1.2 Hz heart rhythm
        "SCPN_omega": OMEGA_N_16[3],
        "note": "omega_4 = 1.520 rad/s ~ 0.24 Hz, in Ca2+ wave range"
    },
}

for layer, data in bio_oscillations.items():
    print(f"--- {layer} ---")
    omega_scpn = data["SCPN_omega"]
    freq_Hz = omega_scpn / (2 * np.pi)
    period_s = 1.0 / freq_Hz if freq_Hz > 0 else float("inf")
    print(f"  SCPN omega = {omega_scpn:.3f} rad/s = {freq_Hz:.3f} Hz (period {period_s:.2f}s)")
    for key, val in data.items():
        if key not in ("SCPN_omega", "note"):
            ratio = omega_scpn / val if val > 0 else 0
            print(f"  {key}: {val} Hz, ratio SCPN/bio = {ratio:.2e}")
    print(f"  Note: {data['note']}")
    print()

# ============================================================
# 2. COUPLING STRENGTH VALIDATION
# ============================================================
print("=" * 70)
print("TEST 2: DO K_nm COUPLINGS MATCH GAP JUNCTION CONDUCTANCES?")
print("=" * 70)
print()

# Gap junction conductances (nanosiemens) for comparison
# Connexin-43 (most common): 60-100 pS per channel
# Typical cell has 100-1000 channels -> 6-100 nS total
# Normalised coupling: G_ij / G_max ~ K_nm / K_max

def build_knm(L, K_base=0.45, K_alpha=0.3):
    idx = np.arange(L)
    K = K_base * np.exp(-K_alpha * np.abs(idx[:, None] - idx[None, :]))
    anchors = {(0, 1): 0.302, (1, 2): 0.201, (2, 3): 0.252, (3, 4): 0.154}
    for (i, j), val in anchors.items():
        if i < L and j < L:
            K[i, j] = K[j, i] = val
    return K

K16 = build_knm(16)
K_norm = K16 / np.max(K16)

print("Paper 27 nearest-neighbour couplings (normalised):")
for i in range(min(8, 15)):
    print(f"  K[{i},{i+1}] = {K16[i, i+1]:.3f} (normalised: {K_norm[i, i+1]:.3f})")

print()
print("Comparison with Connexin-43 gap junction coupling decay:")
print("  Exponential model: K(d) = K_base * exp(-alpha * d)")
print(f"  SCPN: K_base = 0.45, alpha = 0.3 -> half-coupling at d = {np.log(2)/0.3:.1f} layers")
print(f"  Connexin-43: conductance drops ~50% per cell diameter (~20 um)")
print(f"  If 1 SCPN layer ~ 1 cell diameter: alpha=0.3 -> 50% at {np.log(2)/0.3:.1f} layers")
print(f"  Biological: 50% at 1 cell -> alpha_bio ~ {np.log(2):.2f}")
print(f"  SCPN alpha=0.3 < alpha_bio=0.69 -> SCPN layers couple MORE strongly")
print(f"  This is physically correct: SCPN layers span ontological scales, not cells")

# ============================================================
# 3. CHEMICAL REACTION RATE MATCHING
# ============================================================
print()
print("=" * 70)
print("TEST 3: NEUROTRANSMITTER CASCADE AS COUPLED OSCILLATOR CHAIN")
print("=" * 70)
print()

# Dopamine synthesis pathway: Tyr -> L-DOPA -> DA -> NE -> Epi
# Each enzymatic step has a characteristic turnover rate (k_cat)
enzyme_rates = {
    "Tyrosine hydroxylase (TH)": 0.3,      # s^-1 (rate-limiting)
    "AADC (L-DOPA -> DA)": 20.0,           # s^-1 (fast)
    "Dopamine beta-hydroxylase (DBH)": 2.0, # s^-1
    "PNMT (NE -> Epi)": 0.5,               # s^-1
}

print("Enzymatic turnover rates as oscillator frequencies:")
rates = list(enzyme_rates.values())
for (name, rate), omega_i in zip(enzyme_rates.items(), OMEGA_N_16[:4]):
    ratio = omega_i / rate
    print(f"  {name}: k_cat = {rate} s^-1, omega_SCPN = {omega_i:.3f}, ratio = {ratio:.3f}")

# Correlation between SCPN frequencies and enzyme rates?
omega_4 = OMEGA_N_16[:4]
corr = np.corrcoef(rates, omega_4)[0, 1]
print(f"\n  Pearson correlation (enzyme rates vs SCPN omega): r = {corr:.4f}")
print(f"  {'CORRELATED' if abs(corr) > 0.5 else 'UNCORRELATED'}")
print(f"  Note: SCPN frequencies are NOT derived from enzyme rates.")
print(f"  A nonzero correlation would be coincidental but interesting.")

# ============================================================
# 4. ION CHANNEL COUPLING AS K_nm
# ============================================================
print()
print("=" * 70)
print("TEST 4: ION CHANNEL ENERGETICS")
print("=" * 70)
print()

# Na/K-ATPase: 3Na+ out, 2K+ in, 1 ATP -> ADP + Pi
# Energy: ~50 kJ/mol ATP = 8.3e-20 J per molecule
# At 37C: kT = 4.28e-21 J
# Coupling energy: ATP/kT ~ 19.4 kT (strong coupling)

kT_37C = 1.38e-23 * 310  # J at body temperature
ATP_energy = 50e3 / 6.022e23  # J per molecule
coupling_kT = ATP_energy / kT_37C

print(f"Na/K-ATPase coupling energy:")
print(f"  ATP hydrolysis: {ATP_energy:.2e} J = {coupling_kT:.1f} kT")
print(f"  This is STRONG coupling (>> 1 kT)")
print()

# Voltage difference across membrane: ~70 mV
# Ion channel conductance: ~20 pS (single channel)
# Current: I = g * V = 20e-12 * 70e-3 = 1.4 pA
# Power: P = I*V = 1.4e-12 * 70e-3 = 9.8e-14 W
# Energy per cycle (~1 ms): E = P * tau = 9.8e-17 J = 22.9 kT

V_membrane = 70e-3  # V
g_channel = 20e-12  # S (siemens)
I_channel = g_channel * V_membrane
P_channel = I_channel * V_membrane
tau_cycle = 1e-3  # s
E_cycle = P_channel * tau_cycle
E_cycle_kT = E_cycle / kT_37C

print(f"Single ion channel per gating cycle:")
print(f"  Conductance: {g_channel*1e12:.0f} pS")
print(f"  Current: {I_channel*1e12:.1f} pA")
print(f"  Energy per cycle: {E_cycle:.2e} J = {E_cycle_kT:.1f} kT")
print()

# Map to SCPN K_nm
# K_nm = K_base = 0.45 in dimensionless units
# If K_nm ~ coupling_energy / kT: K_base ~ 0.45 corresponds to
# a coupling of ~0.45 kT -> WEAK coupling in biological terms
# This suggests SCPN K_nm represents normalised relative coupling,
# not absolute energy
print(f"SCPN K_base = 0.45 in dimensionless units")
print(f"Biological coupling: {coupling_kT:.1f} kT (ATP), {E_cycle_kT:.1f} kT (channel)")
print(f"Interpretation: K_nm is NORMALISED relative coupling (0-1 scale),")
print(f"not absolute energy. The exponential decay alpha=0.3 captures the")
print(f"TOPOLOGY of coupling, not the absolute strength.")

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 70)
print("SUMMARY: BIOCHEMICAL VALIDATION OF SCPN PARAMETERS")
print("=" * 70)
print()
print("1. SCPN frequencies (0.48-3.78 rad/s) are in the macroscopic range")
print("   (0.08-0.60 Hz), matching Ca2+ wave and metabolic oscillations.")
print("   They do NOT match quantum-scale rates (MHz-GHz).")
print()
print("2. Exponential coupling decay (alpha=0.3) is WEAKER than biological")
print("   gap junction decay (alpha~0.69), consistent with SCPN layers")
print("   spanning ontological scales (not individual cells).")
print()
print("3. Enzyme rate vs SCPN frequency correlation: r = {:.4f}".format(corr))
print("   SCPN frequencies are NOT derived from enzyme kinetics.")
print()
print("4. K_nm represents normalised relative coupling topology,")
print("   not absolute biochemical energies. The BKT transition at")
print("   K_c ~ 2.2-3.6 occurs in this normalised space.")

results = {
    "omega_Hz": (OMEGA_N_16 / (2 * np.pi)).tolist(),
    "enzyme_correlation": round(corr, 4),
    "coupling_alpha": 0.3,
    "bio_alpha": round(np.log(2), 3),
    "ATP_coupling_kT": round(coupling_kT, 1),
    "channel_coupling_kT": round(E_cycle_kT, 1),
}
print()
print(json.dumps(results, indent=2))
print("\nDone.")
