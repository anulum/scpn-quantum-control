# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Extended Enzyme Rate Validation
#
# Tests the r=0.89 enzyme-SCPN frequency correlation with a much larger
# dataset of biological oscillatory processes across all 16 SCPN layers.
# The question: is the correlation real or a n=4 artefact?

import json
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy", "scipy"])

import numpy as np
from scipy import stats

OMEGA_N_16 = np.array([
    1.329, 2.610, 0.844, 1.520, 0.710, 3.780, 1.055, 0.625,
    2.210, 1.740, 0.480, 3.210, 0.915, 1.410, 2.830, 0.991,
])

# ============================================================
# COMPREHENSIVE BIOLOGICAL RATE DATABASE
# Each SCPN layer mapped to multiple measurable biological rates
# All rates in s^-1 (Hz) from published literature
# ============================================================

bio_rates_per_layer = {
    0: {  # L1: Quantum biology
        "name": "Quantum Biology",
        "rates": {
            "cryptochrome_radical_pair_decay": 1e6,      # Ritz 2004, ~1 us
            "FMN_triplet_lifetime_inv": 5e4,             # flavin photochemistry
            "enzyme_H_tunnelling_KIE": 1e3,              # kinetic isotope effect
            "microtubule_GHz_mode": 1e9,                 # Hameroff (disputed)
            "photosynthetic_exciton_transfer": 1e12,     # Engel 2007, fs timescale
            "retinal_isomerisation": 5e11,               # 200 fs
            "proton_tunnelling_ADH": 50,                 # alcohol dehydrogenase
            "superoxide_dismutase": 1e9,                 # diffusion-limited
        }
    },
    1: {  # L2: Neurochemical
        "name": "Neurochemical",
        "rates": {
            "tyrosine_hydroxylase_kcat": 0.3,            # rate-limiting, DA synthesis
            "AADC_kcat": 20.0,                           # L-DOPA -> DA
            "DBH_kcat": 2.0,                             # DA -> NE
            "PNMT_kcat": 0.5,                            # NE -> Epi
            "MAO_A_kcat": 12.0,                          # monoamine oxidase A
            "MAO_B_kcat": 4.0,                           # monoamine oxidase B
            "COMT_kcat": 1.5,                            # catechol-O-methyltransferase
            "acetylcholinesterase_kcat": 25000,          # fastest enzyme
            "glutamate_decarboxylase_kcat": 3.0,         # GABA synthesis
            "tryptophan_hydroxylase_kcat": 0.2,          # serotonin synthesis
            "choline_acetyltransferase_kcat": 50,        # ACh synthesis
            "dopamine_transporter_rate": 5.0,            # reuptake
        }
    },
    2: {  # L3: Genomic
        "name": "Genomic",
        "rates": {
            "RNA_polymerase_II_rate": 30,                # nucleotides/s
            "ribosome_translation_rate": 6,              # amino acids/s
            "DNMT3a_kcat": 0.002,                        # DNA methyltransferase
            "TET2_kcat": 0.01,                           # demethylation
            "histone_acetyltransferase": 0.5,            # HAT
            "histone_deacetylase": 0.1,                  # HDAC
            "circadian_clock_period_inv": 1.16e-5,       # 1/86400 Hz
            "cell_cycle_G1_inv": 5.6e-5,                 # ~5 hours
            "p53_oscillation": 5.6e-4,                   # ~30 min period
            "NF_kB_oscillation": 1.1e-3,                 # ~15 min period
        }
    },
    3: {  # L4: Cellular synchronisation
        "name": "Cellular Sync",
        "rates": {
            "connexin43_channel_conductance_rate": 10,   # gating ~10 Hz
            "calcium_wave_intercellular": 0.1,           # ~0.1 Hz
            "cardiac_pacemaker_SA_node": 1.2,            # ~72 bpm
            "smooth_muscle_slow_wave": 0.05,             # ~3/min
            "pancreatic_islet_oscillation": 0.03,        # ~2/min
            "Na_K_ATPase_turnover": 100,                 # 100 cycles/s
            "Ca_ATPase_SERCA_rate": 40,                  # SERCA pump
            "IP3_receptor_opening_rate": 5,              # ~5 Hz
            "gap_junction_permeability_rate": 1.0,       # ~1 Hz effective
        }
    },
    4: {  # L5: Organismal self
        "name": "Organismal",
        "rates": {
            "breathing_rate": 0.25,                      # ~15/min
            "heart_rate": 1.2,                           # ~72 bpm
            "EEG_alpha_rhythm": 10,                      # 8-12 Hz
            "EEG_theta_rhythm": 6,                       # 4-8 Hz
            "EEG_gamma_rhythm": 40,                      # 30-100 Hz
            "blink_rate": 0.25,                          # ~15/min
            "peristalsis": 0.05,                         # ~3/min
            "cortisol_ultradian": 2.8e-4,                # ~1 hour period
        }
    },
    5: {  # L6: Biosphere
        "name": "Biosphere",
        "rates": {
            "circadian_rhythm": 1.16e-5,                 # 24h
            "ultradian_90min": 1.85e-4,                  # BRAC cycle
            "tidal_rhythm": 2.24e-5,                     # 12.4h
            "lunar_cycle": 3.8e-7,                       # 29.5 days
            "seasonal_melatonin": 3.2e-8,                # ~1 year
            "Schumann_resonance": 7.83,                  # Hz
        }
    },
    6: {  # L7: Symbolic
        "name": "Symbolic",
        "rates": {
            "speech_syllable_rate": 4,                   # ~4 Hz
            "reading_fixation_rate": 3.5,                # ~3.5 Hz
            "working_memory_refresh": 5,                 # ~5 Hz (Lisman)
            "attentional_blink": 3,                      # ~3 Hz
            "phonological_loop": 2,                      # ~2 Hz
        }
    },
    7: {  # L8: Cosmic phase-locking
        "name": "Cosmic",
        "rates": {
            "Earth_rotation": 1.16e-5,                   # 1/86400 Hz
            "solar_cycle_11yr": 2.9e-9,                  # ~11 years
            "galactic_year_inv": 4.8e-16,                # ~225 Myr
            "CMB_temperature_fluctuation": 1e-18,        # effectively static
        }
    },
}

# ============================================================
# ANALYSIS: CORRELATION PER LAYER
# ============================================================
print("=" * 70)
print("EXTENDED ENZYME/BIOLOGICAL RATE VALIDATION")
print("Question: does r=0.89 survive with more data points?")
print("=" * 70)

all_bio_rates = []
all_scpn_omega = []
layer_correlations = {}

for layer_idx in range(min(8, 16)):
    if layer_idx not in bio_rates_per_layer:
        continue
    layer = bio_rates_per_layer[layer_idx]
    omega_scpn = OMEGA_N_16[layer_idx]
    rates = list(layer.get("rates", {}).values())

    # Log-space comparison (rates span many orders of magnitude)
    log_rates = [np.log10(r) for r in rates if r > 0]
    log_omega = np.log10(omega_scpn) if omega_scpn > 0 else 0

    # Mean log-rate for this layer
    mean_log_rate = np.mean(log_rates) if log_rates else 0
    median_rate = np.median(rates) if rates else 0

    all_bio_rates.append(mean_log_rate)
    all_scpn_omega.append(log_omega)

    print(f"\nL{layer_idx+1} ({layer['name']}): omega = {omega_scpn:.3f} rad/s")
    print(f"  {len(rates)} biological rates: {[f'{r:.2e}' for r in sorted(rates)[:5]]}...")
    print(f"  Log-mean rate: 10^{mean_log_rate:.2f} = {10**mean_log_rate:.2e} Hz")
    print(f"  Log omega: {log_omega:.3f}")
    print(f"  Median rate: {median_rate:.2e} Hz")

# ============================================================
# OVERALL CORRELATIONS
# ============================================================
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

all_bio_rates = np.array(all_bio_rates)
all_scpn_omega = np.array(all_scpn_omega)
n_layers = len(all_bio_rates)

# Pearson on log-scale
r_log, p_log = stats.pearsonr(all_bio_rates, all_scpn_omega)
print(f"\n1. Pearson (log rates vs log omega, n={n_layers}):")
print(f"   r = {r_log:.4f}, p = {p_log:.4f}")
print(f"   {'SIGNIFICANT' if p_log < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

# Spearman rank correlation (nonparametric)
rho, p_rho = stats.spearmanr(all_bio_rates, all_scpn_omega)
print(f"\n2. Spearman rank (n={n_layers}):")
print(f"   rho = {rho:.4f}, p = {p_rho:.4f}")
print(f"   {'SIGNIFICANT' if p_rho < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

# Original 4-enzyme correlation (reproduce)
enzyme_rates_4 = [0.3, 20.0, 2.0, 0.5]
omega_4 = OMEGA_N_16[:4].tolist()
r_orig, p_orig = stats.pearsonr(enzyme_rates_4, omega_4)
print(f"\n3. Original 4-enzyme correlation (reproduce):")
print(f"   r = {r_orig:.4f}, p = {p_orig:.4f}")
print(f"   {'SIGNIFICANT' if p_orig < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")
print(f"   Note: n=4 has very low statistical power")

# Permutation test for significance
print(f"\n4. Permutation test (10000 shuffles):")
rng = np.random.default_rng(42)
n_perm = 10000
r_observed = abs(r_log)
count_exceed = 0
for _ in range(n_perm):
    shuffled = rng.permutation(all_bio_rates)
    r_perm = abs(np.corrcoef(shuffled, all_scpn_omega)[0, 1])
    if r_perm >= r_observed:
        count_exceed += 1
p_perm = count_exceed / n_perm
print(f"   Observed |r| = {r_observed:.4f}")
print(f"   Permutation p = {p_perm:.4f} ({count_exceed}/{n_perm} exceeded)")
print(f"   {'SIGNIFICANT' if p_perm < 0.05 else 'NOT SIGNIFICANT'}")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print()
if abs(r_log) > 0.5 and p_log < 0.05:
    print("CONFIRMED: SCPN frequencies correlate with biological rate hierarchies")
    print(f"across {n_layers} layers (r={r_log:.3f}, p={p_log:.4f}).")
elif abs(r_log) > 0.3:
    print("WEAK CORRELATION: suggestive but not conclusive.")
    print(f"r={r_log:.3f}, p={p_log:.4f} with n={n_layers} layers.")
else:
    print("NO CORRELATION: the r=0.89 from n=4 was a small-sample artefact.")
    print(f"Extended analysis (n={n_layers}): r={r_log:.3f}, p={p_log:.4f}.")

print()
print("CAVEAT: biological rates span 20+ orders of magnitude per layer.")
print("The mean/median rate per layer compresses this diversity.")
print("The correlation tests whether SCPN frequencies track the")
print("ORDER OF MAGNITUDE of each layer's characteristic timescale,")
print("not the exact rate of any single process.")

results = {
    "n_layers": n_layers,
    "pearson_r": round(r_log, 4),
    "pearson_p": round(p_log, 4),
    "spearman_rho": round(rho, 4),
    "spearman_p": round(p_rho, 4),
    "permutation_p": round(p_perm, 4),
    "original_4enzyme_r": round(r_orig, 4),
    "original_4enzyme_p": round(p_orig, 4),
    "log_bio_rates": all_bio_rates.tolist(),
    "log_scpn_omega": all_scpn_omega.tolist(),
}
print()
print(json.dumps(results, indent=2))
print("\nDone.")
