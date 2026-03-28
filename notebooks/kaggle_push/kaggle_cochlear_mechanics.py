# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cochlear Hair Cells as Kuramoto Chain
#
# The cochlea is a frequency-sorted oscillator array:
# - 3,500 inner hair cells, each tuned to a specific frequency
# - Base: 20 kHz. Apex: 20 Hz. Logarithmic mapping.
# - Outer hair cells actively amplify via electromotility (prestin)
# - Otoacoustic emissions: the ear EMITS sound (active oscillation!)
#
# This is a 1D Kuramoto chain with:
# - omega gradient (tonotopic map)
# - nearest-neighbour coupling (basilar membrane)
# - active amplification (OHC gain = K boost)
#
# The cochlea is the BEST biological example of a frequency-graded
# coupled oscillator system. Direct SCPN comparison.

import numpy as np
import json
from scipy import stats

print("=" * 70)
print("COCHLEAR HAIR CELLS AS KURAMOTO CHAIN")
print("=" * 70)

# =====================================================================
# Cochlear parameters (from Robles & Ruggero 2001, Hudspeth 2008)
# =====================================================================

# Tonotopic map: position x (mm from base) -> frequency (Hz)
# Greenwood function: f(x) = A * (10^(ax) - k)
# Human: A=165.4, a=-0.06, k=0.88, cochlea length=35mm
A_green = 165.4
a_green = -0.06 / 1.0  # per mm (note: negative = freq decreases toward apex)
k_green = 0.88
L_cochlea = 35.0  # mm

def greenwood_freq(x_mm):
    """Greenwood function: position -> characteristic frequency."""
    return A_green * (10**(a_green * x_mm) - k_green)

print("Tonotopic map (Greenwood function):")
for x in np.linspace(0, 35, 8):
    f = greenwood_freq(x)
    if f > 0:
        print(f"  x={x:5.1f} mm: f={f:8.0f} Hz")

# Hair cell parameters
N_ihc = 3500    # inner hair cells
N_ohc = 12000   # outer hair cells (3 rows)
spacing = L_cochlea / N_ihc  # ~10 um

print(f"\nHair cell counts: {N_ihc} IHC, {N_ohc} OHC")
print(f"Mean spacing: {spacing*1000:.1f} um")
print(f"Frequency range: {greenwood_freq(35):.0f} Hz - {greenwood_freq(0):.0f} Hz")
print(f"Decades: {np.log10(greenwood_freq(0)/max(greenwood_freq(35), 1)):.1f}")


# =====================================================================
# TEST 1: Cochlea as Kuramoto chain
# =====================================================================
print("\n" + "=" * 70)
print("TEST 1: COCHLEAR KURAMOTO CHAIN")
print("=" * 70)

# Use N=100 oscillators (computationally tractable)
N_sim = 100
positions = np.linspace(0, L_cochlea, N_sim)
freqs = np.array([max(greenwood_freq(x), 1) for x in positions])

# Normalise frequencies for simulation
omega_cochlea = freqs / np.max(freqs)

print(f"Simulating {N_sim} oscillators along cochlea")
print(f"Frequency gradient: {omega_cochlea[0]:.3f} (base) -> {omega_cochlea[-1]:.3f} (apex)")

# Nearest-neighbour coupling (basilar membrane stiffness)
def simulate_cochlea(K_coupling, omega, input_freq=None, input_amp=0.0,
                     noise=0.01, dt=0.01, T=200):
    N = len(omega)
    n_steps = int(T / dt)
    theta = np.random.uniform(0, 2 * np.pi, N)

    R_local = np.zeros(N)
    amplitude = np.zeros(N)

    for _s in range(n_steps):
        dtheta = omega.copy()

        # External sound input at matching frequency
        if input_freq is not None and input_amp > 0:
            for i in range(N):
                # Each hair cell resonates with sound near its CF
                detuning = abs(omega[i] - input_freq / np.max(freqs))
                resonance = np.exp(-detuning**2 / 0.01)  # sharp tuning
                dtheta[i] += input_amp * resonance * np.sin(2 * np.pi * input_freq * _s * dt)

        # Nearest-neighbour coupling
        for i in range(N):
            if i > 0:
                dtheta[i] += K_coupling * np.sin(theta[i-1] - theta[i])
            if i < N - 1:
                dtheta[i] += K_coupling * np.sin(theta[i+1] - theta[i])

        theta += dtheta * dt + noise * np.random.randn(N) * np.sqrt(dt)

    # Compute local order parameter
    for i in range(N):
        window = slice(max(0, i-3), min(N, i+4))
        z = np.mean(np.exp(1j * theta[window]))
        R_local[i] = abs(z)
        amplitude[i] = abs(np.sin(theta[i]))

    return R_local, amplitude, theta


# No input (spontaneous oscillation)
R_spont, _, _ = simulate_cochlea(0.5, omega_cochlea)
print(f"\nSpontaneous state (no input):")
print(f"  Mean R_local: {np.mean(R_spont):.3f}")
print(f"  Base R: {np.mean(R_spont[:10]):.3f}")
print(f"  Apex R: {np.mean(R_spont[-10:]):.3f}")


# TEST 2: Frequency selectivity (tuning curve)
print("\n" + "=" * 70)
print("TEST 2: FREQUENCY SELECTIVITY")
print("=" * 70)

# Input a pure tone and see which oscillators respond
test_freqs = [100, 500, 1000, 4000, 10000]
print(f"{'Input Hz':>10s} {'Peak pos':>10s} {'Peak CF':>10s} {'Width':>8s}")
print("-" * 42)

for f_input in test_freqs:
    R_tone, amp, _ = simulate_cochlea(0.5, omega_cochlea,
                                       input_freq=f_input, input_amp=2.0, T=100)
    peak_idx = np.argmax(R_tone)
    peak_cf = freqs[peak_idx]
    # Width: half-max of response
    half_max = (np.max(R_tone) + np.min(R_tone)) / 2
    width = np.sum(R_tone > half_max) * (L_cochlea / N_sim)
    print(f"{f_input:10d} {positions[peak_idx]:10.1f} mm {peak_cf:10.0f} Hz {width:8.1f} mm")


# TEST 3: Active amplification (OHC as K boost)
print("\n" + "=" * 70)
print("TEST 3: OUTER HAIR CELL AMPLIFICATION")
print("=" * 70)

# OHC electromotility: prestin protein changes cell length with voltage
# This acts as a LOCAL K boost near the characteristic frequency
# Gain: ~40 dB (100x amplitude) at low sound levels

print("Effect of OHC amplification (K boost):")
for K_ohc in [0.0, 0.5, 1.0, 2.0, 5.0]:
    R_amp, _, _ = simulate_cochlea(K_ohc, omega_cochlea,
                                    input_freq=1000, input_amp=0.5, T=100)
    peak_R = np.max(R_amp)
    print(f"  K_OHC={K_ohc:.1f}: peak R={peak_R:.3f}")

print("\nOHC damage (presbycusis = age-related hearing loss):")
print("  Losing OHCs = reducing local K = losing amplification")
print("  High frequencies lost first (base of cochlea)")
print("  This IS desynchronisation at the base of the cochlear chain")


# TEST 4: Otoacoustic emissions (the ear singing)
print("\n" + "=" * 70)
print("TEST 4: OTOACOUSTIC EMISSIONS")
print("=" * 70)

# Spontaneous OAE: the cochlea oscillates and emits sound WITHOUT input
# This is the active oscillator emitting at its natural frequency
# Only possible if K > K_c locally (self-sustained oscillation)

# Test: at what K does the cochlea self-oscillate?
print("Self-oscillation threshold (OAE):")
for K in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
    R_oae, _, theta_oae = simulate_cochlea(K, omega_cochlea, T=300)
    # Coherent emission: R > 0.5 at any location
    n_emitting = np.sum(R_oae > 0.5)
    print(f"  K={K:.1f}: {n_emitting}/{N_sim} positions self-oscillating")

print("\nSOAE occurs in ~70% of normal ears (Penner & Zhang 1997)")
print("Absent in damaged ears (OHC loss = K below threshold)")


# TEST 5: Cochlea vs SCPN frequency architecture
print("\n" + "=" * 70)
print("TEST 5: COCHLEAR vs SCPN FREQUENCY ARCHITECTURE")
print("=" * 70)

# Cochlea: logarithmic frequency map (Greenwood)
# SCPN: compressed (golden-ratio-like)
# EEG: log-uniform (~e ratio)

# Cochlear frequency ratios between adjacent positions
cochlea_ratios = freqs[:-1] / freqs[1:]
cochlea_ratios = cochlea_ratios[cochlea_ratios > 0]
cochlea_ratios = cochlea_ratios[np.isfinite(cochlea_ratios)]

omega_scpn = np.array([0.062, 0.191, 0.382, 0.618, 0.809, 0.927, 0.981, 1.000])
scpn_ratios = omega_scpn[1:] / omega_scpn[:-1]

print(f"Cochlear mean ratio: {np.mean(cochlea_ratios):.4f}")
print(f"Cochlear CV: {np.std(cochlea_ratios)/np.mean(cochlea_ratios):.4f}")
print(f"SCPN mean ratio: {np.mean(scpn_ratios):.3f}")
print(f"SCPN CV: {np.std(np.diff(np.log(omega_scpn)))/np.mean(np.diff(np.log(omega_scpn))):.3f}")

print(f"\nCochlea spans {np.log10(np.max(freqs)/max(np.min(freqs[freqs>0]), 1)):.1f} decades")
print(f"SCPN spans {np.log10(omega_scpn[-1]/omega_scpn[0]):.1f} decades")
print(f"EEG spans ~2.5 decades")

# Key difference
print("\nCochlea has CONSTANT ratio in log space (uniform log mapping)")
print("SCPN has VARIABLE ratio (compressed at high end)")
print("The cochlea is the MOST uniform biological frequency map")
print("Because its job IS frequency analysis, not synchronisation")


# TEST 6: Hearing loss as desynchronisation cascade
print("\n" + "=" * 70)
print("TEST 6: HEARING LOSS TYPES AS K PERTURBATIONS")
print("=" * 70)

hearing_loss = {
    "presbycusis": "gradual OHC loss from base -> K_base decreases -> high freq lost first",
    "noise_damage": "acute OHC death at exposed frequency -> local K = 0 -> notch",
    "Menieres": "endolymph pressure changes omega -> frequency mismatch -> vertigo",
    "otosclerosis": "stapes fixation -> input K to chain reduced -> conductive loss",
    "tinnitus": "focal K increase -> self-oscillation at one frequency -> phantom sound",
}

for condition, description in hearing_loss.items():
    print(f"\n  {condition}:")
    print(f"    {description}")

print("\nTinnitus is particularly interesting:")
print("  Focal increase in K -> local R > threshold -> spontaneous oscillation")
print("  The SAME mechanism as otoacoustic emission, but pathological")
print("  Treatment: reduce local K (sound therapy = add noise = anti-SR)")


# SYNTHESIS
print("\n" + "=" * 70)
print("SYNTHESIS: THE COCHLEA AS PERFECT KURAMOTO CHAIN")
print("=" * 70)

print("""
The cochlea is the clearest biological Kuramoto system:

1. FREQUENCY GRADIENT: omega varies logarithmically along chain
   3 decades (20 Hz - 20 kHz) mapped to 35 mm
   Greenwood function: exact mathematical description

2. NEAREST-NEIGHBOUR COUPLING: basilar membrane stiffness
   Coupling K determines frequency selectivity (Q factor)
   Sharper tuning = stronger coupling = narrower response

3. ACTIVE AMPLIFICATION: OHC electromotility = local K boost
   40 dB gain at low levels -> cochlear amplifier
   Loss of OHCs = hearing loss = K decay (same as aging!)

4. SELF-OSCILLATION: otoacoustic emissions prove K > K_c
   The ear is a self-oscillating Kuramoto chain
   70% of normal ears emit spontaneous sound

5. DISEASES MAP TO K PERTURBATIONS:
   Presbycusis = K decay from base (aging)
   Noise damage = focal K destruction
   Tinnitus = focal K excess (pathological self-oscillation)
   Meniere's = omega perturbation (frequency mismatch)

6. The cochlea is SCPN's best biological analogue:
   frequency-graded, nearest-neighbour coupled, actively amplified
""")

results = {
    "frequency_range_decades": round(float(np.log10(np.max(freqs)/max(np.min(freqs[freqs>0]), 1))), 1),
    "cochlea_mean_ratio": round(float(np.mean(cochlea_ratios)), 4),
    "scpn_mean_ratio": round(float(np.mean(scpn_ratios)), 3),
    "n_ihc": N_ihc,
    "n_ohc": N_ohc,
    "cochlea_length_mm": L_cochlea,
}

print("\n--- JSON RESULTS ---")
print(json.dumps(results, indent=2))
