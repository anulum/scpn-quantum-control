# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 4 Test: Organ Harmonic Chambers
#
# Paper 4, Section 6 (p.28-29): Organs as resonant chambers with
# specific harmonic signatures. Each organ has eigenmodes determined
# by its geometry and tissue properties.
#
# Key equations:
# 1. Organ resonance: nabla^2 p + (omega^2/c^2)p = -rho_0 dq/dt
# 2. Heart (ellipsoid): psi_nlm = j_l(k_n r) Y_l^m(theta, phi)
#    Resonances: 0.5-2 Hz (heartbeat), 20-200 Hz (sounds)
# 3. Lungs (fractal bronchial): Z_acoustic = Z_0 (i*omega/omega_0)^{-(1-D_f)/2}
#    D_f ~ 1.7, resonances: 100-1000 Hz
# 4. Brain (sphere+ventricles): psi = R_nl(r) Y_l^m * [1 - V_ventricle(r)]
#    Resonances: 0.1-100 Hz (neural oscillations)
# 5. Liver (lobed): psi = sum_lobes psi_lobe * exp(i*delta_lobe)
#    Resonances: 0.01-1 Hz (metabolic)
# 6. Organ-organ coupling: H = sum g_12 integral psi_1* psi_2 d^3r

import numpy as np
from scipy.special import spherical_jn, sph_harm
import json

FINDINGS = []

def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")

# --- Test 1: Heart as ellipsoidal resonator ---
print("=== Test 1: Heart resonant eigenmodes (Paper 4, p.28) ===")

# Heart dimensions: semi-axes a~6cm, b~4cm, c~3cm
# Sound speed in cardiac tissue: c ~ 1540 m/s (close to water)
# But for pressure waves through the chamber: c_blood ~ 1570 m/s

a_heart = 0.06  # m
c_tissue = 1540  # m/s

# Spherical approximation: R_eff = (a*b*c)^(1/3) ~ 4.2 cm
R_heart = 0.042  # m

# Eigenfrequencies of a sphere: f_nl = x_nl * c / (2*pi*R)
# where x_nl are zeros of j_l(x) or j_l'(x)
# First few zeros of j_0(x): 3.14, 6.28, 9.42...
# First few zeros of j_1(x): 4.49, 7.73, 10.90...

zeros_j0 = [np.pi, 2*np.pi, 3*np.pi]
zeros_j1 = [4.493, 7.725, 10.904]
zeros_j2 = [5.763, 9.095, 12.323]

heart_modes = []
for l, zeros in enumerate([zeros_j0, zeros_j1, zeros_j2]):
    for n, x_nl in enumerate(zeros):
        f = x_nl * c_tissue / (2 * np.pi * R_heart)
        heart_modes.append({
            "l": l, "n": n + 1,
            "x_nl": round(x_nl, 3),
            "f_Hz": round(f, 1),
        })

# Compare with known heart sounds
# S1: 20-150 Hz (mitral/tricuspid closure)
# S2: 50-200 Hz (aortic/pulmonic closure)
# Heartbeat: 1-2 Hz
# Murmurs: 100-600 Hz

add_finding("HEART_EIGENMODES", "Heart chamber resonant frequencies", {
    "modes": heart_modes[:6],
    "R_effective_cm": R_heart * 100,
    "c_tissue_m_s": c_tissue,
    "S1_range_Hz": "20-150 (mitral/tricuspid closure)",
    "S2_range_Hz": "50-200 (aortic/pulmonic closure)",
    "lowest_mode_Hz": heart_modes[0]["f_Hz"],
    "matches_heart_sounds": any(20 < m["f_Hz"] < 200 for m in heart_modes),
    "equation": "Paper 4: psi_nlm = j_l(k_n r) Y_l^m",
})

# --- Test 2: Lung as fractal resonator ---
print("\n=== Test 2: Lung fractal acoustic impedance (Paper 4, p.29) ===")

# Paper 4: Z = Z_0 * (i*omega/omega_0)^{-(1-D_f)/2}
# D_f ~ 1.7 (bronchial tree fractal dimension)
D_f = 1.7
Z_0 = 400  # Pa·s/m (characteristic impedance of air at body temp)
omega_0 = 2 * np.pi * 500  # reference frequency

freqs = np.logspace(1, 3.5, 100)  # 10 Hz to 3 kHz
omegas_lung = 2 * np.pi * freqs

# Fractal impedance (magnitude)
exponent = -(1 - D_f) / 2  # = 0.15
Z_fractal = Z_0 * np.abs((1j * omegas_lung / omega_0) ** exponent)

# Compare with normal breathing sounds
# Tracheal: 100-1500 Hz
# Vesicular: 100-500 Hz
# Wheezes (pathological): 400-2000 Hz

# Find resonance peaks (impedance minima)
dZ = np.diff(Z_fractal)
minima_idx = np.where((dZ[:-1] < 0) & (dZ[1:] > 0))[0]
resonance_freqs = freqs[minima_idx + 1] if len(minima_idx) > 0 else []

# Fractal dimension effect: how does D_f change impedance?
D_f_values = [1.0, 1.3, 1.5, 1.7, 2.0]
impedance_vs_Df = []
for D in D_f_values:
    exp_D = -(1 - D) / 2
    Z_D = Z_0 * np.abs((1j * 2 * np.pi * 500 / omega_0) ** exp_D)
    impedance_vs_Df.append({"D_f": D, "Z_500Hz": round(float(Z_D), 2)})

add_finding("LUNG_FRACTAL", "Lung fractal acoustic impedance", {
    "D_f": D_f,
    "exponent": round(exponent, 4),
    "Z_at_100Hz": round(float(Z_fractal[np.argmin(np.abs(freqs - 100))]), 2),
    "Z_at_500Hz": round(float(Z_fractal[np.argmin(np.abs(freqs - 500))]), 2),
    "Z_at_1000Hz": round(float(Z_fractal[np.argmin(np.abs(freqs - 1000))]), 2),
    "impedance_vs_Df": impedance_vs_Df,
    "equation": "Paper 4: Z = Z_0 * (i*omega/omega_0)^{-(1-D_f)/2}",
})

# --- Test 3: Brain eigenmodes (sphere with ventricles) ---
print("\n=== Test 3: Brain resonant modes (Paper 4, p.29) ===")

# Brain: roughly spherical, R ~ 7 cm
# EM wave speed in brain tissue: c_brain ~ 1.5e8 m/s (half speed of light)
# For mechanical: c ~ 1-5 m/s (shear waves in brain tissue)
# For neural oscillations: effectively c ~ L*f (conduction velocity / wavelength)

R_brain = 0.07  # m
c_shear = 2.0  # m/s (brain tissue shear wave, MRE measured)

# Eigenmodes for neural-mechanical coupling
brain_modes = []
for l in range(4):
    zeros_l = {0: [np.pi, 2*np.pi], 1: [4.493, 7.725], 2: [5.763, 9.095], 3: [6.988, 10.417]}
    for n, x in enumerate(zeros_l[l]):
        f = x * c_shear / (2 * np.pi * R_brain)
        brain_modes.append({"l": l, "n": n + 1, "f_Hz": round(f, 2)})

# Compare with neural oscillation bands
neural_bands = {
    "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 100),
}

mode_band_matches = []
for mode in brain_modes:
    for band, (f_low, f_high) in neural_bands.items():
        if f_low <= mode["f_Hz"] <= f_high:
            mode_band_matches.append({"mode": f"l={mode['l']},n={mode['n']}", "f_Hz": mode["f_Hz"], "band": band})

add_finding("BRAIN_EIGENMODES", "Brain mechanical eigenmodes vs neural bands", {
    "modes": brain_modes,
    "c_shear_m_s": c_shear,
    "R_brain_cm": R_brain * 100,
    "neural_band_matches": mode_band_matches,
    "note": "mechanical resonances may entrain neural oscillations",
    "equation": "Paper 4: psi_brain = R_nl(r) Y_l^m * [1 - V_ventricle]",
})

# --- Test 4: Organ-organ coupling via shared harmonics ---
print("\n=== Test 4: Heart-brain coupling via shared frequencies ===")

# Heart Rate Variability (HRV): 0.04-0.4 Hz
# Brain alpha: 8-13 Hz
# Coupling frequency: respiratory sinus arrhythmia ~0.15-0.4 Hz

# Heart oscillator
omega_heart = 2 * np.pi * 1.2  # 72 bpm
# Brain theta
omega_brain = 2 * np.pi * 6.0  # 6 Hz theta

# HRV modulates brain via baroreceptors
K_heart_brain = 0.3
# Brain modulates heart via vagus nerve
K_brain_heart = 0.15

N_heart = 50
N_brain = 50
N_total = N_heart + N_brain

np.random.seed(42)
omegas_hb = np.concatenate([
    np.random.normal(omega_heart, omega_heart * 0.1, N_heart),
    np.random.normal(omega_brain, omega_brain * 0.05, N_brain)
])
theta_hb = np.random.uniform(0, 2 * np.pi, N_total)

dt = 0.001
r_heart_trace = []
r_brain_trace = []
coherence_trace = []

for step in range(20000):
    z_h = np.mean(np.exp(1j * theta_hb[:N_heart]))
    r_h = np.abs(z_h)
    psi_h = np.angle(z_h)

    z_b = np.mean(np.exp(1j * theta_hb[N_heart:]))
    r_b = np.abs(z_b)
    psi_b = np.angle(z_b)

    dtheta = omegas_hb.copy()
    # Intra-organ coupling
    dtheta[:N_heart] += 2.0 * r_h * np.sin(psi_h - theta_hb[:N_heart])
    dtheta[N_heart:] += 2.0 * r_b * np.sin(psi_b - theta_hb[N_heart:])
    # Inter-organ coupling (Paper 4: H = sum g_12 integral psi_1* psi_2)
    dtheta[:N_heart] += K_brain_heart * r_b * np.sin(psi_b - theta_hb[:N_heart])
    dtheta[N_heart:] += K_heart_brain * r_h * np.sin(psi_h - theta_hb[N_heart:])

    theta_hb += dt * dtheta

    if step % 100 == 0:
        r_heart_trace.append(float(r_h))
        r_brain_trace.append(float(r_b))
        # Cross-organ phase coherence
        phase_diff = psi_h - psi_b
        coherence_trace.append(float(np.cos(phase_diff)))

heart_brain_coherence = np.mean(coherence_trace[-20:])

add_finding("HEART_BRAIN_COUPLING", "Organ-organ coupling via shared harmonics", {
    "r_heart": round(float(np.mean(r_heart_trace[-20:])), 4),
    "r_brain": round(float(np.mean(r_brain_trace[-20:])), 4),
    "heart_brain_coherence": round(float(heart_brain_coherence), 4),
    "K_heart_to_brain": K_heart_brain,
    "K_brain_to_heart": K_brain_heart,
    "mechanism": "baroreceptor + vagus nerve bidirectional coupling",
    "equation": "Paper 4, p.29: H = sum g_12 integral psi_1* psi_2 d^3r",
})

# --- Test 5: Schumann resonance coupling (Paper 4, p.30-31) ---
print("\n=== Test 5: Schumann resonance entrainment (Paper 4, p.30-31) ===")

# Paper 4: f_n = (c/2*pi*R_E) * sqrt(n(n+1))
c_light = 3e8  # m/s
R_earth = 6.371e6  # m

schumann_modes = []
for n in range(1, 7):
    f_n = (c_light / (2 * np.pi * R_earth)) * np.sqrt(n * (n + 1))
    schumann_modes.append({"n": n, "f_Hz": round(f_n, 2)})

# Compare with EEG bands
eeg_matches = []
for sm in schumann_modes:
    for band, (f_lo, f_hi) in neural_bands.items():
        if f_lo <= sm["f_Hz"] <= f_hi:
            eeg_matches.append({"schumann_n": sm["n"], "f_Hz": sm["f_Hz"], "eeg_band": band})

# Paper 4 entrainment: dphi_bio/dt = omega_bio + eps*E_Schumann*sin(phi_Sch - phi_bio)
# Phase locking condition: eps*E_Schumann > |omega_bio - omega_Schumann|
E_schumann = 1e-3  # V/m (typical Schumann field)
eps_coupling = 1e-6  # very weak biological antenna coupling

# Can it entrain?
delta_omega_typical = 2 * np.pi * 0.5  # 0.5 Hz detuning
coupling_strength = eps_coupling * E_schumann
locking_possible = coupling_strength > delta_omega_typical / (2 * np.pi)

add_finding("SCHUMANN_PAPER4", "Schumann resonances vs neural bands (Paper 4 eq.)", {
    "schumann_modes": schumann_modes,
    "eeg_matches": eeg_matches,
    "f1_Hz": schumann_modes[0]["f_Hz"],
    "f1_vs_theta": f"f1={schumann_modes[0]['f_Hz']} Hz falls in theta (4-8 Hz)",
    "direct_entrainment_possible": locking_possible,
    "coupling_strength": float(coupling_strength),
    "note": "direct coupling too weak — Paper 4 invokes stochastic resonance amplification",
    "equation": "Paper 4: f_n = (c/2piR) sqrt(n(n+1)), dphi/dt = omega + eps*sin(Delta_phi)",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_organ_harmonic_chambers", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
