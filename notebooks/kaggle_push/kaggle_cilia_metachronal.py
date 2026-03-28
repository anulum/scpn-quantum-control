# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cilia Metachronal Waves as Kuramoto
#
# Cilia are ~10 um hair-like structures that beat at 10-40 Hz.
# Neighbouring cilia synchronise through hydrodynamic coupling
# to form metachronal waves (travelling phase waves).
#
# Biology:
# - Airway cilia: 200 cilia/cell, 10-15 Hz, mucociliary clearance
# - Ependymal cilia: CSF flow in brain ventricles
# - Oviduct cilia: egg transport
# - Paramecium: ~5000 cilia, metachronal wave = locomotion
#
# Physics: Golestanian (2011) showed hydrodynamic coupling between
# cilia is EXACTLY Kuramoto with distance-dependent K:
#   K_ij ~ 1/(eta * d_ij)  (Stokes flow, eta = viscosity)
#
# Disease: Primary Ciliary Dyskinesia (PCD) = genetic loss of
# dynein → cilia still beat but can't synchronise. Desync → no
# mucus clearance → chronic infections. Kartagener syndrome: 50%
# have situs inversus (random L-R due to loss of nodal cilia sync).

import numpy as np
import json

FINDINGS = []

def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")

def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)

# --- Test 1: 1D ciliary array with hydrodynamic coupling ---
print("=== Test 1: Metachronal wave in 1D ciliary array ===")

N = 200  # cilia in a line
spacing_um = 0.3  # 300 nm spacing (dense packing)
eta = 1e-3  # Pa·s (water viscosity)
a = 5e-6  # cilium length ~5 um

np.random.seed(42)
omega_0 = 2 * np.pi * 12  # 12 Hz
sigma_omega = omega_0 * 0.05  # 5% natural variation
omegas = np.random.normal(omega_0, sigma_omega, N)

# Hydrodynamic coupling: K_ij = K0 / d_ij (Stokeslet)
positions = np.arange(N) * spacing_um  # um
distances = np.abs(positions[:, None] - positions[None, :])
distances[distances == 0] = 1e-10
K0 = 2.0  # effective coupling strength
K_matrix = K0 / (distances / spacing_um)  # normalised
np.fill_diagonal(K_matrix, 0)
# Truncate long-range (hydrodynamic screening)
K_matrix[distances > 5 * spacing_um] = 0

theta0 = np.random.uniform(0, 2 * np.pi, N)

dt = 1e-4  # seconds (need small dt for 12 Hz)
T = 2.0  # seconds
steps = int(T / dt)

theta = theta0.copy()
r_trace = []
# Track phase wave propagation
wave_snapshots = []

for step in range(steps):
    dtheta = np.copy(omegas)
    for i in range(N):
        neighbours = np.where(K_matrix[i] > 0)[0]
        dtheta[i] += np.sum(K_matrix[i, neighbours] * np.sin(theta[neighbours] - theta[i]))
    theta += dt * dtheta

    if step % 1000 == 0:
        r, _ = order_param(theta)
        r_trace.append(r)
        wave_snapshots.append(theta.copy() % (2 * np.pi))

# Check for metachronal wave: phase should increase linearly with position
final_phases = theta % (2 * np.pi)
# Unwrap and fit linear gradient
phase_unwrapped = np.unwrap(final_phases)
gradient = np.polyfit(positions, phase_unwrapped, 1)
wavelength_um = 2 * np.pi / abs(gradient[0]) if abs(gradient[0]) > 0.01 else float('inf')
wave_speed_um_s = omega_0 * wavelength_um / (2 * np.pi)

add_finding("METACHRONAL_WAVE", "Metachronal wave formation in ciliary array", {
    "global_r": round(float(r_trace[-1]), 4),
    "phase_gradient_rad_per_um": round(float(gradient[0]), 4),
    "wavelength_um": round(float(wavelength_um), 2),
    "wave_speed_um_s": round(float(wave_speed_um_s), 1),
    "N_cilia": N,
    "beat_frequency_Hz": 12,
    "pattern": "metachronal" if abs(gradient[0]) > 0.1 else "synchronous",
})

# --- Test 2: PCD — reduced coupling (dynein loss) ---
print("\n=== Test 2: Primary Ciliary Dyskinesia — coupling loss ===")

K_fractions = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
pcd_results = []

for frac in K_fractions:
    K_pcd = K_matrix * frac
    theta_pcd = theta0.copy()
    for step in range(int(1.0 / dt)):  # 1 second
        dtheta = np.copy(omegas)
        for i in range(N):
            nb = np.where(K_pcd[i] > 0)[0]
            if len(nb) > 0:
                dtheta[i] += np.sum(K_pcd[i, nb] * np.sin(theta_pcd[nb] - theta_pcd[i]))
        theta_pcd += dt * dtheta

    r_pcd, _ = order_param(theta_pcd)
    # Check wave coherence
    phase_uw = np.unwrap(theta_pcd % (2 * np.pi))
    grad = np.polyfit(positions, phase_uw, 1)
    residuals = phase_uw - np.polyval(grad, positions)
    phase_noise = np.std(residuals)

    pcd_results.append({
        "K_fraction": frac,
        "r": round(float(r_pcd), 4),
        "phase_noise_rad": round(float(phase_noise), 4),
        "clearance": "normal" if phase_noise < 1.0 else "impaired" if phase_noise < 2.0 else "absent",
    })

add_finding("PCD_DESYNC", "PCD: progressive coupling loss → clearance failure", {
    "results": pcd_results,
    "clinical": "PCD patients have chronic sinusitis, bronchiectasis from failed clearance",
})

# --- Test 3: Viscosity dependence (mucus vs water) ---
print("\n=== Test 3: Mucus viscosity changes coupling ===")

# In CF (cystic fibrosis), mucus viscosity increases 10-100x
# This changes K_ij because hydrodynamic coupling scales as 1/eta
viscosity_ratios = [1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
viscosity_results = []

for eta_ratio in viscosity_ratios:
    K_visc = K_matrix / eta_ratio  # coupling decreases with viscosity
    theta_v = theta0.copy()
    for step in range(int(1.0 / dt)):
        dtheta = np.copy(omegas)
        for i in range(N):
            nb = np.where(K_visc[i] > 0)[0]
            if len(nb) > 0:
                dtheta[i] += np.sum(K_visc[i, nb] * np.sin(theta_v[nb] - theta_v[i]))
        theta_v += dt * dtheta

    r_v, _ = order_param(theta_v)
    viscosity_results.append({
        "eta_ratio": eta_ratio,
        "r": round(float(r_v), 4),
    })

add_finding("CILIA_VISCOSITY", "Mucus viscosity modulates ciliary sync", {
    "results": viscosity_results,
    "CF_prediction": "CF mucus (100x viscosity) → K/100 → complete desync",
    "therapeutic": "mucolytics reduce viscosity → restore coupling → restore clearance",
})

# --- Test 4: Situs inversus from nodal cilia desync ---
print("\n=== Test 4: Nodal cilia — left-right symmetry breaking ===")

# Nodal cilia create leftward flow by spinning. Sync is needed
# for coherent flow. Without sync: random L-R determination (50/50)

N_nodal = 30  # nodal pit has ~30-300 cilia
omega_nodal = 2 * np.pi * 10  # 10 Hz rotation
K_nodal_values = np.linspace(0, 5, 30)

# Flow is proportional to r (coherent rotation → directional flow)
flow_vs_K = []
for K_n in K_nodal_values:
    theta_n = np.random.uniform(0, 2 * np.pi, N_nodal)
    omegas_n = np.random.normal(omega_nodal, omega_nodal * 0.1, N_nodal)
    dt_n = 1e-4
    for step in range(int(0.5 / dt_n)):
        z = np.mean(np.exp(1j * theta_n))
        r_inst = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_n + K_n * r_inst * np.sin(psi - theta_n)
        theta_n += dt_n * dtheta

    r_n, _ = order_param(theta_n)
    flow_vs_K.append(float(r_n))

# Flow threshold for symmetry breaking
flow_threshold = 0.5
K_c_nodal = None
for i, f in enumerate(flow_vs_K):
    if f > flow_threshold:
        K_c_nodal = K_nodal_values[i]
        break

add_finding("SITUS_INVERSUS", "Nodal cilia sync determines body asymmetry", {
    "K_c_for_flow": round(float(K_c_nodal), 3) if K_c_nodal else "not reached",
    "flow_at_max_K": round(flow_vs_K[-1], 4),
    "N_cilia": N_nodal,
    "PCD_prediction": "K=0 → r≈0 → no flow → 50% situs inversus",
    "clinical_match": "Kartagener syndrome: ~50% situs inversus in PCD patients",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "cilia_metachronal", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
