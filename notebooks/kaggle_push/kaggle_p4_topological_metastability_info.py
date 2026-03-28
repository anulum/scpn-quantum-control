# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 4 Test: Topological Phases + Metastability + Phi
#
# Paper 4 specifies:
#
# A. Topological phase transitions (p.71-72):
#    Winding number W = (1/2pi) oint nabla phi . dl = n (integer)
#    Topological Synchronization Index TSI (Chern number)
#    Phase diagram: TSI=1 (sync) vs TSI=0 (chimera/async)
#    Edge states: psi_edge ~ exp(-x/xi_loc) * exp(ik_edge*x)
#
# B. Metastability Index (p.53-55):
#    MI = std_t[R(t)], R(t) = (1/N)|sum exp(i theta_j)|
#    High MI (0.1-0.2) = flexible switching
#    Low MI (~0.05) = rigid, less adaptable
#    MI ~ 0.3 = unstable desynchronisation
#
# C. Integrated Information Phi (p.73-74):
#    Phi peaks at sigma=1 (criticality)
#    Phi(sigma) ~ Phi_max * exp(-(sigma-1)^2 / (2*sigma_Phi^2))
#    sigma_Phi ~ 0.1
#    Transfer entropy: TE_max at Delta_phi = pi/2 (excitatory)

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

np.random.seed(42)

# === PART A: Topological Phases ===
print("=== Part A: Topological winding numbers in oscillator lattice ===")

# --- A1: Winding number detection ---
# 2D lattice of oscillators — detect phase vortices
Nx, Ny = 30, 30
N_topo = Nx * Ny
omega_0 = 2 * np.pi * 10
K_topo = 1.5

omegas_t = np.random.normal(omega_0, omega_0 * 0.05, N_topo)
# Initial condition: embed a vortex (winding number = 1)
theta_t = np.zeros(N_topo)
for iy in range(Ny):
    for ix in range(Nx):
        idx = iy * Nx + ix
        theta_t[idx] = np.arctan2(iy - Ny/2, ix - Nx/2)  # vortex at centre

dt = 0.001
for step in range(10000):
    dtheta = omegas_t.copy()
    for iy in range(Ny):
        for ix in range(Nx):
            idx = iy * Nx + ix
            neighbours = []
            if ix > 0: neighbours.append(iy * Nx + (ix - 1))
            if ix < Nx - 1: neighbours.append(iy * Nx + (ix + 1))
            if iy > 0: neighbours.append((iy - 1) * Nx + ix)
            if iy < Ny - 1: neighbours.append((iy + 1) * Nx + ix)
            for j in neighbours:
                dtheta[idx] += K_topo * np.sin(theta_t[j] - theta_t[idx]) / len(neighbours)
    theta_t += dt * dtheta

# Compute winding number around loops
def winding_number(phases, cx, cy, radius, Nx):
    ring = []
    for angle_step in range(16):
        a = 2 * np.pi * angle_step / 16
        rx = int(cx + radius * np.cos(a))
        ry = int(cy + radius * np.sin(a))
        rx = np.clip(rx, 0, Nx - 1)
        ry = np.clip(ry, 0, Ny - 1)
        ring.append(phases[ry * Nx + rx])
    dphis = np.diff(ring + [ring[0]])
    dphis = (dphis + np.pi) % (2 * np.pi) - np.pi
    return np.sum(dphis) / (2 * np.pi)

W_centre = winding_number(theta_t, Nx//2, Ny//2, 5, Nx)
W_corner = winding_number(theta_t, 5, 5, 3, Nx)

# Global order parameter
r_topo, _ = order_param(theta_t)

add_finding("TOPOLOGICAL_WINDING", "Winding number detection in 2D oscillator lattice", {
    "W_at_centre": round(float(W_centre), 2),
    "W_at_corner": round(float(W_corner), 2),
    "vortex_survived": abs(W_centre) > 0.5,
    "global_r": round(float(r_topo), 4),
    "grid": f"{Nx}x{Ny}",
    "K": K_topo,
    "equation": "Paper 4, p.71: W = (1/2pi) oint nabla phi . dl = n",
})

# --- A2: Topological phase diagram ---
print("\n--- A2: Phase diagram (K vs disorder) ---")

K_sweep = np.linspace(0.1, 3.0, 10)
disorder_sweep = [0.01, 0.05, 0.1, 0.2, 0.5]
phase_diagram = []

for sigma_d in disorder_sweep:
    for K_pd in K_sweep:
        N_pd = 100
        omegas_pd = np.random.normal(omega_0, omega_0 * sigma_d, N_pd)
        theta_pd = np.random.uniform(0, 2 * np.pi, N_pd)
        dt = 0.001
        for step in range(5000):
            z = np.mean(np.exp(1j * theta_pd))
            r = np.abs(z)
            psi = np.angle(z)
            dtheta = omegas_pd + K_pd * r * np.sin(psi - theta_pd)
            theta_pd += dt * dtheta
        r_pd, _ = order_param(theta_pd)
        phase_diagram.append({
            "K": round(float(K_pd), 2),
            "disorder": sigma_d,
            "r": round(float(r_pd), 4),
        })

add_finding("TOPO_PHASE_DIAGRAM", "K vs disorder phase diagram", {
    "results_sample": phase_diagram[::5],
    "paper4": "TSI=1 (sync) at high K, low disorder; TSI=0 otherwise",
})

# === PART B: Metastability Index ===
print("\n=== Part B: Metastability Index (Paper 4, p.53-55) ===")

# MI = std_t[R(t)] over window 100-500 ms
N_mi = 200
K_mi_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
mi_results = []

for K_mi in K_mi_values:
    omegas_mi = np.random.normal(omega_0, omega_0 * 0.15, N_mi)
    theta_mi = np.random.uniform(0, 2 * np.pi, N_mi)
    dt = 0.001
    r_window = []

    for step in range(30000):
        z = np.mean(np.exp(1j * theta_mi))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_mi + K_mi * r * np.sin(psi - theta_mi) + 0.3 * np.random.randn(N_mi)
        theta_mi += dt * dtheta

        if step > 20000 and step % 10 == 0:
            r_window.append(float(r))

    r_arr = np.array(r_window)
    MI = float(np.std(r_arr))
    r_mean = float(np.mean(r_arr))

    mi_results.append({
        "K": round(K_mi, 2),
        "r_mean": round(r_mean, 4),
        "MI": round(MI, 4),
        "regime": "flexible" if 0.08 < MI < 0.25 else "rigid" if MI < 0.08 else "unstable",
    })

add_finding("METASTABILITY_INDEX", "MI quantifies switching flexibility", {
    "results": mi_results,
    "paper4_optimal_MI": "0.1-0.2 (flexible switching)",
    "paper4_rigid_MI": "~0.05 (less adaptable)",
    "paper4_unstable_MI": "~0.3 (desynchronised)",
    "equation": "Paper 4, p.53: MI = std_t[R(t)], t ~ 100-500 ms",
})

# === PART C: Integrated Information Phi ===
print("\n=== Part C: Phi peaks at criticality (Paper 4, p.73) ===")

# Paper 4: Phi(sigma) ~ Phi_max * exp(-(sigma-1)^2 / (2*sigma_Phi^2))
# sigma_Phi ~ 0.1
# Approximate Phi via mutual information between two halves

N_phi = 100
sigma_br_values = np.linspace(0.5, 1.5, 20)
phi_results = []

for sigma_br in sigma_br_values:
    # Simulate with branching-like coupling
    K_phi = 2.0 * sigma_br
    omegas_phi = np.random.normal(omega_0, omega_0 * 0.1, N_phi)
    theta_phi = np.random.uniform(0, 2 * np.pi, N_phi)
    dt = 0.001

    # Collect time series
    r_half1_trace = []
    r_half2_trace = []

    for step in range(20000):
        z = np.mean(np.exp(1j * theta_phi))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_phi + K_phi * r * np.sin(psi - theta_phi) + 0.2 * np.random.randn(N_phi)
        theta_phi += dt * dtheta

        if step > 15000 and step % 10 == 0:
            r1, _ = order_param(theta_phi[:N_phi//2])
            r2, _ = order_param(theta_phi[N_phi//2:])
            r_half1_trace.append(float(r1))
            r_half2_trace.append(float(r2))

    # Proxy for Phi: correlation between halves (integration)
    # × variance within halves (differentiation)
    r1_arr = np.array(r_half1_trace)
    r2_arr = np.array(r_half2_trace)
    correlation = np.corrcoef(r1_arr, r2_arr)[0, 1] if len(r1_arr) > 2 else 0
    variance = np.std(r1_arr) * np.std(r2_arr)
    phi_proxy = abs(correlation) * variance * 100  # arbitrary scale

    phi_results.append({
        "sigma": round(float(sigma_br), 3),
        "phi_proxy": round(float(phi_proxy), 4),
        "integration": round(float(abs(correlation)), 4),
        "differentiation": round(float(variance), 4),
    })

# Find peak
phi_vals = [x["phi_proxy"] for x in phi_results]
sigma_peak = sigma_br_values[np.argmax(phi_vals)]

add_finding("PHI_CRITICALITY", "Integrated information peaks at criticality", {
    "results_sample": phi_results[::2],
    "sigma_at_peak_Phi": round(float(sigma_peak), 3),
    "paper4_prediction": "Phi peaks at sigma=1.0",
    "sigma_Phi_width": 0.1,
    "equation": "Paper 4, p.73: Phi(sigma) ~ Phi_max * exp(-(sigma-1)^2 / 0.02)",
    "consciousness_link": "Phi measures integration — peaks at criticality = optimal awareness",
})

# === PART D: Multi-organ coupling matrix (Paper 4, p.60) ===
print("\n=== Part D: Multi-organ coupling matrix ===")

# Paper 4 specifies:
# C_organs = [[1.0, 0.3, 0.2, 0.1],  # Brain
#             [0.3, 1.0, 0.4, 0.2],  # Heart
#             [0.2, 0.4, 1.0, 0.1],  # Lungs
#             [0.1, 0.2, 0.1, 1.0]]  # GI tract

C_organs = np.array([
    [1.0, 0.3, 0.2, 0.1],
    [0.3, 1.0, 0.4, 0.2],
    [0.2, 0.4, 1.0, 0.1],
    [0.1, 0.2, 0.1, 1.0],
])
organ_names = ["Brain", "Heart", "Lungs", "GI"]
organ_freqs = [10.0, 1.2, 0.25, 0.05]  # Hz

N_per_organ = 30
N_organs = 4
N_total = N_per_organ * N_organs

omegas_org = np.concatenate([
    np.random.normal(2 * np.pi * f, 2 * np.pi * f * 0.1, N_per_organ)
    for f in organ_freqs
])
theta_org = np.random.uniform(0, 2 * np.pi, N_total)

dt = 0.001
for step in range(20000):
    for i in range(N_organs):
        sl_i = slice(i * N_per_organ, (i + 1) * N_per_organ)
        z_i = np.mean(np.exp(1j * theta_org[sl_i]))
        r_i = np.abs(z_i)
        psi_i = np.angle(z_i)

        # Intra-organ coupling
        dtheta_i = omegas_org[sl_i] + 3.0 * r_i * np.sin(psi_i - theta_org[sl_i])

        # Inter-organ coupling (Paper 4 matrix)
        for j in range(N_organs):
            if i != j:
                sl_j = slice(j * N_per_organ, (j + 1) * N_per_organ)
                z_j = np.mean(np.exp(1j * theta_org[sl_j]))
                dtheta_i += C_organs[i, j] * np.abs(z_j) * np.sin(np.angle(z_j) - theta_org[sl_i])

        theta_org[sl_i] += dt * dtheta_i

# Measure inter-organ coherence
coherence_matrix = np.zeros((N_organs, N_organs))
for i in range(N_organs):
    for j in range(N_organs):
        sl_i = slice(i * N_per_organ, (i + 1) * N_per_organ)
        sl_j = slice(j * N_per_organ, (j + 1) * N_per_organ)
        z_i = np.mean(np.exp(1j * theta_org[sl_i]))
        z_j = np.mean(np.exp(1j * theta_org[sl_j]))
        coherence_matrix[i, j] = round(float(np.abs(np.conj(z_i) * z_j)), 4)

organ_sync = {}
for i in range(N_organs):
    sl = slice(i * N_per_organ, (i + 1) * N_per_organ)
    r, _ = order_param(theta_org[sl])
    organ_sync[organ_names[i]] = round(float(r), 4)

add_finding("ORGAN_COUPLING_MATRIX", "Multi-organ coupling from Paper 4", {
    "C_paper4": C_organs.tolist(),
    "organ_sync": organ_sync,
    "coherence_measured": coherence_matrix.tolist(),
    "heart_brain_coherence": coherence_matrix[0, 1],
    "heart_lung_coherence": coherence_matrix[1, 2],
    "equation": "Paper 4, p.60: C_ij coupling matrix, eps_HB ~ 0.01-0.05",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_topological_metastability_info", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
