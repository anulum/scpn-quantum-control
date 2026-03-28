# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Cardiac SA Node as Kuramoto Network
#
# The sinoatrial (SA) node contains ~10,000 pacemaker cells.
# Each cell is an autonomous oscillator (HCN channels → funny current).
# Gap junctions (connexin-43) provide electrical coupling K_ij.
#
# Known biology:
# - Natural freq: 60-100 bpm (1.0-1.67 Hz), spread ~0.2 Hz
# - Gap junction conductance: 1-10 nS per junction
# - Each cell connects to ~6 neighbours (hexagonal packing)
# - Coupling converts 10,000 noisy oscillators → one clean rhythm
#
# Diseases as Kuramoto failures:
# - Sick sinus syndrome: K_ij degrades (fibrosis blocks gap junctions)
# - Atrial fibrillation: spatial chaos from re-entrant waves
# - Heart block: K between SA and AV nodes drops below threshold
# - Defibrillation: electrical reset forces all phases to theta=0

import numpy as np
from scipy.integrate import solve_ivp
import json

FINDINGS = []

def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")

# --- Test 1: SA node as Kuramoto network ---
N = 500  # reduced from 10,000 for compute
omega_0 = 2 * np.pi * 1.2  # 72 bpm = 1.2 Hz
sigma_omega = 2 * np.pi * 0.15  # 0.15 Hz natural spread

np.random.seed(42)
omegas = np.random.normal(omega_0, sigma_omega, N)

# Hexagonal nearest-neighbour coupling (sparse, not all-to-all)
# Each cell connects to ~6 neighbours
from scipy.spatial import Delaunay

positions = np.random.rand(N, 2) * np.sqrt(N)
tri = Delaunay(positions)

# Build adjacency from Delaunay triangulation
adj = np.zeros((N, N), dtype=bool)
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            adj[simplex[i], simplex[j]] = True
            adj[simplex[j], simplex[i]] = True

n_neighbours = np.sum(adj, axis=1)
mean_neighbours = np.mean(n_neighbours)

print(f"=== Test 1: SA node Kuramoto ({N} cells, mean {mean_neighbours:.1f} neighbours) ===")

# Gap junction coupling strength
g_gap = 5.0  # nS (typical)
K_gap = g_gap * 0.1  # conversion to rad/s coupling (phenomenological)

def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)

# Sparse Kuramoto
theta0 = np.random.uniform(0, 2 * np.pi, N)

def sa_node_rhs(t, theta, K):
    dtheta = np.copy(omegas)
    for i in range(N):
        neighbours = np.where(adj[i])[0]
        if len(neighbours) > 0:
            dtheta[i] += K * np.mean(np.sin(theta[neighbours] - theta[i]))
    return dtheta

# Sweep K to find K_c for sparse network
K_values = np.linspace(0, 3.0, 25)
r_vs_K = []

for K in K_values:
    theta_run = theta0.copy()
    dt = 0.005
    for _ in range(4000):  # 20 seconds
        k1 = np.array(sa_node_rhs(0, theta_run, K))
        theta_run += dt * k1

    r, _ = order_param(theta_run)
    r_vs_K.append(r)

r_vs_K = np.array(r_vs_K)
idx_sync = np.where(r_vs_K > 0.5)[0]
K_c_sa = K_values[idx_sync[0]] if len(idx_sync) > 0 else float('nan')

add_finding("SA_NODE_KC", "Critical coupling for SA node sync", {
    "K_c_sparse": round(float(K_c_sa), 4),
    "N_cells": N,
    "mean_neighbours": round(float(mean_neighbours), 1),
    "natural_freq_Hz": 1.2,
    "freq_spread_Hz": 0.15,
    "r_at_max_K": round(float(r_vs_K[-1]), 4),
})

# --- Test 2: Sick sinus syndrome (gap junction degradation) ---
print("\n=== Test 2: Sick Sinus Syndrome — progressive K decay ===")

K_healthy = 2.5  # well above K_c
degradation_fractions = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
rhythm_quality = []

for frac in degradation_fractions:
    K_sick = K_healthy * frac
    theta_run = theta0.copy()
    dt = 0.005
    r_trace = []
    for step in range(6000):  # 30 seconds
        k1 = np.array(sa_node_rhs(0, theta_run, K_sick))
        theta_run += dt * k1
        if step % 20 == 0:
            r, _ = order_param(theta_run)
            r_trace.append(r)

    r_mean = np.mean(r_trace[-50:])
    r_std = np.std(r_trace[-50:])
    rhythm_quality.append({
        "K_fraction": frac,
        "K_value": round(K_sick, 3),
        "r_mean": round(float(r_mean), 4),
        "r_variability": round(float(r_std), 4),
        "clinical": "normal" if r_mean > 0.8 else "bradycardia" if r_mean > 0.4 else "sick_sinus",
    })

add_finding("SICK_SINUS", "Gap junction degradation → rhythm collapse", {
    "results": rhythm_quality,
    "K_healthy": K_healthy,
    "prediction": "gradual r decay then sharp collapse at ~40% gap junction loss",
})

# --- Test 3: Atrial fibrillation as spatial chaos ---
print("\n=== Test 3: Atrial fibrillation — re-entrant spiral waves ===")

# Create a heterogeneous region (scar tissue blocks conduction)
N_af = 200
positions_af = np.random.rand(N_af, 2) * 15
tri_af = Delaunay(positions_af)
adj_af = np.zeros((N_af, N_af), dtype=bool)
for simplex in tri_af.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            adj_af[simplex[i], simplex[j]] = True
            adj_af[simplex[j], simplex[i]] = True

# Add scar: disconnect nodes in a region
scar_centre = np.array([7.5, 7.5])
scar_radius = 3.0
in_scar = np.sqrt(np.sum((positions_af - scar_centre) ** 2, axis=1)) < scar_radius
# Reduce coupling TO/FROM scar nodes by 90%
adj_scar = adj_af.copy()
for i in np.where(in_scar)[0]:
    # Keep only 10% of connections
    neighbours = np.where(adj_scar[i])[0]
    mask = np.random.rand(len(neighbours)) > 0.1
    for j in neighbours[mask]:
        adj_scar[i, j] = False
        adj_scar[j, i] = False

omegas_af = np.random.normal(omega_0, sigma_omega, N_af)
theta_af = np.random.uniform(0, 2 * np.pi, N_af)

K_af = 2.0
dt = 0.005
r_healthy_trace = []
r_scar_trace = []

# Run healthy
theta_h = theta_af.copy()
for step in range(8000):
    dtheta = np.copy(omegas_af)
    for i in range(N_af):
        nb = np.where(adj_af[i])[0]
        if len(nb) > 0:
            dtheta[i] += K_af * np.mean(np.sin(theta_h[nb] - theta_h[i]))
    theta_h += dt * dtheta
    if step % 40 == 0:
        r, _ = order_param(theta_h)
        r_healthy_trace.append(r)

# Run with scar
theta_s = theta_af.copy()
for step in range(8000):
    dtheta = np.copy(omegas_af)
    for i in range(N_af):
        nb = np.where(adj_scar[i])[0]
        if len(nb) > 0:
            dtheta[i] += K_af * np.mean(np.sin(theta_s[nb] - theta_s[i]))
    theta_s += dt * dtheta
    if step % 40 == 0:
        r, _ = order_param(theta_s)
        r_scar_trace.append(r)

add_finding("AFIB_SCAR", "Scar tissue creates AF substrate", {
    "r_healthy_final": round(float(np.mean(r_healthy_trace[-20:])), 4),
    "r_scarred_final": round(float(np.mean(r_scar_trace[-20:])), 4),
    "r_reduction": round(float(1 - np.mean(r_scar_trace[-20:]) / max(np.mean(r_healthy_trace[-20:]), 0.01)), 3),
    "scar_fraction": round(float(np.sum(in_scar) / N_af), 3),
    "K_coupling": K_af,
    "N": N_af,
})

# --- Test 4: Defibrillation as phase reset ---
print("\n=== Test 4: Defibrillation = forced phase reset ===")

# Start from chaotic state (low r)
theta_chaos = np.random.uniform(0, 2 * np.pi, N)
K_defib = 2.0
dt = 0.005

# Run to chaos (with reduced coupling)
for _ in range(2000):
    dtheta = np.copy(omegas)
    for i in range(N):
        nb = np.where(adj[i])[0]
        if len(nb) > 0:
            dtheta[i] += 0.3 * np.mean(np.sin(theta_chaos[nb] - theta_chaos[i]))
    theta_chaos += dt * dtheta

r_pre, _ = order_param(theta_chaos)

# DEFIB: reset all phases to 0 (± noise from electrode distance)
defib_noise = 0.3  # ~17 degrees phase uncertainty
theta_defib = np.random.normal(0, defib_noise, N)

# Run with healthy coupling after reset
r_post_trace = []
for step in range(4000):
    dtheta = np.copy(omegas)
    for i in range(N):
        nb = np.where(adj[i])[0]
        if len(nb) > 0:
            dtheta[i] += K_defib * np.mean(np.sin(theta_defib[nb] - theta_defib[i]))
    theta_defib += dt * dtheta
    if step % 20 == 0:
        r, _ = order_param(theta_defib)
        r_post_trace.append(r)

r_post = np.mean(r_post_trace[-20:])
recovery_time = None
for i, r in enumerate(r_post_trace):
    if r > 0.8:
        recovery_time = i * 20 * dt
        break

add_finding("DEFIBRILLATION", "Phase reset restores rhythm", {
    "r_before_defib": round(float(r_pre), 4),
    "r_after_defib": round(float(r_post), 4),
    "recovery_time_s": round(float(recovery_time), 2) if recovery_time else "never",
    "defib_phase_noise_rad": defib_noise,
    "model": "phase reset + healthy coupling restores order",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "cardiac_sa_node", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
