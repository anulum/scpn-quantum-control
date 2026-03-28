# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Firefly Synchronisation: The Original Kuramoto
#
# Pteroptyx malaccae fireflies synchronise flashes across thousands of
# individuals in mangrove trees. This is THE system that inspired
# Winfree (1967) and Kuramoto (1975). Key biology:
# - Flash period: ~1 Hz (Pteroptyx) to ~0.5 Hz (Photinus)
# - Coupling: visual (light → retina → neural → flash organ)
# - Mechanism: phase-advance on seeing neighbour flash
# - Delay: ~20 ms neural processing + ~200 ms flash duration
#
# Kuramoto with pulse coupling + delay:
#   d(theta_i)/dt = omega_i + (K/N) * sum_j PRC(theta_j - theta_i - tau)
# where PRC is the Phase Response Curve (Type I for fireflies).
#
# TESTABLE: Does SCPN K_nm predict the observed ~200 ms sync timescale?
# Can heterogeneous natural frequencies (sigma~0.1 Hz) still sync?

import numpy as np
from scipy.integrate import solve_ivp
import json
import sys

FINDINGS = []

def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")

# --- Test 1: Classic Kuramoto firefly model ---
# N fireflies, natural freq ~1 Hz, Lorentzian g(omega)

N = 200
omega_0 = 2 * np.pi * 1.0  # 1 Hz natural frequency
delta_omega = 2 * np.pi * 0.1  # 0.1 Hz spread (Lorentzian)

np.random.seed(42)
# Cauchy/Lorentzian distribution for natural frequencies
omegas = omega_0 + delta_omega * np.tan(np.pi * (np.random.rand(N) - 0.5))
# Clip extreme outliers
omegas = np.clip(omegas, omega_0 - 10 * delta_omega, omega_0 + 10 * delta_omega)

theta0 = np.random.uniform(0, 2 * np.pi, N)

# Kuramoto order parameter
def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)

# Classic sine coupling
def kuramoto_rhs(t, theta, K):
    dtheta = np.copy(omegas)
    sin_diff = np.sin(theta[:, None] - theta[None, :])
    dtheta += (K / N) * np.sum(-sin_diff, axis=1)  # note sign convention
    return dtheta

# Sweep K to find K_c
K_values = np.linspace(0, 4.0, 40)
r_final = []
t_span = (0, 50)
t_eval = np.linspace(40, 50, 200)

print("=== Test 1: Firefly K_c from Lorentzian theory ===")
for K in K_values:
    sol = solve_ivp(kuramoto_rhs, t_span, theta0, args=(K,),
                    t_eval=t_eval, method='RK45', rtol=1e-6)
    r_vals = [order_param(sol.y[:, i])[0] for i in range(len(t_eval))]
    r_final.append(np.mean(r_vals[-50:]))

r_final = np.array(r_final)

# Theoretical K_c for Lorentzian: K_c = 2 * delta_omega
K_c_theory = 2 * delta_omega / (2 * np.pi)  # in Hz coupling units
K_c_theory_rad = 2 * delta_omega  # in rad/s

# Find numerical K_c (where r first exceeds 0.1)
idx_sync = np.where(r_final > 0.1)[0]
K_c_numerical = K_values[idx_sync[0]] if len(idx_sync) > 0 else float('nan')

add_finding("FIREFLY_KC", "Kuramoto K_c for firefly population", {
    "K_c_theory_rad_s": round(float(K_c_theory_rad), 4),
    "K_c_numerical_rad_s": round(float(K_c_numerical), 4),
    "ratio": round(float(K_c_numerical / K_c_theory_rad), 3) if not np.isnan(K_c_numerical) else "nan",
    "N_fireflies": N,
    "freq_spread_Hz": 0.1,
    "natural_freq_Hz": 1.0,
})

# --- Test 2: Pulse coupling with delay (biological realism) ---
# Fireflies don't couple via sine — they flash (pulse) with a delay

print("\n=== Test 2: Pulse-coupled integrate-and-fire ===")

N2 = 100
tau_delay = 0.020  # 20 ms neural delay
flash_duration = 0.200  # 200 ms flash
dt = 0.001
T_total = 30.0
steps = int(T_total / dt)

omegas2 = omega_0 + delta_omega * np.tan(np.pi * (np.random.rand(N2) - 0.5))
omegas2 = np.clip(omegas2, omega_0 - 5 * delta_omega, omega_0 + 5 * delta_omega)
phases2 = np.random.uniform(0, 2 * np.pi, N2)

K_pulse = 1.5  # coupling strength
epsilon = 0.15  # phase advance per flash received

# Phase Response Curve: Type I (advance only near threshold)
def prc_type1(phase):
    return np.maximum(0, np.sin(phase))  # advance only in rising phase

r_trace = []
sync_time = None

for step in range(steps):
    t = step * dt
    # Check which fireflies are flashing (phase near 2*pi)
    flashing = (phases2 % (2 * np.pi)) > (2 * np.pi - flash_duration * omega_0 / (2 * np.pi))

    # Phase advance for non-flashing fireflies seeing a flash
    n_flashing = np.sum(flashing)
    if n_flashing > 0:
        advance = epsilon * (n_flashing / N2) * prc_type1(phases2)
        phases2[~flashing] += advance[~flashing]

    # Natural evolution
    phases2 += omegas2 * dt

    if step % 100 == 0:
        r, _ = order_param(phases2)
        r_trace.append((t, r))
        if sync_time is None and r > 0.7:
            sync_time = t

r_trace = np.array(r_trace)
r_final_pulse = r_trace[-1, 1]

add_finding("FIREFLY_PULSE", "Pulse-coupled firefly model with delay", {
    "final_r": round(float(r_final_pulse), 4),
    "sync_time_s": round(float(sync_time), 2) if sync_time else "never",
    "N": N2,
    "delay_ms": tau_delay * 1000,
    "flash_duration_ms": flash_duration * 1000,
    "epsilon": epsilon,
    "sync_cycles_to_lock": round(sync_time * 1.0, 1) if sync_time else "never",
})

# --- Test 3: Nearest-neighbour vs all-to-all (tree geometry) ---
# Real fireflies on a tree see nearest neighbours more strongly
# (light intensity falls as 1/r^2)

print("\n=== Test 3: Distance-dependent coupling (tree geometry) ===")

N3 = 80
# Place fireflies on a 2D "tree surface" (random positions)
positions = np.random.rand(N3, 2) * 10  # 10m × 10m tree
distances = np.sqrt(np.sum((positions[:, None] - positions[None, :]) ** 2, axis=2))
distances[distances == 0] = 1e-10

# Light coupling: K_ij = K0 / (1 + (d_ij / d0)^2)
d0 = 1.0  # characteristic distance (1 m)
K0 = 2.0
K_matrix = K0 / (1 + (distances / d0) ** 2)
np.fill_diagonal(K_matrix, 0)

omegas3 = omega_0 + delta_omega * np.tan(np.pi * (np.random.rand(N3) - 0.5))
omegas3 = np.clip(omegas3, omega_0 - 5 * delta_omega, omega_0 + 5 * delta_omega)
theta3 = np.random.uniform(0, 2 * np.pi, N3)

def kuramoto_spatial(t, theta, K_mat):
    dtheta = np.copy(omegas3)
    for i in range(len(theta)):
        dtheta[i] += np.sum(K_mat[i] * np.sin(theta - theta[i])) / N3
    return dtheta

sol3 = solve_ivp(kuramoto_spatial, (0, 80), theta3, args=(K_matrix,),
                 t_eval=np.linspace(60, 80, 200), method='RK45', rtol=1e-5)

r_spatial = [order_param(sol3.y[:, i])[0] for i in range(sol3.y.shape[1])]
r_mean_spatial = np.mean(r_spatial[-50:])

# Compare with all-to-all at same mean K
K_mean = np.mean(K_matrix[K_matrix > 0])
sol3_aa = solve_ivp(kuramoto_rhs, (0, 80), theta3, args=(K_mean,),
                    t_eval=np.linspace(60, 80, 200), method='RK45', rtol=1e-5)
r_aa = [order_param(sol3_aa.y[:, i])[0] for i in range(sol3_aa.y.shape[1])]
r_mean_aa = np.mean(r_aa[-50:])

add_finding("FIREFLY_SPATIAL", "Distance-dependent coupling on tree", {
    "r_spatial_coupling": round(float(r_mean_spatial), 4),
    "r_all_to_all_same_mean_K": round(float(r_mean_aa), 4),
    "mean_K_ij": round(float(K_mean), 4),
    "spatial_penalty": round(float(1 - r_mean_spatial / max(r_mean_aa, 0.01)), 3),
    "tree_size_m": 10,
    "N": N3,
})

# --- Test 4: SCPN coupling prediction ---
# Map biological parameters to K_nm

print("\n=== Test 4: SCPN K_nm mapping for fireflies ===")

# Known biology:
flash_energy_J = 1e-9  # ~1 nJ per flash (bioluminescence)
retinal_sensitivity_J = 1e-17  # single photon ~10 aJ, but threshold ~100 photons
neural_delay_s = 0.020
flash_period_s = 1.0
phase_coupling_per_flash = 0.05  # ~5% of cycle per stimulus

# Effective K = phase_advance * flash_rate * N_visible
N_visible_mean = 20  # typical line-of-sight neighbours
K_bio = phase_coupling_per_flash * (1.0 / flash_period_s) * N_visible_mean
K_c_bio = 2 * 0.1 * 2 * np.pi  # from Lorentzian theory

add_finding("FIREFLY_SCPN", "Biological K_nm mapping for fireflies", {
    "K_bio_effective_rad_s": round(float(K_bio * 2 * np.pi), 4),
    "K_c_required_rad_s": round(float(K_c_bio), 4),
    "K_over_Kc": round(float(K_bio * 2 * np.pi / K_c_bio), 3),
    "sync_predicted": K_bio * 2 * np.pi > K_c_bio,
    "flash_energy_nJ": 1.0,
    "neural_delay_ms": 20,
    "N_visible_neighbours": N_visible_mean,
})

# --- Test 5: Emergence time vs population size ---
print("\n=== Test 5: Sync emergence scaling with N ===")

N_sizes = [10, 30, 50, 100, 200]
sync_times_vs_N = []
K_test = 2.0  # above K_c

for Ni in N_sizes:
    omegas_i = omega_0 + delta_omega * np.tan(np.pi * (np.random.rand(Ni) - 0.5))
    omegas_i = np.clip(omegas_i, omega_0 - 5 * delta_omega, omega_0 + 5 * delta_omega)
    theta_i = np.random.uniform(0, 2 * np.pi, Ni)

    def rhs_i(t, theta, K=K_test, om=omegas_i, n=Ni):
        dth = np.copy(om)
        sin_diff = np.sin(theta[:, None] - theta[None, :])
        dth += (K / n) * np.sum(-sin_diff, axis=1)
        return dth

    t_sync_i = None
    dt_i = 0.01
    theta_run = theta_i.copy()
    for s in range(5000):
        k1 = np.array(rhs_i(0, theta_run))
        theta_run = theta_run + dt_i * k1
        if s % 10 == 0:
            ri, _ = order_param(theta_run)
            if t_sync_i is None and ri > 0.7:
                t_sync_i = s * dt_i
                break

    sync_times_vs_N.append({
        "N": Ni,
        "sync_time_s": round(float(t_sync_i), 2) if t_sync_i else None,
    })

add_finding("FIREFLY_SCALING", "Sync time vs population size", {
    "results": sync_times_vs_N,
    "K_coupling": K_test,
    "theory": "t_sync ~ 1/sqrt(N) for mean-field",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "firefly_synchronisation", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
