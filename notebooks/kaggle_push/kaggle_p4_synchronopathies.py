# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 4 Test: Synchronopathies (Disease as Desync)
#
# Paper 4 (p.67-68) defines diseases as synchronisation failures:
#
# 1. Epilepsy = HYPER-synchrony: sigma > sigma_c, typically 1.5-2.0
#    Avalanches: tau < 1.5 (vs 1.5 healthy). System escapes critical regime.
#
# 2. Parkinson's = pathological BETA synchrony:
#    PLV_STN-M1(beta) > 0.6 (vs <0.3 healthy)
#    "Frozen" network state — movement initiation blocked.
#
# 3. Alzheimer's = hierarchical DESYNCHRONISATION:
#    Decreased long-range gamma coherence: C_long < 0.2
#    Increased local theta/delta power: P(theta/delta) > 2x baseline
#    Loss of cross-freq coupling: MI(theta,gamma) < 0.1 bits
#
# 4. Autism = altered E/I balance:
#    sigma_ASD ~ 0.7-0.8 (subcritical) or 1.2-1.3 (supercritical)
#    Hypo- or hyper-sensitivity depending on direction.
#
# 5. Recovery: transcranial stimulation, pharmacology, sensory entrainment.

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

# --- Test 1: Epilepsy as supercritical sigma ---
print("=== Test 1: Epilepsy — sigma > 1 drives hypersynchrony ===")

N = 200
omega_0 = 2 * np.pi * 10
omegas = np.random.normal(omega_0, omega_0 * 0.1, N)

# Random network
adj = (np.random.rand(N, N) < 6.0/N).astype(float)
adj = np.maximum(adj, adj.T)
np.fill_diagonal(adj, 0)

sigma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
epilepsy_results = []

for sigma_br in sigma_values:
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1
    prop_probs = adj * sigma_br / row_sums[:, None]

    # Run avalanches
    avalanche_sizes = []
    for trial in range(2000):
        active = np.zeros(N, dtype=bool)
        active[np.random.randint(N)] = True
        total = 1
        for gen in range(30):
            new_active = np.zeros(N, dtype=bool)
            for i in np.where(active)[0]:
                for j in np.where(adj[i] > 0)[0]:
                    if np.random.rand() < prop_probs[i, j] and not new_active[j]:
                        new_active[j] = True
            if not np.any(new_active):
                break
            active = new_active
            total += int(np.sum(new_active))
        avalanche_sizes.append(total)

    sizes = np.array(avalanche_sizes)
    mean_size = np.mean(sizes)
    max_size = int(np.max(sizes))
    # Fraction of system-spanning avalanches
    spanning = np.sum(sizes > N * 0.5) / len(sizes)

    # Fit power law exponent
    sizes_gt1 = sizes[sizes > 1]
    tau_est = float('nan')
    if len(sizes_gt1) > 20:
        bins = np.logspace(0, np.log10(max(sizes_gt1)), 15)
        counts, edges = np.histogram(sizes_gt1, bins=bins)
        bc = 0.5 * (edges[:-1] + edges[1:])
        valid = counts > 0
        if np.sum(valid) > 3:
            slope, _ = np.polyfit(np.log10(bc[valid]), np.log10(counts[valid]), 1)
            tau_est = -slope

    epilepsy_results.append({
        "sigma": sigma_br,
        "mean_avalanche_size": round(float(mean_size), 1),
        "max_avalanche_size": max_size,
        "spanning_fraction": round(float(spanning), 4),
        "tau_exponent": round(float(tau_est), 3) if not np.isnan(tau_est) else "nan",
        "regime": "subcritical" if sigma_br < 0.9 else "critical" if sigma_br < 1.1 else "supercritical",
    })

add_finding("EPILEPSY_SIGMA", "Epilepsy: supercritical sigma drives system-spanning avalanches", {
    "results": epilepsy_results,
    "paper4_prediction": "tau < 1.5 at sigma > 1 (supercritical)",
    "healthy_sigma": 1.0,
    "epileptic_sigma": "1.5-2.0",
    "equation": "Paper 4, p.67: sigma_epileptic > sigma_critical, tau < 1.5",
})

# --- Test 2: Parkinson's as pathological beta lock ---
print("\n=== Test 2: Parkinson's — excessive beta-band phase locking ===")

N_pd = 100
# STN and M1 populations
N_stn = 50
N_m1 = 50
omega_beta = 2 * np.pi * 20  # 20 Hz beta

# Healthy: weak STN-M1 coupling (PLV < 0.3)
# Parkinson's: strong STN-M1 coupling (PLV > 0.6)
K_within = 2.0  # intra-population
K_stn_m1_values = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]  # inter-population

pd_results = []
for K_cross in K_stn_m1_values:
    omegas_stn = np.random.normal(omega_beta, omega_beta * 0.05, N_stn)
    omegas_m1 = np.random.normal(omega_beta * 0.9, omega_beta * 0.08, N_m1)  # M1 slightly different

    theta_stn = np.random.uniform(0, 2 * np.pi, N_stn)
    theta_m1 = np.random.uniform(0, 2 * np.pi, N_m1)

    dt = 0.001
    for step in range(10000):
        z_stn = np.mean(np.exp(1j * theta_stn))
        z_m1 = np.mean(np.exp(1j * theta_m1))

        # STN dynamics
        dth_stn = omegas_stn + K_within * np.abs(z_stn) * np.sin(np.angle(z_stn) - theta_stn)
        dth_stn += K_cross * np.abs(z_m1) * np.sin(np.angle(z_m1) - theta_stn)
        theta_stn += dt * dth_stn

        # M1 dynamics
        dth_m1 = omegas_m1 + K_within * np.abs(z_m1) * np.sin(np.angle(z_m1) - theta_m1)
        dth_m1 += K_cross * np.abs(z_stn) * np.sin(np.angle(z_stn) - theta_m1)
        theta_m1 += dt * dth_m1

    # Phase Locking Value
    phase_diff = np.angle(z_stn) - np.angle(z_m1)
    r_stn, _ = order_param(theta_stn)
    r_m1, _ = order_param(theta_m1)
    plv = np.abs(np.mean(np.exp(1j * (theta_stn[:N_m1] - theta_m1))))

    pd_results.append({
        "K_STN_M1": round(K_cross, 2),
        "PLV_STN_M1": round(float(plv), 4),
        "r_STN": round(float(r_stn), 4),
        "r_M1": round(float(r_m1), 4),
        "clinical": "healthy" if plv < 0.3 else "parkinsonian" if plv > 0.6 else "borderline",
    })

add_finding("PARKINSON_BETA", "Parkinson's: excessive STN-M1 beta coupling", {
    "results": pd_results,
    "PLV_threshold_healthy": "<0.3",
    "PLV_threshold_parkinsonian": ">0.6",
    "DBS_mechanism": "deep brain stimulation disrupts pathological PLV",
    "equation": "Paper 4, p.67: PLV_STN-M1(beta) > 0.6",
})

# --- Test 3: Alzheimer's hierarchical desynchronisation ---
print("\n=== Test 3: Alzheimer's — progressive long-range desync ===")

N_alz = 100
# Two coupled regions (long-range)
N_local = 50
K_local = 3.0  # strong local coupling

# Progressive long-range coupling loss
K_long_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
alz_results = []

for K_long in K_long_values:
    # Region 1: gamma frequency
    omegas_r1 = np.random.normal(2 * np.pi * 40, 2 * np.pi * 3, N_local)
    # Region 2: gamma frequency
    omegas_r2 = np.random.normal(2 * np.pi * 40, 2 * np.pi * 3, N_local)

    theta_r1 = np.random.uniform(0, 2 * np.pi, N_local)
    theta_r2 = np.random.uniform(0, 2 * np.pi, N_local)

    dt = 0.0005
    for step in range(20000):
        z1 = np.mean(np.exp(1j * theta_r1))
        z2 = np.mean(np.exp(1j * theta_r2))

        dth1 = omegas_r1 + K_local * np.abs(z1) * np.sin(np.angle(z1) - theta_r1)
        dth1 += K_long * np.abs(z2) * np.sin(np.angle(z2) - theta_r1)
        theta_r1 += dt * dth1

        dth2 = omegas_r2 + K_local * np.abs(z2) * np.sin(np.angle(z2) - theta_r2)
        dth2 += K_long * np.abs(z1) * np.sin(np.angle(z1) - theta_r2)
        theta_r2 += dt * dth2

    r1, _ = order_param(theta_r1)
    r2, _ = order_param(theta_r2)
    # Long-range gamma coherence
    C_long = np.abs(np.mean(np.exp(1j * (theta_r1 - theta_r2))))

    alz_results.append({
        "K_long_range": round(K_long, 3),
        "C_gamma_long_range": round(float(C_long), 4),
        "r_local_1": round(float(r1), 4),
        "r_local_2": round(float(r2), 4),
        "clinical": "healthy" if C_long > 0.3 else "MCI" if C_long > 0.15 else "Alzheimers",
    })

add_finding("ALZHEIMER_DESYNC", "Alzheimer's: progressive long-range gamma desync", {
    "results": alz_results,
    "paper4_criteria": "C_long < 0.2, local theta/delta > 2x, MI(theta,gamma) < 0.1",
    "mechanism": "amyloid plaques disrupt long-range axonal coupling",
    "local_stays_high": "local r remains high while long-range C drops",
})

# --- Test 4: Autism E/I balance shift ---
print("\n=== Test 4: Autism — shifted E/I balance ===")

N_asd = 100
# E/I ratio controls effective sigma
ei_ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
asd_results = []

for ei in ei_ratios:
    K_exc = 2.0 * ei
    K_inh = 2.0 * (2 - ei)  # complementary

    omegas_asd = np.random.normal(omega_0, omega_0 * 0.15, N_asd)
    theta_asd = np.random.uniform(0, 2 * np.pi, N_asd)

    dt = 0.001
    r_trace_asd = []
    for step in range(15000):
        z = np.mean(np.exp(1j * theta_asd))
        r = np.abs(z)
        psi = np.angle(z)
        # Excitatory coupling (positive)
        dtheta = omegas_asd + K_exc * r * np.sin(psi - theta_asd)
        # Inhibitory coupling (negative, pushes toward anti-phase)
        dtheta -= K_inh * r * np.sin(theta_asd - psi + np.pi)
        # Net noise
        dtheta += 0.5 * np.random.randn(N_asd)
        theta_asd += dt * dtheta

        if step > 10000 and step % 50 == 0:
            r_trace_asd.append(float(r))

    r_arr = np.array(r_trace_asd)
    mi_metastability = float(np.std(r_arr))

    asd_results.append({
        "EI_ratio": round(ei, 2),
        "r_mean": round(float(np.mean(r_arr)), 4),
        "MI_metastability": round(mi_metastability, 4),
        "regime": "subcritical" if ei < 0.85 else "critical" if ei < 1.15 else "supercritical",
        "sensitivity": "hypo" if ei < 0.85 else "optimal" if ei < 1.15 else "hyper",
    })

add_finding("AUTISM_EI_BALANCE", "Autism: shifted E/I balance alters criticality", {
    "results": asd_results,
    "paper4_sigma_ASD": "0.7-0.8 (subcritical) or 1.2-1.3 (supercritical)",
    "optimal_EI": 1.0,
    "clinical": "explains both hypo-sensitive and hyper-sensitive ASD phenotypes",
})

# --- Test 5: Therapeutic synchronisation recovery ---
print("\n=== Test 5: Recovery via transcranial stimulation ===")

# Start from parkinsonian state (high beta PLV)
N_rec = 100
omegas_rec = np.random.normal(omega_beta, omega_beta * 0.05, N_rec)
theta_rec = np.random.uniform(0, 2 * np.pi, N_rec)
K_pathological = 5.0  # too high → frozen state

# Drive to pathological state
dt = 0.001
for step in range(5000):
    z = np.mean(np.exp(1j * theta_rec))
    dtheta = omegas_rec + K_pathological * np.abs(z) * np.sin(np.angle(z) - theta_rec)
    theta_rec += dt * dtheta

r_before, _ = order_param(theta_rec)

# Apply desynchronising stimulation (random phase reset)
# DBS-like: inject random phase perturbations at 130 Hz
f_dbs = 130  # Hz
A_dbs = 2.0  # strong perturbation

r_recovery = []
for step in range(20000):
    t = step * dt
    z = np.mean(np.exp(1j * theta_rec))
    dtheta = omegas_rec + K_pathological * np.abs(z) * np.sin(np.angle(z) - theta_rec)
    # DBS perturbation
    dtheta += A_dbs * np.sin(2 * np.pi * f_dbs * t + np.random.uniform(0, 2*np.pi, N_rec))
    theta_rec += dt * dtheta

    if step % 100 == 0:
        r, _ = order_param(theta_rec)
        r_recovery.append(float(r))

add_finding("DBS_RECOVERY", "DBS desynchronises pathological beta lock", {
    "r_before_DBS": round(float(r_before), 4),
    "r_during_DBS": round(float(np.mean(r_recovery[-20:])), 4),
    "reduction_percent": round(float((1 - np.mean(r_recovery[-20:]) / r_before) * 100), 1),
    "f_DBS_Hz": f_dbs,
    "mechanism": "high-frequency stimulation disrupts pathological phase locking",
    "paper4": "recovery via targeted synchronisation (p.68)",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_synchronopathies", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
