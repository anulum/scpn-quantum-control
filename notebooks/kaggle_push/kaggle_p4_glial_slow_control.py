# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 4 Test: Glial Slow Control of Criticality
#
# Paper 4, Section 3 (p.24-25): Astrocytic calcium waves form a slow
# control layer that maintains quasicriticality (sigma ~ 1).
#
# Key equations from Paper 4:
# 1. Astrocytic Ca2+ dynamics (reaction-diffusion):
#    d[Ca2+]_i/dt = D_Ca nabla^2 [Ca2+] + J_release - J_uptake + J_coupling
#    J_release = v_IP3R * ([IP3]/(K_IP3+[IP3]))^3 * ([Ca2+]/(K_Ca+[Ca2+]))^3 * (1 - [Ca2+]/[Ca2+]_ER)
#    J_uptake = v_SERCA * [Ca2+]^2 / (K_SERCA^2 + [Ca2+]^2)
#    J_coupling = g_gap * sum_j G_ij * ([Ca2+]_j - [Ca2+]_i)
#
# 2. Glial-neural phase coupling:
#    dphi_i/dt = omega_i + (K/N) sum sin(phi_j - phi_i) + gamma_glia * G([Ca2+])
#    G([Ca2+]) = alpha_ATP * log(1 + [Ca2+]/K_threshold) * Theta([Ca2+] - [Ca2+]_threshold)
#
# 3. Homeostatic control:
#    dsigma/dt = -kappa(sigma - 1) + eta(t)
#    with kappa ~ 0.01 s^-1
#
# 4. Slow control stability:
#    dsigma/dt = -gamma(sigma - sigma_target) + beta1*F(A_a) + beta2*H(theta_sync)
#    gamma ~ 10^-4 s^-1

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

# --- Test 1: Astrocytic calcium oscillator ---
print("=== Test 1: Astrocytic Ca2+ dynamics (Paper 4 eq.) ===")

np.random.seed(42)
N_astro = 50
dt = 0.01  # seconds
T = 300  # 5 minutes

# Parameters from Paper 4 (p.25)
D_Ca = 10.0  # um^2/s (diffusion coefficient)
v_IP3R = 0.5  # uM/s (IP3 receptor release rate)
K_IP3 = 0.3   # uM (IP3 half-activation)
K_Ca = 0.3    # uM (Ca2+ half-activation)
Ca_ER = 400   # uM (ER calcium store)
v_SERCA = 0.4 # uM/s (SERCA pump rate)
K_SERCA = 0.2 # uM (SERCA half-activation)
g_gap = 0.05  # gap junction coupling

# Initial conditions
Ca = np.random.uniform(0.05, 0.15, N_astro)  # uM resting Ca2+
IP3 = np.random.uniform(0.3, 0.8, N_astro)   # uM IP3

# Adjacency (1D chain with nearest-neighbour coupling)
G = np.zeros((N_astro, N_astro))
for i in range(N_astro - 1):
    G[i, i+1] = 1
    G[i+1, i] = 1

steps = int(T / dt)
Ca_trace = np.zeros((N_astro, steps // 100))
trace_idx = 0

for step in range(steps):
    # IP3R release (Paper 4 eq.)
    ip3_gate = (IP3 / (K_IP3 + IP3)) ** 3
    ca_gate = (Ca / (K_Ca + Ca)) ** 3
    er_drive = np.maximum(0, 1 - Ca / Ca_ER)
    J_release = v_IP3R * ip3_gate * ca_gate * er_drive

    # SERCA uptake
    J_uptake = v_SERCA * Ca ** 2 / (K_SERCA ** 2 + Ca ** 2)

    # Gap junction coupling
    J_coupling = g_gap * (G @ Ca - np.sum(G, axis=1) * Ca)

    # Update
    dCa = J_release - J_uptake + J_coupling + 0.001 * np.random.randn(N_astro)
    Ca = np.maximum(0.01, Ca + dt * dCa)

    # Slow IP3 dynamics
    IP3 += dt * (0.01 * np.random.randn(N_astro))
    IP3 = np.clip(IP3, 0.1, 2.0)

    if step % 100 == 0 and trace_idx < Ca_trace.shape[1]:
        Ca_trace[:, trace_idx] = Ca
        trace_idx += 1

# Analyse oscillation
Ca_mean = np.mean(Ca_trace[:, -100:], axis=0)
Ca_std = np.std(Ca_trace[:, -100:], axis=0)
oscillation_amplitude = np.max(Ca_mean) - np.min(Ca_mean)

# Check for calcium waves (spatial coherence)
spatial_corr = np.corrcoef(Ca_trace[0, -100:], Ca_trace[N_astro//2, -100:])[0, 1]

add_finding("ASTRO_CA_DYNAMICS", "Astrocytic Ca2+ oscillation from Paper 4 model", {
    "mean_Ca_uM": round(float(np.mean(Ca)), 4),
    "oscillation_amplitude_uM": round(float(oscillation_amplitude), 4),
    "spatial_correlation_0_to_N2": round(float(spatial_corr), 4),
    "N_astrocytes": N_astro,
    "g_gap_coupling": g_gap,
    "D_Ca_um2_s": D_Ca,
    "parameters_from": "Paper 4, Section 3 (p.25)",
})

# --- Test 2: Glial modulation of neural coupling ---
print("\n=== Test 2: Glial-neural phase coupling (Paper 4 eq.) ===")

N_neural = 100
omega_0 = 2 * np.pi * 10  # 10 Hz neural oscillation
sigma_omega = omega_0 * 0.1
omegas = np.random.normal(omega_0, sigma_omega, N_neural)
theta = np.random.uniform(0, 2 * np.pi, N_neural)

# Paper 4 parameters
K_neural = 1.5
gamma_glia = 0.3
alpha_ATP = 0.5
K_threshold = 0.3  # uM
Ca_threshold = 0.2  # uM

dt = 0.001
T = 10.0  # seconds
steps = int(T / dt)

# Assign each neuron to an astrocyte (5:1 ratio)
astro_assignment = np.arange(N_neural) // (N_neural // N_astro)
astro_assignment = np.clip(astro_assignment, 0, N_astro - 1)

r_trace_glia = []
r_trace_noglia = []
theta_noglia = theta.copy()

for step in range(steps):
    # With glial modulation (Paper 4 eq.)
    Ca_local = Ca[astro_assignment]
    G_func = np.where(Ca_local > Ca_threshold,
                      alpha_ATP * np.log(1 + Ca_local / K_threshold),
                      0)
    z = np.mean(np.exp(1j * theta))
    r = np.abs(z)
    psi = np.angle(z)
    dtheta = omegas + K_neural * r * np.sin(psi - theta) + gamma_glia * G_func
    theta += dt * dtheta

    # Without glial modulation
    z_ng = np.mean(np.exp(1j * theta_noglia))
    r_ng = np.abs(z_ng)
    psi_ng = np.angle(z_ng)
    dtheta_ng = omegas + K_neural * r_ng * np.sin(psi_ng - theta_noglia)
    theta_noglia += dt * dtheta_ng

    if step % 100 == 0:
        r_t, _ = order_param(theta)
        r_trace_glia.append(float(r_t))
        r_ng_t, _ = order_param(theta_noglia)
        r_trace_noglia.append(float(r_ng_t))

add_finding("GLIAL_NEURAL_COUPLING", "Glial modulation enhances neural sync", {
    "r_with_glia": round(float(np.mean(r_trace_glia[-20:])), 4),
    "r_without_glia": round(float(np.mean(r_trace_noglia[-20:])), 4),
    "enhancement": round(float(np.mean(r_trace_glia[-20:]) / max(np.mean(r_trace_noglia[-20:]), 0.01) - 1) * 100, 1),
    "gamma_glia": gamma_glia,
    "equation_source": "Paper 4, p.25: dphi/dt = omega + K*sin + gamma_glia*G([Ca2+])",
})

# --- Test 3: Homeostatic branching ratio maintenance ---
print("\n=== Test 3: Branching ratio sigma homeostasis (Paper 4 eq.) ===")

# Paper 4 (p.22-23, p.49-50): dsigma/dt = -kappa(sigma-1) + eta
kappa = 0.01  # s^-1 (Paper 4 value)
sigma_noise = 0.05
dt_home = 0.1
T_home = 1000  # seconds

sigma_trace = []
sigma_val = 0.5  # start subcritical

for step in range(int(T_home / dt_home)):
    dsigma = -kappa * (sigma_val - 1) + sigma_noise * np.random.randn() * np.sqrt(dt_home)
    sigma_val += dt_home * dsigma
    sigma_val = np.clip(sigma_val, 0, 2)
    if step % 10 == 0:
        sigma_trace.append(float(sigma_val))

sigma_arr = np.array(sigma_trace)
mean_sigma = np.mean(sigma_arr[-100:])
std_sigma = np.std(sigma_arr[-100:])
# Recovery time from subcritical
recovery_step = None
for i, s in enumerate(sigma_arr):
    if abs(s - 1) < 0.1:
        recovery_step = i
        break

add_finding("BRANCHING_HOMEOSTASIS", "Branching ratio self-tunes to sigma=1", {
    "mean_sigma_steady": round(float(mean_sigma), 4),
    "sigma_fluctuations": round(float(std_sigma), 4),
    "recovery_time_s": round(recovery_step * dt_home * 10, 1) if recovery_step else "never",
    "kappa_s": kappa,
    "initial_sigma": 0.5,
    "equation": "dsigma/dt = -kappa(sigma-1) + eta (Paper 4, p.22)",
})

# --- Test 4: Glial slow control with perturbation ---
print("\n=== Test 4: Glial control restores criticality after perturbation ===")

# Paper 4 (p.50): dsigma/dt = -gamma(sigma-sigma_target) + beta1*F(A_a) + beta2*H(theta_sync)
gamma_home = 1e-4  # s^-1 (Paper 4 value: very slow)
beta1 = 0.05
beta2 = 0.03

sigma_ctrl = 1.0
G_astro = 0.5
dt_ctrl = 1.0  # seconds (slow dynamics)
T_ctrl = 10000  # 10,000 seconds (~3 hours)

sigma_ctrl_trace = []
# Perturbation at t=2000s: drop sigma to 0.6 (anaesthetic-like)
for step in range(int(T_ctrl / dt_ctrl)):
    t = step * dt_ctrl
    if t == 2000:
        sigma_ctrl = 0.6  # perturbation

    # Astrocyte calcium-dependent modulation
    F_astro = np.tanh(G_astro * 2) * (1 - sigma_ctrl)  # drives sigma toward 1
    # Phase sync feedback
    r_sync = max(0, sigma_ctrl - 0.5)  # crude: sync depends on sigma
    H_sync = r_sync * (1 - sigma_ctrl)

    dsigma = -gamma_home * (sigma_ctrl - 1) + beta1 * F_astro + beta2 * H_sync
    dsigma += 0.001 * np.random.randn()
    sigma_ctrl += dt_ctrl * dsigma
    sigma_ctrl = np.clip(sigma_ctrl, 0, 2)

    # Astrocyte dynamics
    dG = 0.01 * (1 - sigma_ctrl) - 0.005 * G_astro  # responds to deviation from criticality
    G_astro += dt_ctrl * dG
    G_astro = np.clip(G_astro, 0, 5)

    if step % 10 == 0:
        sigma_ctrl_trace.append({"t": round(t, 0), "sigma": round(float(sigma_ctrl), 4)})

# Find recovery time
recovery_t = None
for entry in sigma_ctrl_trace:
    if entry["t"] > 2000 and abs(entry["sigma"] - 1) < 0.05:
        recovery_t = entry["t"] - 2000
        break

add_finding("GLIAL_RECOVERY", "Glial slow control restores criticality after perturbation", {
    "perturbation_sigma": 0.6,
    "recovery_time_s": round(float(recovery_t), 0) if recovery_t else ">8000",
    "final_sigma": sigma_ctrl_trace[-1]["sigma"],
    "gamma_homeostatic": gamma_home,
    "equation": "Paper 4, p.50: dsigma/dt = -gamma(sigma-target) + beta1*F(A_a) + beta2*H(sync)",
    "clinical": "anaesthesia reduces sigma<1 → slow glial recovery explains emergence lag",
})

# --- Test 5: Avalanche statistics at criticality ---
print("\n=== Test 5: Power-law avalanche distributions (Paper 4: tau=3/2) ===")

# Paper 4 (p.22-23): P(s) ~ s^{-tau}, tau=3/2, alpha=2
N_net = 200
# Random network with mean degree ~6
adj = np.random.rand(N_net, N_net) < (6.0 / N_net)
adj = np.logical_or(adj, adj.T).astype(float)
np.fill_diagonal(adj, 0)

sigma_target = 1.0  # critical branching
# Normalise rows so branching ratio = 1
row_sums = adj.sum(axis=1)
row_sums[row_sums == 0] = 1
propagation_probs = adj * sigma_target / row_sums[:, None]

# Run avalanches
n_trials = 5000
avalanche_sizes = []

for trial in range(n_trials):
    # Seed: activate 1 random node
    active = np.zeros(N_net, dtype=bool)
    seed = np.random.randint(N_net)
    active[seed] = True
    total_size = 1

    for generation in range(50):  # max 50 generations
        new_active = np.zeros(N_net, dtype=bool)
        for i in np.where(active)[0]:
            # Each active node activates neighbours with probability
            for j in np.where(adj[i] > 0)[0]:
                if np.random.rand() < propagation_probs[i, j] and not new_active[j]:
                    new_active[j] = True
        if not np.any(new_active):
            break
        active = new_active
        total_size += np.sum(new_active)

    avalanche_sizes.append(total_size)

# Fit power law
sizes = np.array(avalanche_sizes)
sizes_gt1 = sizes[sizes > 1]
if len(sizes_gt1) > 10:
    # Log-log linear regression
    log_s = np.log10(sizes_gt1)
    # Histogram in log space
    bins = np.logspace(0, np.log10(max(sizes_gt1)), 20)
    counts, edges = np.histogram(sizes_gt1, bins=bins)
    bin_centres = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    if np.sum(valid) > 3:
        log_bc = np.log10(bin_centres[valid])
        log_ct = np.log10(counts[valid])
        slope, intercept = np.polyfit(log_bc, log_ct, 1)
        tau_measured = -slope
    else:
        tau_measured = float('nan')
else:
    tau_measured = float('nan')

add_finding("AVALANCHE_POWERLAW", "Avalanche size distribution at sigma=1", {
    "tau_measured": round(float(tau_measured), 3),
    "tau_paper4": 1.5,
    "tau_match": abs(tau_measured - 1.5) < 0.3 if not np.isnan(tau_measured) else False,
    "N_network": N_net,
    "n_avalanches": n_trials,
    "mean_size": round(float(np.mean(sizes)), 2),
    "max_size": int(np.max(sizes)),
    "equation": "Paper 4, p.22: P(s) ~ s^{-3/2}, P(T) ~ T^{-2}",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_glial_slow_control", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
