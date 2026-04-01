# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 4: Decoherence Cascade + Adaptive Plasticity
import json

import numpy as np
from scipy.special import jv as bessel_j

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

# === PART A: Decoherence-Protected Amplification ===
print("=== Part A: Quantum-to-classical amplification cascade ===")

# Paper 4 (p.66-67): 4 stages, each amplifies by orders of magnitude
stages = {
    "1_molecular": {
        "timescale": (1e-15, 1e-12),
        "amplification": 1e3,
        "mechanism": "quantum coherence",
    },
    "2_protein": {
        "timescale": (1e-9, 1e-6),
        "amplification": 1e4,
        "mechanism": "conformational waves v~1000 m/s",
    },
    "3_cellular": {
        "timescale": (1e-3, 1e0),
        "amplification": 1e5,
        "mechanism": "Ca2+ oscillation gating",
    },
    "4_tissue": {
        "timescale": (1e0, 1e3),
        "amplification": 1e3,
        "mechanism": "gap junction coupling",
    },
}

# Simulate the cascade: signal starts as quantum fluctuation
signal_amplitude = 1e-20  # initial quantum fluctuation (Joules)
cascade_results = []

for stage_name, params in stages.items():
    signal_amplitude *= params["amplification"]
    cascade_results.append(
        {
            "stage": stage_name,
            "timescale_s": f"{params['timescale'][0]:.0e} - {params['timescale'][1]:.0e}",
            "amplification": f"{params['amplification']:.0e}",
            "cumulative_signal": f"{signal_amplitude:.2e}",
            "mechanism": params["mechanism"],
        }
    )

total_amp = 1e3 * 1e4 * 1e5 * 1e3
final_signal = 1e-20 * total_amp

add_finding(
    "AMPLIFICATION_CASCADE",
    "Quantum-to-tissue amplification (Paper 4, p.66)",
    {
        "stages": cascade_results,
        "total_amplification": f"{total_amp:.0e}",
        "initial_quantum_J": "1e-20",
        "final_tissue_J": f"{final_signal:.2e}",
        "final_as_kBT": round(final_signal / (1.38e-23 * 310), 1),
        "detectable": final_signal > 1.38e-23 * 310,
        "equation": "Paper 4: A_total = prod A_i = 10^15",
    },
)

# --- Simulate Stage 2: Protein conformational wave ---
print("\n--- A2: Protein conformational wave propagation ---")

# Paper 4: d2u/dt2 = v^2 * d2u/dx2 - gamma*du/dt + F_quantum
v_conf = 1000  # m/s (Paper 4 value)
gamma_conf = 100  # damping
L_protein = 10e-9  # 10 nm protein
dx = 0.5e-9  # 0.5 nm steps
Nx_prot = int(L_protein / dx)
dt_prot = 0.5 * dx / v_conf  # CFL

u = np.zeros(Nx_prot)
v_u = np.zeros(Nx_prot)
u[0] = 1e-12  # initial quantum-driven displacement (pm)

u_max_trace = []
for step in range(1000):
    d2u = np.zeros(Nx_prot)
    d2u[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    a = v_conf**2 * d2u - gamma_conf * v_u
    v_u += dt_prot * a
    u += dt_prot * v_u
    u_max_trace.append(float(np.max(np.abs(u))))

propagation_time = Nx_prot * dx / v_conf

add_finding(
    "PROTEIN_WAVE",
    "Conformational wave in protein (Paper 4, p.66)",
    {
        "v_propagation_m_s": v_conf,
        "L_protein_nm": L_protein * 1e9,
        "propagation_time_ps": round(propagation_time * 1e12, 2),
        "max_displacement_pm": round(float(max(u_max_trace)) * 1e12, 4),
        "damping_rate": gamma_conf,
    },
)

# === PART B: Adaptive Synchronisation Plasticity ===
print("\n=== Part B: Coupling plasticity + memory formation ===")

# Paper 4 (p.75): dK_ij/dt = eta_K * [R_ij - R_target]
N_plast = 50
omega_plast = 2 * np.pi * 10
omegas_p = np.random.normal(omega_plast, omega_plast * 0.1, N_plast)
theta_p = np.random.uniform(0, 2 * np.pi, N_plast)

# Start with uniform coupling
K_matrix = np.ones((N_plast, N_plast)) * 0.5
np.fill_diagonal(K_matrix, 0)

eta_K = 0.001  # learning rate
R_target = 0.3  # target PLV (Paper 4 value)
dt = 0.001

r_trace_plast = []
K_mean_trace = []
K_std_trace = []

for step in range(30000):
    # Kuramoto dynamics
    z = np.mean(np.exp(1j * theta_p))
    r = np.abs(z)
    psi = np.angle(z)
    dtheta = omegas_p + r * np.mean(K_matrix, axis=1) * np.sin(psi - theta_p)
    dtheta += 0.3 * np.random.randn(N_plast)
    theta_p += dt * dtheta

    # Plasticity: update K based on pairwise phase locking
    if step % 100 == 0 and step > 5000:
        for i in range(N_plast):
            for j in range(i + 1, min(i + 10, N_plast)):  # local only
                R_ij = np.cos(theta_p[i] - theta_p[j])
                dK = eta_K * (R_ij - R_target)
                K_matrix[i, j] += dK
                K_matrix[j, i] += dK
                K_matrix[i, j] = np.clip(K_matrix[i, j], 0, 5)
                K_matrix[j, i] = K_matrix[i, j]

    if step % 200 == 0:
        r_trace_plast.append(float(r))
        K_vals = K_matrix[K_matrix > 0]
        K_mean_trace.append(float(np.mean(K_vals)))
        K_std_trace.append(float(np.std(K_vals)))

add_finding(
    "SYNC_PLASTICITY",
    "Coupling plasticity self-tunes network",
    {
        "r_initial": round(r_trace_plast[0], 4) if r_trace_plast else None,
        "r_final": round(r_trace_plast[-1], 4) if r_trace_plast else None,
        "K_initial_mean": round(K_mean_trace[0], 4) if K_mean_trace else None,
        "K_final_mean": round(K_mean_trace[-1], 4) if K_mean_trace else None,
        "K_final_std": round(K_std_trace[-1], 4) if K_std_trace else None,
        "R_target": R_target,
        "eta_K": eta_K,
        "equation": "Paper 4, p.75: dK_ij/dt = eta_K * [R_ij - R_target]",
    },
)

# --- Memory formation through synchrony ---
print("\n--- B2: Memory as persistent phase relationship ---")

# Paper 4: M_ij = integral W(t) * cos(phi_i - phi_j) dt
# Train: present a pattern (specific phase relationship) then test recall

N_mem = 30
omegas_mem = np.random.normal(omega_plast, omega_plast * 0.05, N_mem)
K_mem = np.ones((N_mem, N_mem)) * 1.0
np.fill_diagonal(K_mem, 0)

# Pattern: first 15 oscillators in-phase, last 15 shifted by pi
pattern = np.zeros(N_mem)
pattern[15:] = np.pi

# Training phase: clamp to pattern, update K
theta_train = pattern.copy() + 0.1 * np.random.randn(N_mem)
eta_mem = 0.01

for step in range(5000):
    # Clamp to pattern (strong)
    dtheta = omegas_mem + 0.1 * np.random.randn(N_mem)
    for i in range(N_mem):
        dtheta[i] += 2.0 * np.sin(pattern[i] - theta_train[i])  # pattern forcing
        for j in range(N_mem):
            if i != j:
                dtheta[i] += K_mem[i, j] * np.sin(theta_train[j] - theta_train[i]) / N_mem
    theta_train += dt * dtheta

    # Hebbian update
    if step % 50 == 0:
        for i in range(N_mem):
            for j in range(i + 1, N_mem):
                M_ij = np.cos(theta_train[i] - theta_train[j])
                K_mem[i, j] += eta_mem * M_ij
                K_mem[j, i] = K_mem[i, j]
                K_mem[i, j] = np.clip(K_mem[i, j], 0, 5)
                K_mem[j, i] = K_mem[i, j]

# Test recall: start from random, see if pattern emerges
theta_recall = np.random.uniform(0, 2 * np.pi, N_mem)
for step in range(10000):
    dtheta = omegas_mem.copy()
    for i in range(N_mem):
        for j in range(N_mem):
            if i != j:
                dtheta[i] += K_mem[i, j] * np.sin(theta_recall[j] - theta_recall[i]) / N_mem
    dtheta += 0.1 * np.random.randn(N_mem)
    theta_recall += dt * dtheta

# Check if pattern recalled: phase diff between groups
group1_phase = np.mean(np.exp(1j * theta_recall[:15]))
group2_phase = np.mean(np.exp(1j * theta_recall[15:]))
recalled_diff = np.abs(np.angle(group1_phase) - np.angle(group2_phase))
recalled_diff = min(recalled_diff, 2 * np.pi - recalled_diff)

add_finding(
    "SYNC_MEMORY",
    "Memory encoded in coupling strengths",
    {
        "pattern_phase_diff": round(float(np.pi), 4),
        "recalled_phase_diff": round(float(recalled_diff), 4),
        "recall_accuracy": round(float(1 - abs(recalled_diff - np.pi) / np.pi), 4),
        "K_mean_within_group": round(float(np.mean(K_mem[:15, :15])), 3),
        "K_mean_between_groups": round(float(np.mean(K_mem[:15, 15:])), 3),
        "equation": "Paper 4, p.76: M_ij = integral W(t) cos(phi_i - phi_j) dt",
    },
)

# === PART C: Ghost Stochastic Resonance ===
print("\n=== Part C: Ghost stochastic resonance (Paper 4, p.72) ===")

# Paper 4: P(f_ghost) = P_0 * [J_0(A/D)]^2 * delta(f - f_ghost)
# Two input frequencies f1, f2 → ghost at f_ghost = |nf1 ± mf2|

f1 = 40.0  # Hz (gamma)
f2 = 6.0  # Hz (theta)
# Ghost frequencies: f1-f2=34, f1+f2=46, 2f2=12, f1-2f2=28, etc.
# Most interesting: f2 subharmonic at f1/f2 ratio

A_signal = 0.5
D_values = np.logspace(-1, 1, 20)

ghost_results = []
for D in D_values:
    ratio = A_signal / D
    J0_val = bessel_j(0, ratio)
    P_ghost = J0_val**2

    ghost_results.append(
        {
            "D": round(float(D), 4),
            "A_over_D": round(float(ratio), 4),
            "J0_squared": round(float(P_ghost), 6),
        }
    )

# Optimal D for ghost detection
j0_vals = [x["J0_squared"] for x in ghost_results]
D_ghost_opt = ghost_results[np.argmax(j0_vals)]["D"]

# Ghost frequencies
ghost_freqs = []
for n in range(1, 4):
    for m in range(1, 4):
        for sign in [1, -1]:
            f_ghost = abs(n * f1 + sign * m * f2)
            if 0 < f_ghost < 200:
                ghost_freqs.append(
                    {"n": n, "m": m, "sign": "+" if sign > 0 else "-", "f_ghost_Hz": f_ghost}
                )

add_finding(
    "GHOST_SR",
    "Ghost stochastic resonance for missing frequencies",
    {
        "f1_Hz": f1,
        "f2_Hz": f2,
        "ghost_frequencies": ghost_freqs[:8],
        "D_optimal": D_ghost_opt,
        "peak_J0_squared": round(float(max(j0_vals)), 4),
        "equation": "Paper 4, p.72: P(f_ghost) = P_0 * J_0(A/D)^2",
        "significance": "detects cosmological rhythms not directly present in neural dynamics",
    },
)

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_decoherence_cascade_plasticity", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
