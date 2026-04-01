# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Power Grid Synchronisation = Kuramoto
import json

import numpy as np
from scipy.integrate import solve_ivp

FINDINGS = []


def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")


def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)


# --- Test 1: Swing equation as second-order Kuramoto ---
print("=== Test 1: Power grid swing equation ===")

N = 50  # generators
np.random.seed(42)

# Generator parameters
M = np.random.uniform(5, 15, N)  # inertia constants (seconds)
D = np.random.uniform(1, 3, N)  # damping
P = np.random.normal(0, 0.3, N)  # power injection (balanced grid: sum ≈ 0)
P -= np.mean(P)  # enforce power balance

# Network: random graph with ~4 connections per node
from scipy.spatial import Delaunay

positions = np.random.rand(N, 2) * 10
tri = Delaunay(positions)
adj = np.zeros((N, N))
for simplex in tri.simplices:
    for i in range(3):
        for j in range(i + 1, 3):
            K_val = np.random.uniform(1, 5)
            adj[simplex[i], simplex[j]] = K_val
            adj[simplex[j], simplex[i]] = K_val


# Swing equation: d(delta)/dt = omega, M*d(omega)/dt = P - D*omega - sum K sin(delta_i - delta_j)
def swing_rhs(t, y):
    delta = y[:N]
    omega = y[N:]
    ddelta = omega
    domega = np.zeros(N)
    for i in range(N):
        coupling = np.sum(adj[i] * np.sin(delta[i] - delta))
        domega[i] = (P[i] - D[i] * omega[i] - coupling) / M[i]
    return np.concatenate([ddelta, domega])


delta0 = np.random.uniform(-0.1, 0.1, N)
omega0 = np.zeros(N)
y0 = np.concatenate([delta0, omega0])

sol = solve_ivp(swing_rhs, (0, 30), y0, t_eval=np.linspace(20, 30, 500), method="RK45", rtol=1e-6)

freq_deviations = sol.y[N:, :]  # omega(t) for each generator
max_freq_dev = np.max(np.abs(freq_deviations[:, -100:]))
mean_freq_dev = np.mean(np.abs(freq_deviations[:, -100:]))

r_grid = [order_param(sol.y[:N, i])[0] for i in range(sol.y.shape[1])]
r_mean = np.mean(r_grid[-100:])

add_finding(
    "GRID_SYNC",
    "Power grid steady-state synchronisation",
    {
        "r_phase_coherence": round(float(r_mean), 4),
        "max_freq_deviation_rad_s": round(float(max_freq_dev), 6),
        "max_freq_deviation_mHz": round(float(max_freq_dev * 1000 / (2 * np.pi)), 3),
        "N_generators": N,
        "mean_inertia_s": round(float(np.mean(M)), 2),
        "mean_K_line": round(float(np.mean(adj[adj > 0])), 3),
    },
)

# --- Test 2: Renewable intermittency increases desync risk ---
print("\n=== Test 2: Renewable fluctuations → frequency instability ===")

noise_levels = [0, 0.1, 0.3, 0.5, 1.0, 2.0]
freq_instability = []

for noise_amp in noise_levels:

    def swing_noisy(t, y, noise=noise_amp):
        delta = y[:N]
        omega = y[N:]
        ddelta = omega
        P_noisy = P + noise * np.random.randn(N) * np.sqrt(0.01)  # Ornstein-Uhlenbeck-like
        domega = np.zeros(N)
        for i in range(N):
            coupling = np.sum(adj[i] * np.sin(delta[i] - delta))
            domega[i] = (P_noisy[i] - D[i] * omega[i] - coupling) / M[i]
        return np.concatenate([ddelta, domega])

    # Euler-Maruyama for stochastic
    y_run = y0.copy()
    dt = 0.01
    r_trace = []
    for step in range(3000):
        dy = swing_noisy(0, y_run, noise_amp)
        y_run += dt * dy
        if step % 30 == 0:
            r, _ = order_param(y_run[:N])
            r_trace.append(r)

    freq_instability.append(
        {
            "noise_amplitude": noise_amp,
            "r_mean": round(float(np.mean(r_trace[-20:])), 4),
            "r_std": round(float(np.std(r_trace[-20:])), 4),
        }
    )

add_finding(
    "GRID_RENEWABLES",
    "Renewable noise degrades grid sync",
    {
        "results": freq_instability,
        "prediction": "r drops with noise; blackout threshold at noise ~ K_c",
    },
)

# --- Test 3: Cascading failure (line tripping) ---
print("\n=== Test 3: Cascading failure — progressive line loss ===")

line_fractions = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]
cascade_results = []

for frac in line_fractions:
    adj_degraded = adj.copy()
    # Randomly remove lines
    for i in range(N):
        for j in range(i + 1, N):
            if adj_degraded[i, j] > 0 and np.random.rand() > frac:
                adj_degraded[i, j] = 0
                adj_degraded[j, i] = 0

    def swing_cascade(t, y, _adj=adj_degraded):
        delta = y[:N]
        omega = y[N:]
        ddelta = omega
        domega = np.zeros(N)
        for i in range(N):
            coupling = np.sum(_adj[i] * np.sin(delta[i] - delta))
            domega[i] = (P[i] - D[i] * omega[i] - coupling) / M[i]
        return np.concatenate([ddelta, domega])

    y_c = y0.copy()
    dt = 0.01
    r_c = []
    for step in range(2000):
        dy = swing_cascade(0, y_c)
        y_c += dt * dy
        if step % 20 == 0:
            r, _ = order_param(y_c[:N])
            r_c.append(r)

    cascade_results.append(
        {
            "line_fraction": frac,
            "r_final": round(float(np.mean(r_c[-10:])), 4),
            "desync": np.mean(r_c[-10:]) < 0.3,
        }
    )

add_finding(
    "GRID_CASCADE",
    "Cascading failure desynchronisation",
    {
        "results": cascade_results,
        "blackout_threshold": "sharp transition expected at ~30-50% line loss",
    },
)

# --- Test 4: Inertia matters (battery vs turbine) ---
print("\n=== Test 4: Low-inertia grid (battery/solar replacing turbines) ===")

M_low = M * 0.2  # batteries have ~5x less inertia than turbines


def swing_low_inertia(t, y):
    delta = y[:N]
    omega = y[N:]
    ddelta = omega
    domega = np.zeros(N)
    for i in range(N):
        coupling = np.sum(adj[i] * np.sin(delta[i] - delta))
        domega[i] = (P[i] - D[i] * omega[i] - coupling) / M_low[i]
    return np.concatenate([ddelta, domega])


# Step disturbance: sudden 20% load change
y_step = y0.copy()
dt = 0.01
freq_high_inertia = []
freq_low_inertia = []

# High inertia response
y_h = y0.copy()
for step in range(2000):
    P_disturbed = P.copy()
    if step > 500:
        P_disturbed[0] -= 2.0  # large generator trips

    def swing_h(t, y, _P=P_disturbed):
        delta = y[:N]
        omega = y[N:]
        ddelta = omega
        domega = np.zeros(N)
        for i in range(N):
            coupling = np.sum(adj[i] * np.sin(delta[i] - delta))
            domega[i] = (_P[i] - D[i] * omega[i] - coupling) / M[i]
        return np.concatenate([ddelta, domega])

    dy = swing_h(0, y_h)
    y_h += dt * dy
    if step % 10 == 0:
        freq_high_inertia.append(np.max(np.abs(y_h[N:])))

# Low inertia response
y_l = y0.copy()
for step in range(2000):
    P_disturbed = P.copy()
    if step > 500:
        P_disturbed[0] -= 2.0

    def swing_l(t, y, _P=P_disturbed):
        delta = y[:N]
        omega = y[N:]
        ddelta = omega
        domega = np.zeros(N)
        for i in range(N):
            coupling = np.sum(adj[i] * np.sin(delta[i] - delta))
            domega[i] = (_P[i] - D[i] * omega[i] - coupling) / M_low[i]
        return np.concatenate([ddelta, domega])

    dy = swing_l(0, y_l)
    y_l += dt * dy
    if step % 10 == 0:
        freq_low_inertia.append(np.max(np.abs(y_l[N:])))

add_finding(
    "GRID_INERTIA",
    "Low-inertia grid more vulnerable to disturbance",
    {
        "max_freq_dev_high_inertia": round(float(max(freq_high_inertia)), 4),
        "max_freq_dev_low_inertia": round(float(max(freq_low_inertia)), 4),
        "vulnerability_ratio": round(
            float(max(freq_low_inertia) / max(max(freq_high_inertia), 1e-6)), 2
        ),
        "note": "grid with 5x less inertia responds ~Nx faster to disturbance",
    },
)

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "power_grid_sync", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
