# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Huygens' Metronomes: The First Sync Observation
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


# --- Test 1: Huygens' original 2-pendulum setup ---
print("=== Test 1: Huygens' two clocks on a beam ===")

g = 9.81
L = 0.25  # 25 cm pendulum
m = 0.1  # 100 g
M_p = 1.0  # 1 kg platform
b = 0.01  # pendulum damping
c = 0.1  # platform damping

omega_nat = np.sqrt(g / L)
T_nat = 2 * np.pi / omega_nat


# Full nonlinear model for 2 pendula on platform
def huygens_2(t, y):
    # y = [theta1, theta2, dtheta1, dtheta2, X, dX]
    th1, th2, dth1, dth2, X, dX = y
    cos1, cos2 = np.cos(th1), np.cos(th2)
    sin1, sin2 = np.sin(th1), np.sin(th2)

    # Platform acceleration from pendula reaction forces
    # Simplified: small angle → cos ≈ 1
    ddX = (
        -m * L * ((-g / L) * sin1 - b * dth1) * cos1
        - m * L * ((-g / L) * sin2 - b * dth2) * cos2
        - c * dX
    ) / (M_p + 2 * m * (1 - 0.5 * (sin1**2 + sin2**2)))

    ddth1 = -(g / L) * sin1 - (ddX / L) * cos1 - (b / (m * L)) * dth1
    ddth2 = -(g / L) * sin2 - (ddX / L) * cos2 - (b / (m * L)) * dth2

    return [dth1, dth2, ddth1, ddth2, dX, ddX]


# Start with different phases
y0_2 = [0.15, -0.05, 0, 0, 0, 0]  # ~60° apart
sol2 = solve_ivp(
    huygens_2, (0, 60), y0_2, method="RK45", rtol=1e-8, t_eval=np.linspace(50, 60, 2000)
)

# Phase difference between the two pendula
phase1 = np.arctan2(sol2.y[2], sol2.y[0] * omega_nat)  # (dtheta, omega*theta)
phase2 = np.arctan2(sol2.y[3], sol2.y[1] * omega_nat)
phase_diff = (phase1 - phase2 + np.pi) % (2 * np.pi) - np.pi
mean_phase_diff = np.mean(phase_diff[-500:])

# Check: anti-phase (pi) or in-phase (0)?
is_antiphase = abs(abs(mean_phase_diff) - np.pi) < 0.5

add_finding(
    "HUYGENS_2CLOCK",
    "Huygens' two clocks sync result",
    {
        "mean_phase_diff_rad": round(float(mean_phase_diff), 4),
        "mean_phase_diff_deg": round(float(np.degrees(mean_phase_diff)), 1),
        "sync_type": "anti-phase" if is_antiphase else "in-phase",
        "matches_huygens_1665": is_antiphase,
        "pendulum_period_s": round(float(T_nat), 4),
        "platform_mass_ratio": M_p / (2 * m),
    },
)

# --- Test 2: N metronomes on rolling platform ---
print("\n=== Test 2: N metronomes — sync transition ===")

N_values = [2, 5, 10, 20, 50]
sync_results_N = []

for N_met in N_values:
    # Reduced Kuramoto model (valid for small amplitude)
    # Effective K = m * L * omega^2 / (M_p + N*m)
    K_eff = m * L * omega_nat**2 / (M_p + N_met * m)

    np.random.seed(42)
    omegas_met = np.random.normal(omega_nat, omega_nat * 0.02, N_met)  # 2% spread
    theta_met = np.random.uniform(0, 2 * np.pi, N_met)

    dt = 0.001
    for _step in range(int(30 / dt)):
        z = np.mean(np.exp(1j * theta_met))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_met + K_eff * r * np.sin(psi - theta_met)
        theta_met += dt * dtheta

    r_met, _ = order_param(theta_met)
    sync_results_N.append(
        {
            "N": N_met,
            "K_eff": round(float(K_eff), 6),
            "r_final": round(float(r_met), 4),
            "synced": float(r_met) > 0.8,
        }
    )

add_finding(
    "METRONOME_N",
    "Sync vs number of metronomes",
    {
        "results": sync_results_N,
        "K_scaling": "K_eff ~ 1/(M_p + N*m) — more metronomes = less coupling per pair",
        "prediction": "larger N needs lighter platform",
    },
)

# --- Test 3: Platform mass controls K ---
print("\n=== Test 3: Platform mass as coupling controller ===")

N_fixed = 10
M_p_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
mass_results = []

for M_val in M_p_values:
    K_eff = m * L * omega_nat**2 / (M_val + N_fixed * m)
    np.random.seed(42)
    omegas_m = np.random.normal(omega_nat, omega_nat * 0.02, N_fixed)
    theta_m = np.random.uniform(0, 2 * np.pi, N_fixed)

    dt = 0.001
    for _step in range(int(30 / dt)):
        z = np.mean(np.exp(1j * theta_m))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_m + K_eff * r * np.sin(psi - theta_m)
        theta_m += dt * dtheta

    r_m, _ = order_param(theta_m)
    mass_results.append(
        {
            "M_platform_kg": M_val,
            "K_eff": round(float(K_eff), 6),
            "r_final": round(float(r_m), 4),
        }
    )

add_finding(
    "METRONOME_MASS",
    "Platform mass controls coupling strength",
    {
        "results": mass_results,
        "SCPN_analogy": "platform mass = coupling medium impedance. Heavy beam = weak K.",
    },
)

# --- Test 4: In-phase vs anti-phase selection ---
print("\n=== Test 4: Phase selection mechanism ===")

# In-phase minimises platform kinetic energy
# Anti-phase minimises total energy for fixed platform

# Free platform (rolling): in-phase expected
np.random.seed(42)
theta_ip = np.random.uniform(0, 2 * np.pi, 10)
omegas_ip = np.random.normal(omega_nat, omega_nat * 0.01, 10)
K_ip = 0.5
dt = 0.001

for _step in range(int(30 / dt)):
    z = np.mean(np.exp(1j * theta_ip))
    r = np.abs(z)
    psi = np.angle(z)
    dtheta = omegas_ip + K_ip * r * np.sin(psi - theta_ip)
    theta_ip += dt * dtheta

r_ip, _ = order_param(theta_ip)

# With second harmonic coupling (anti-phase preference)
theta_ap = np.random.uniform(0, 2 * np.pi, 10)
for _step in range(int(30 / dt)):
    z2 = np.mean(np.exp(2j * theta_ap))  # second harmonic order parameter
    r2 = np.abs(z2)
    psi2 = np.angle(z2) / 2
    dtheta = omegas_ip + K_ip * r2 * np.sin(2 * (psi2 - theta_ap))
    theta_ap += dt * dtheta

# Check if pairs are anti-phase
phase_diffs_ap = []
sorted_phases = np.sort(theta_ap % (2 * np.pi))
for i in range(0, len(sorted_phases) - 1, 2):
    diff = abs(sorted_phases[i + 1] - sorted_phases[i])
    phase_diffs_ap.append(min(diff, 2 * np.pi - diff))

add_finding(
    "PHASE_SELECTION",
    "In-phase vs anti-phase selection",
    {
        "r_inphase_coupling": round(float(r_ip), 4),
        "mean_pair_diff_antiphase_rad": round(float(np.mean(phase_diffs_ap)), 4)
        if phase_diffs_ap
        else None,
        "huygens_was_right": "anti-phase for fixed beam (his setup)",
        "modern_demo": "in-phase for rolling platform (YouTube demos)",
    },
)

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "huygens_metronomes", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
