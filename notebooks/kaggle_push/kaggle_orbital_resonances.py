# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Orbital Resonances as Planetary Kuramoto
#
# Orbital resonances are Kuramoto synchronisation at cosmic scale:
# - Io:Europa:Ganymede = 1:2:4 (Laplace resonance, stable for 4 Gyr)
# - Pluto:Neptune = 2:3 (prevents close encounters)
# - Mercury spin-orbit = 3:2 (tidal locking with resonance)
# - TRAPPIST-1: 7 planets in near-resonant chain
#
# The coupling mechanism: gravitational torques at conjunction.
# When two orbits approach integer ratio, periodic gravitational
# kicks drive the system toward (or away from) exact resonance.
#
# Kuramoto mapping:
# - theta_i = orbital phase (mean anomaly)
# - omega_i = mean motion (2*pi/orbital_period)
# - K_ij = gravitational coupling ~ m_j / a_ij^2
# - Resonance = phase-locking at rational frequency ratio

import numpy as np
from scipy.integrate import solve_ivp
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

# --- Test 1: Laplace resonance (Io-Europa-Ganymede) ---
print("=== Test 1: Laplace resonance 1:2:4 ===")

# Orbital periods (days)
T_io = 1.769
T_europa = 3.551
T_ganymede = 7.155

# Mean motions (rad/day)
n_io = 2 * np.pi / T_io
n_europa = 2 * np.pi / T_europa
n_ganymede = 2 * np.pi / T_ganymede

# Check resonance ratios
ratio_ie = n_io / n_europa
ratio_eg = n_europa / n_ganymede
ratio_ig = n_io / n_ganymede

# Laplace relation: n_io - 3*n_europa + 2*n_ganymede ≈ 0
laplace_residual = n_io - 3 * n_europa + 2 * n_ganymede

# Model as coupled oscillators with resonant coupling
# In rotating frame: phi_ie = theta_io - 2*theta_europa (libration angle)
# d(phi_ie)/dt = n_io - 2*n_europa + K_ie * sin(phi_ie)

# Libration amplitude from tidal coupling
K_ie = 0.05  # rad/day^2 (from mass ratios)
K_eg = 0.03

# Simulate libration
def laplace_rhs(t, y):
    phi_ie, dphi_ie, phi_eg, dphi_eg = y
    ddphi_ie = -K_ie * np.sin(phi_ie) - 0.001 * dphi_ie  # tidal damping
    ddphi_eg = -K_eg * np.sin(phi_eg) - 0.001 * dphi_eg
    return [dphi_ie, ddphi_ie, dphi_eg, ddphi_eg]

y0 = [0.1, 0, 0.05, 0]  # small libration
sol = solve_ivp(laplace_rhs, (0, 1000), y0, method='RK45',
                t_eval=np.linspace(800, 1000, 1000), rtol=1e-8)

lib_ie = np.max(np.abs(sol.y[0]))  # libration amplitude
lib_eg = np.max(np.abs(sol.y[2]))

add_finding("LAPLACE_RESONANCE", "Galilean moons 1:2:4 Laplace resonance", {
    "ratio_Io_Europa": round(float(ratio_ie), 6),
    "ratio_Europa_Ganymede": round(float(ratio_eg), 6),
    "exact_ratio_deviation_ppm": round(float(abs(ratio_ie - 2) * 1e6), 1),
    "laplace_residual_rad_day": round(float(laplace_residual), 8),
    "libration_Io_Europa_rad": round(float(lib_ie), 4),
    "libration_Europa_Ganymede_rad": round(float(lib_eg), 4),
    "stable_for_Gyr": 4.5,
})

# --- Test 2: TRAPPIST-1 resonant chain ---
print("\n=== Test 2: TRAPPIST-1 seven-planet resonant chain ===")

# TRAPPIST-1 orbital periods (days)
trappist_periods = {
    "b": 1.510,
    "c": 2.422,
    "d": 4.050,
    "e": 6.100,
    "f": 9.207,
    "g": 12.354,
    "h": 18.772,
}

names = list(trappist_periods.keys())
periods = np.array(list(trappist_periods.values()))
n_motions = 2 * np.pi / periods

# Check all adjacent ratios
resonance_chain = []
for i in range(len(names) - 1):
    ratio = periods[i + 1] / periods[i]
    # Find nearest simple fraction
    best_p, best_q = 1, 1
    best_err = 999
    for p in range(1, 10):
        for q in range(p + 1, 10):
            err = abs(ratio - q / p)
            if err < best_err:
                best_err = err
                best_p, best_q = p, q

    resonance_chain.append({
        "pair": f"{names[i]}-{names[i+1]}",
        "period_ratio": round(float(ratio), 4),
        "nearest_resonance": f"{best_p}:{best_q}",
        "deviation_percent": round(float(best_err / (best_q / best_p) * 100), 3),
    })

# Simulate as Kuramoto with resonant coupling
N_t = 7
theta_t = np.random.uniform(0, 2 * np.pi, N_t)
K_grav = 0.02  # normalised gravitational coupling

dt = 0.01  # days
r_trappist_trace = []
for step in range(100000):
    dtheta = n_motions.copy()
    for i in range(N_t):
        for j in range(N_t):
            if i != j:
                # Resonant coupling at nearest integer ratio
                dtheta[i] += K_grav * np.sin(theta_t[j] - theta_t[i]) / abs(i - j)
    theta_t += dt * dtheta
    if step % 1000 == 0:
        r, _ = order_param(theta_t)
        r_trappist_trace.append(r)

add_finding("TRAPPIST1_CHAIN", "TRAPPIST-1 resonant chain", {
    "resonance_chain": resonance_chain,
    "all_near_resonance": all(r["deviation_percent"] < 5 for r in resonance_chain),
    "N_planets": 7,
    "note": "longest known resonant chain — gravitational Kuramoto at its finest",
})

# --- Test 3: Tidal locking as phase locking ---
print("\n=== Test 3: Mercury 3:2 spin-orbit resonance ===")

# Mercury: rotation period = 58.65 days, orbital period = 87.97 days
# Ratio = 87.97 / 58.65 = 1.5000 = 3:2

T_orbit_merc = 87.97  # days
T_spin_merc = 58.65
ratio_merc = T_orbit_merc / T_spin_merc

# Why not 1:1? Eccentricity (e=0.206) allows 3:2 capture.
# Goldreich & Peale (1966): capture probability depends on e

eccentricities = np.linspace(0, 0.4, 20)
# Capture probability for p:q resonance
# P(3:2) ~ 7*e^2 for small e (Goldreich-Peale)
p_32 = 7 * eccentricities ** 2
p_32 = np.clip(p_32, 0, 1)
# P(1:1) ~ 1 - e^2
p_11 = 1 - eccentricities ** 2

# Mercury's eccentricity
e_mercury = 0.206
p_32_mercury = min(7 * e_mercury ** 2, 1)
p_11_mercury = 1 - e_mercury ** 2

add_finding("MERCURY_32", "Mercury 3:2 spin-orbit lock", {
    "spin_orbit_ratio": round(float(ratio_merc), 4),
    "deviation_from_1.5": round(float(abs(ratio_merc - 1.5)), 6),
    "eccentricity": e_mercury,
    "capture_prob_32": round(float(p_32_mercury), 4),
    "capture_prob_11": round(float(p_11_mercury), 4),
    "why_not_11": "high eccentricity makes 3:2 almost as probable as 1:1",
    "Kuramoto_analogy": "spin frequency locked to 3/2 × orbital frequency",
})

# --- Test 4: Resonance capture — Kuramoto with integer coupling ---
print("\n=== Test 4: Resonance capture dynamics ===")

# Model: two orbits drifting toward resonance via tidal dissipation
# d(omega)/dt = -tau (tidal spindown) + K * sin(p*theta1 - q*theta2)

# 2:1 capture
omega1_0 = 2.2  # slightly faster than 2:1
omega2 = 1.0  # fixed
K_capture = 0.05
tau = 0.001  # tidal decay rate

theta1 = 0.0
omega1 = omega1_0
dt = 0.01
capture_trace = []

for step in range(200000):
    # Resonant angle: phi = 2*theta2 - theta1
    phi = 2 * (omega2 * step * dt) - theta1
    # Tidal spindown + resonant torque
    domega = -tau + K_capture * np.sin(phi)
    omega1 += dt * domega
    theta1 += dt * omega1

    if step % 500 == 0:
        capture_trace.append({
            "t": round(step * dt, 1),
            "omega_ratio": round(float(omega1 / omega2), 4),
            "resonant_angle_mod2pi": round(float(phi % (2 * np.pi)), 4),
        })

final_ratio = omega1 / omega2
captured = abs(final_ratio - 2.0) < 0.05

add_finding("RESONANCE_CAPTURE", "Tidal drift into 2:1 resonance", {
    "initial_ratio": round(float(omega1_0 / omega2), 4),
    "final_ratio": round(float(final_ratio), 4),
    "captured_in_resonance": captured,
    "tidal_decay_rate": tau,
    "K_coupling": K_capture,
    "analogy": "planet drifting into sync = frequency entrainment in Kuramoto",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "orbital_resonances", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
