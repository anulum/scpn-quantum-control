# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Circadian SCN Master Clock as Kuramoto Network
#
# The suprachiasmatic nucleus (SCN) contains ~20,000 neurons.
# Each neuron has a molecular clock (CLOCK/BMAL1/PER/CRY feedback loop).
# Period: ~24.2 hours (slightly > 24h → needs daily light reset).
#
# Known biology:
# - 20,000 neurons with individual periods 22-26 hours
# - VIP (vasoactive intestinal peptide) = primary coupling molecule
# - VIP knockout mice: complete circadian desynchronisation
# - Light input: retinohypothalamic tract → glutamate → phase shift
# - Jet lag: external zeitgeber shifts faster than internal coupling allows
# - Shift work: chronic desync → 17% increased cancer risk (WHO IARC)
#
# This is Kuramoto with:
# - omega_i ~ 24.2 hr period, sigma ~ 1 hr
# - K_ij via VIP diffusion (paracrine, ~100 um range)
# - External forcing: light as periodic drive with phase shift

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

# --- Test 1: SCN as Kuramoto network ---
print("=== Test 1: SCN circadian synchronisation ===")

N = 500  # reduced from 20,000
omega_0 = 2 * np.pi / 24.2  # rad/hour (24.2 hour period)
sigma_omega = 2 * np.pi * (1.0 / 24.2**2)  # ~1 hour spread in period

np.random.seed(42)
# Period distribution: 22-26 hours
periods = np.random.normal(24.2, 1.0, N)
periods = np.clip(periods, 22, 26)
omegas = 2 * np.pi / periods  # rad/hour

theta0 = np.random.uniform(0, 2 * np.pi, N)

# Light forcing: 12h light / 12h dark
def light_phase_response(theta, t_hours):
    """Phase response to light: advances in early subjective night,
    delays in late subjective night. Type 1 PRC."""
    hour_of_day = t_hours % 24
    if 6 <= hour_of_day <= 18:  # light on
        return 0.02 * np.sin(theta)  # advance/delay depending on phase
    return 0.0

# Sweep K
K_values = np.linspace(0, 0.3, 25)
r_vs_K = []

for K in K_values:
    theta_run = theta0.copy()
    dt = 0.1  # hours
    for step in range(int(240 / dt)):  # 10 days
        t = step * dt
        dtheta = np.copy(omegas)
        # Kuramoto coupling
        z = np.mean(np.exp(1j * theta_run))
        r_inst = np.abs(z)
        psi = np.angle(z)
        dtheta += K * r_inst * np.sin(psi - theta_run)  # mean-field
        # Light forcing
        dtheta += light_phase_response(theta_run, t)
        theta_run += dt * dtheta

    r, _ = order_param(theta_run)
    r_vs_K.append(r)

r_vs_K = np.array(r_vs_K)
idx_sync = np.where(r_vs_K > 0.5)[0]
K_c = K_values[idx_sync[0]] if len(idx_sync) > 0 else float('nan')

add_finding("SCN_KC", "Critical VIP coupling for circadian sync", {
    "K_c_rad_per_hr": round(float(K_c), 4),
    "N_neurons": N,
    "mean_period_hr": 24.2,
    "period_spread_hr": 1.0,
    "r_at_max_K": round(float(r_vs_K[-1]), 4),
    "light_forcing": "12L:12D with Type 1 PRC",
})

# --- Test 2: VIP knockout (K → 0) ---
print("\n=== Test 2: VIP knockout — complete desynchronisation ===")

theta_vip = theta0.copy()
dt = 0.1
r_vip_trace = []
r_novip_trace = []

K_healthy = 0.2  # above K_c

# Healthy SCN
theta_h = theta0.copy()
for step in range(int(240 / dt)):
    t = step * dt
    z = np.mean(np.exp(1j * theta_h))
    dtheta = omegas + K_healthy * np.abs(z) * np.sin(np.angle(z) - theta_h)
    dtheta += light_phase_response(theta_h, t)
    theta_h += dt * dtheta
    if step % 24 == 0:  # sample every ~2.4 hours
        r, _ = order_param(theta_h)
        r_vip_trace.append(r)

# VIP knockout (K = 0)
theta_ko = theta0.copy()
for step in range(int(240 / dt)):
    t = step * dt
    dtheta = omegas + light_phase_response(theta_ko, t)
    theta_ko += dt * dtheta
    if step % 24 == 0:
        r, _ = order_param(theta_ko)
        r_novip_trace.append(r)

add_finding("VIP_KNOCKOUT", "VIP knockout desynchronises SCN", {
    "r_healthy_day10": round(float(r_vip_trace[-1]), 4),
    "r_knockout_day10": round(float(r_novip_trace[-1]), 4),
    "r_ratio": round(float(r_novip_trace[-1] / max(r_vip_trace[-1], 0.01)), 3),
    "matches_experiment": r_novip_trace[-1] < 0.3,
    "note": "VIP KO mice show desynchronised SCN — our model should reproduce this",
})

# --- Test 3: Jet lag recovery dynamics ---
print("\n=== Test 3: Jet lag — 8-hour eastward shift ===")

K_jetlag = 0.2
theta_jl = theta0.copy()
dt = 0.1

# First: sync to original timezone (5 days)
for step in range(int(120 / dt)):
    t = step * dt
    z = np.mean(np.exp(1j * theta_jl))
    dtheta = omegas + K_jetlag * np.abs(z) * np.sin(np.angle(z) - theta_jl)
    dtheta += light_phase_response(theta_jl, t)
    theta_jl += dt * dtheta

# Shift: advance light cycle by 8 hours (eastward travel)
phase_shift = 8.0  # hours
r_recovery = []
mean_phase_offset = []

for step in range(int(240 / dt)):  # 10 days recovery
    t = step * dt
    z = np.mean(np.exp(1j * theta_jl))
    dtheta = omegas + K_jetlag * np.abs(z) * np.sin(np.angle(z) - theta_jl)
    # Light now shifted by 8 hours
    dtheta += light_phase_response(theta_jl, t + phase_shift)
    theta_jl += dt * dtheta

    if step % int(24 / dt) == 0:  # once per day
        r, psi = order_param(theta_jl)
        r_recovery.append(r)
        # Expected phase for new timezone
        expected_phase = (omega_0 * (t + phase_shift)) % (2 * np.pi)
        phase_error = np.abs(((psi - expected_phase) + np.pi) % (2 * np.pi) - np.pi)
        mean_phase_offset.append(float(phase_error))

# Find recovery day (phase error < 0.5 rad)
recovery_day = None
for i, pe in enumerate(mean_phase_offset):
    if pe < 0.5:
        recovery_day = i
        break

add_finding("JET_LAG", "Jet lag recovery from 8-hour eastward shift", {
    "recovery_day": recovery_day,
    "phase_error_day1_rad": round(mean_phase_offset[0], 3) if mean_phase_offset else None,
    "phase_error_day10_rad": round(mean_phase_offset[-1], 3) if mean_phase_offset else None,
    "r_minimum": round(float(min(r_recovery)), 4) if r_recovery else None,
    "clinical_expectation": "~1 day per timezone crossed ≈ 8 days",
    "K_coupling": K_jetlag,
})

# --- Test 4: Shift work chronic desync ---
print("\n=== Test 4: Shift work — rotating schedule desync ===")

K_sw = 0.2
theta_sw = theta0.copy()
dt = 0.1
r_shift_work = []

# Rotating shift: light cycle shifts by 8 hours every 7 days
for week in range(8):  # 8 weeks
    shift = (week % 3) * 8  # 3-shift rotation: 0, 8, 16 hours
    for step in range(int(168 / dt)):  # 1 week = 168 hours
        t = week * 168 + step * dt
        z = np.mean(np.exp(1j * theta_sw))
        dtheta = omegas + K_sw * np.abs(z) * np.sin(np.angle(z) - theta_sw)
        dtheta += light_phase_response(theta_sw, t + shift)
        theta_sw += dt * dtheta

        if step % int(24 / dt) == 0:
            r, _ = order_param(theta_sw)
            r_shift_work.append({"week": week, "r": round(float(r), 4)})

r_chronic = [x["r"] for x in r_shift_work[-7:]]

add_finding("SHIFT_WORK", "Rotating shift work chronic desynchronisation", {
    "mean_r_final_week": round(float(np.mean(r_chronic)), 4),
    "r_range": [round(float(min(r_chronic)), 4), round(float(max(r_chronic)), 4)],
    "never_fully_syncs": np.mean(r_chronic) < 0.8,
    "cancer_link": "WHO IARC: shift work = probable carcinogen (Group 2A)",
    "mechanism": "chronic internal desync → immune suppression → cancer risk +17%",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "circadian_scn", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
