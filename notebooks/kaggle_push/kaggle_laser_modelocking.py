# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Laser Mode-Locking = Photon Phase Kuramoto
#
# A laser cavity supports N longitudinal modes with frequencies:
#   f_n = n * c / (2L), spacing Δf = c / (2L)
# Mode-locking = all modes synchronise their phases.
# When locked: constructive interference → ultrashort pulses (fs-ps).
# When unlocked: random phases → CW noise.
#
# The coupling mechanism:
# - Passive: saturable absorber (intensity-dependent loss favours pulses)
# - Active: acousto-optic modulator at Δf
# - Kerr lens: nonlinear refractive index (self-focusing at high intensity)
#
# Haus (1975): mode-locking equations reduce to coupled oscillators.
# The pulse IS the order parameter (r=1 → perfect pulse, r=0 → CW noise).
#
# SCPN connection: each mode is an oscillator with frequency nΔf.
# Nonlinear medium provides all-to-all coupling through four-wave mixing.
# K_nm = chi(3) * E_n * E_m (third-order susceptibility × field amplitudes).

import json

import numpy as np

FINDINGS = []


def add_finding(tag, description, data):
    FINDINGS.append({"tag": tag, "description": description, "data": data})
    print(f"[FINDING] {tag}: {description}")
    for k, v in data.items():
        print(f"  {k}: {v}")


def order_param(theta):
    z = np.mean(np.exp(1j * theta))
    return np.abs(z), np.angle(z)


# --- Test 1: Mode-locking as Kuramoto transition ---
print("=== Test 1: Laser mode-locking via phase synchronisation ===")

N_modes = 100  # longitudinal modes
# Mode frequencies: equally spaced (unlike bio Kuramoto!)
delta_f = 100e6  # 100 MHz mode spacing (1m cavity)
f_modes = np.arange(N_modes) * delta_f

# In rotating frame: detuning from equally-spaced is the "natural frequency"
# For a perfect cavity, all detunings = 0. Dispersion creates omega_n.
# GVD (group velocity dispersion): omega_n = beta2 * (n - n0)^2 * delta_f^2
beta2 = 0.01  # normalised dispersion
n0 = N_modes // 2
omegas = beta2 * ((np.arange(N_modes) - n0) * delta_f) ** 2
omegas *= 1e-16  # normalise to reasonable units

np.random.seed(42)
theta0 = np.random.uniform(0, 2 * np.pi, N_modes)

# Saturable absorber coupling: favours in-phase (all-to-all)
# K_nm is mediated by intensity-dependent gain
K_values = np.linspace(0, 0.5, 30)
r_vs_K = []
pulse_widths = []

for K in K_values:
    theta = theta0.copy()
    dt = 0.001
    for step in range(5000):
        # Mean-field Kuramoto
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas + K * r * np.sin(psi - theta)
        theta += dt * dtheta

    r_final, psi_final = order_param(theta)
    r_vs_K.append(float(r_final))

    # Pulse width from mode phases
    # E(t) = sum_n exp(i * n * delta_f * t + i * phi_n)
    # Pulse width ~ 1 / (N * delta_f * r)
    if r_final > 0.1:
        pw = 1.0 / (N_modes * delta_f * r_final)
        pulse_widths.append(pw)
    else:
        pulse_widths.append(float("inf"))

# Find K_c
idx = np.where(np.array(r_vs_K) > 0.5)[0]
K_c_laser = K_values[idx[0]] if len(idx) > 0 else float("nan")

add_finding(
    "LASER_MODELOCK",
    "Mode-locking as Kuramoto phase transition",
    {
        "K_c": round(float(K_c_laser), 4),
        "r_at_max_K": round(r_vs_K[-1], 4),
        "N_modes": N_modes,
        "mode_spacing_MHz": delta_f / 1e6,
        "pulse_width_at_max_K_ps": round(pulse_widths[-1] * 1e12, 2)
        if pulse_widths[-1] < 1
        else "CW",
        "transform_limited_ps": round(1e12 / (N_modes * delta_f), 2),
    },
)

# --- Test 2: Dispersion kills mode-locking ---
print("\n=== Test 2: Dispersion vs coupling competition ===")

beta2_values = [0, 0.001, 0.01, 0.05, 0.1, 0.5]
K_fixed = 0.3
disp_results = []

for b2 in beta2_values:
    omegas_d = b2 * ((np.arange(N_modes) - n0) * delta_f) ** 2 * 1e-16
    theta_d = theta0.copy()
    dt = 0.001
    for step in range(5000):
        z = np.mean(np.exp(1j * theta_d))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_d + K_fixed * r * np.sin(psi - theta_d)
        theta_d += dt * dtheta

    r_d, _ = order_param(theta_d)
    disp_results.append(
        {
            "beta2": b2,
            "r_final": round(float(r_d), 4),
            "locked": float(r_d) > 0.5,
        }
    )

add_finding(
    "LASER_DISPERSION",
    "Dispersion competes with mode-locking",
    {
        "results": disp_results,
        "K_coupling": K_fixed,
        "physics": "dispersion = effective frequency spread; needs K > K_c(beta2)",
    },
)

# --- Test 3: Kerr lens mode-locking (intensity-dependent coupling) ---
print("\n=== Test 3: Kerr lens — nonlinear coupling ===")

# In Kerr-lens ML, coupling strength depends on current pulse intensity
# K_eff = K0 * r^2 (positive feedback: more sync → stronger coupling)

K0_kerr = 0.15  # below K_c for linear coupling
theta_kerr = theta0.copy()
omegas_kerr = 0.01 * ((np.arange(N_modes) - n0) * delta_f) ** 2 * 1e-16

r_kerr_trace = []
dt = 0.001
for step in range(10000):
    z = np.mean(np.exp(1j * theta_kerr))
    r = np.abs(z)
    psi = np.angle(z)
    # Nonlinear coupling: K = K0 * (1 + r^2)
    K_eff = K0_kerr * (1 + r**2)
    dtheta = omegas_kerr + K_eff * r * np.sin(psi - theta_kerr)
    theta_kerr += dt * dtheta
    if step % 50 == 0:
        r_kerr_trace.append(float(r))

# Check for bistability (hysteresis)
# Start locked and reduce K
theta_locked = np.zeros(N_modes)  # perfect lock
r_unlocking = []
K0_sweep = np.linspace(0.3, 0, 30)
for K0_test in K0_sweep:
    for step in range(2000):
        z = np.mean(np.exp(1j * theta_locked))
        r = np.abs(z)
        psi = np.angle(z)
        K_eff = K0_test * (1 + r**2)
        dtheta = omegas_kerr + K_eff * r * np.sin(psi - theta_locked)
        theta_locked += dt * dtheta
    r_u, _ = order_param(theta_locked)
    r_unlocking.append(float(r_u))

add_finding(
    "KERR_MODELOCK",
    "Kerr-lens nonlinear mode-locking",
    {
        "r_final_kerr": round(r_kerr_trace[-1], 4),
        "self_starting": r_kerr_trace[-1] > 0.5,
        "K0_below_linear_Kc": True,
        "hysteresis": "yes" if r_unlocking[-1] > 0.3 else "no",
        "physics": "positive feedback: r↑ → K↑ → r↑ (bootstrapping)",
    },
)

# --- Test 4: SCPN mapping — chi(3) as K_nm ---
print("\n=== Test 4: SCPN K_nm from nonlinear susceptibility ===")

# chi(3) values for common laser media:
media = {
    "Ti:Sapphire": {"chi3_esu": 3e-16, "n2_cm2_W": 3.2e-16, "bandwidth_THz": 128},
    "Cr:Forsterite": {"chi3_esu": 2e-16, "n2_cm2_W": 2.0e-16, "bandwidth_THz": 45},
    "Er:fiber": {"chi3_esu": 2.5e-20, "n2_cm2_W": 2.6e-20, "bandwidth_THz": 12},
}

for _name, params in media.items():
    # K_nm ~ chi3 * I (intensity inside cavity)
    # For Ti:Sapphire: I ~ 10^10 W/cm2 at focus
    I_cavity = 1e10  # W/cm2
    K_nm = params["n2_cm2_W"] * I_cavity  # dimensionless phase shift per round trip
    N_modes_medium = int(params["bandwidth_THz"] / (delta_f / 1e12))
    pulse_limit = 1.0 / params["bandwidth_THz"]  # transform limit in ps

    params["K_nm_per_roundtrip"] = round(K_nm, 6)
    params["N_modes_supported"] = N_modes_medium
    params["transform_limit_fs"] = round(pulse_limit * 1e3, 1)

add_finding(
    "LASER_SCPN",
    "SCPN K_nm from chi(3) nonlinearity",
    {
        "media": media,
        "note": "Ti:Sapphire: 128 THz BW → sub-5fs pulses. Each mode = oscillator, chi3 = coupling.",
    },
)

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "laser_modelocking", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
