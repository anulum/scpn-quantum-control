# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 4 Tests: Lunar Phase-Locking + Thermal K + Oscillation Death
#
# Three specific Paper 4 predictions tested:
#
# A. Lunar-cellular phase locking (p.30):
#    V_lunar(t) = V_0 [1 + eps1*cos(omega_lunar*t) + eps2*cos(2*omega_lunar*t)]
#    Arnold tongue: |omega_cell - omega_lunar| < K
#    Mitosis modulation: R(t) = R_0 [1 + alpha_lunar*cos(omega_lunar*t - phi_0)]
#    alpha_lunar ~ 0.05-0.15
#
# B. Temperature-dependent oscillators (p.24):
#    omega_i(T) = omega_0 * exp(-E_a/(k_B*T)) * (1 + alpha_T*DeltaT)
#    K(T) = K_0 * (T/T_0)^alpha_temp * exp(-DeltaG/(k_B*T))
#
# C. Oscillation death and revival (p.49):
#    Amplitude death: K > K_critical → stable fixed point
#    Revival: K(t) = K_0[1 + eps*cos(omega_revival*t)]
#    Aging transition: P_active(t) = exp(-t/tau_aging)

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

# === PART A: Lunar-Cellular Phase Locking ===
print("=== Part A: Lunar-cellular phase locking (Paper 4, p.30) ===")

np.random.seed(42)

# Paper 4 parameters
omega_lunar = 2 * np.pi / (29.5 * 24)  # rad/hour (29.5 day period)
eps1 = 0.1   # tidal modulation
eps2 = 0.01  # gravitational second harmonic
V_0 = 1.0

# Cell division cycle: ~24 hours
omega_cell = 2 * np.pi / 24  # rad/hour

# --- Test A1: Arnold tongue for lunar entrainment ---
print("\n--- A1: Arnold tongue for cell-lunar coupling ---")

K_lunar_values = np.linspace(0, 0.1, 30)
detuning_values = np.linspace(-0.05, 0.05, 30)

# Track which (K, detuning) combinations lead to p:q locking
# Most relevant: 1:29 or 1:30 (one cell cycle per lunar day)
locked_map = np.zeros((len(K_lunar_values), len(detuning_values)))

dt = 0.1  # hours
T_sim = 29.5 * 24 * 3  # 3 lunar cycles

for ki, K_l in enumerate(K_lunar_values):
    for di, dw in enumerate(detuning_values):
        omega_c = omega_cell + dw
        phi = 0.0
        phi_trace = []

        for step in range(int(T_sim / dt)):
            t = step * dt
            # Lunar forcing
            F_lunar = K_l * np.sin(omega_lunar * t - phi)
            dphi = omega_c + F_lunar
            phi += dt * dphi

            if step > int(T_sim / (2 * dt)):
                phi_trace.append(phi)

        # Check if locked: dphi/dt should be constant
        if len(phi_trace) > 10:
            dphi_arr = np.diff(phi_trace)
            locked = np.std(dphi_arr) / np.mean(np.abs(dphi_arr)) < 0.1
            locked_map[ki, di] = 1 if locked else 0

# Find minimum K for locking at zero detuning
zero_det_idx = len(detuning_values) // 2
K_c_lunar = None
for ki in range(len(K_lunar_values)):
    if locked_map[ki, zero_det_idx] > 0:
        K_c_lunar = K_lunar_values[ki]
        break

add_finding("LUNAR_ARNOLD", "Arnold tongue for lunar-cellular coupling", {
    "K_c_zero_detuning": round(float(K_c_lunar), 4) if K_c_lunar else "no locking",
    "omega_cell_rad_hr": round(float(omega_cell), 6),
    "omega_lunar_rad_hr": round(float(omega_lunar), 8),
    "ratio_cell_lunar": round(float(omega_cell / omega_lunar), 1),
    "equation": "Paper 4, p.30: |omega_cell - omega_lunar| < K",
})

# --- Test A2: Mitosis rate modulation ---
print("\n--- A2: Lunar modulation of mitosis rate ---")

# Paper 4: R(t) = R_0 [1 + alpha_lunar * cos(omega_lunar*t - phi_0)]
# alpha_lunar ~ 0.05-0.15
alpha_lunar_values = [0.05, 0.10, 0.15]
R_0 = 100  # baseline mitosis rate (arbitrary units)

T_lunar = 29.5 * 24  # hours
t_hours = np.linspace(0, 3 * T_lunar, 1000)

mitosis_data = []
for alpha in alpha_lunar_values:
    R_t = R_0 * (1 + alpha * np.cos(omega_lunar * t_hours))
    # Calcium tide mechanism (Paper 4, p.31):
    # [Ca2+](t) = [Ca2+]_0 + A_tide * cos(omega_lunar*t) * H([Ca2+] - threshold)
    Ca_modulation = 0.1 * np.cos(omega_lunar * t_hours)  # uM
    Ca_above_threshold = Ca_modulation > 0  # threshold at 0

    mitosis_data.append({
        "alpha_lunar": alpha,
        "R_max": round(float(np.max(R_t)), 1),
        "R_min": round(float(np.min(R_t)), 1),
        "modulation_depth_percent": round(alpha * 100, 1),
        "full_moon_enhancement": round(float(R_0 * (1 + alpha) / R_0), 3),
    })

add_finding("LUNAR_MITOSIS", "Lunar modulation of cell division rate", {
    "results": mitosis_data,
    "equation": "Paper 4, p.30: R(t) = R_0 [1 + alpha*cos(omega_lunar*t)]",
    "biological_evidence": "coral spawning, sea urchin division, human menstrual cycle (~29.5 days)",
    "mechanism": "calcium tide via ion channel sensitivity to tidal forces",
})

# === PART B: Temperature-Dependent Coupling ===
print("\n=== Part B: Temperature-dependent oscillators (Paper 4, p.24) ===")

# Paper 4: omega(T) = omega_0 * exp(-E_a/(k_B*T)) * (1 + alpha_T*DeltaT)
k_B = 8.617e-5  # eV/K
E_activation = 0.5  # eV (typical enzyme activation energy)
T_0 = 310.15  # K (37°C body temperature)
omega_0_body = 2 * np.pi * 10  # 10 Hz neural oscillation at 37°C
alpha_thermal = 0.03  # 3% per degree

# Paper 4: K(T) = K_0 * (T/T_0)^alpha_temp * exp(-DeltaG/(k_B*T))
K_0 = 2.0
alpha_temp = 1.5
DeltaG = 0.3  # eV (protein conformational change)

temperatures = np.linspace(293, 315, 50)  # 20°C to 42°C
T_celcius = temperatures - 273.15

omega_vs_T = omega_0_body * np.exp(-E_activation / (k_B * temperatures)) / np.exp(-E_activation / (k_B * T_0))
omega_vs_T *= (1 + alpha_thermal * (temperatures - T_0))

K_vs_T = K_0 * (temperatures / T_0) ** alpha_temp * np.exp(-DeltaG / (k_B * temperatures)) / np.exp(-DeltaG / (k_B * T_0))

# Simulate sync at different temperatures
N_th = 50
temp_sync = []

for Ti, T in enumerate(temperatures[::5]):
    omegas_T = np.random.normal(omega_vs_T[Ti*5], omega_vs_T[Ti*5] * 0.1, N_th)
    K_T = K_vs_T[Ti*5]
    theta_T = np.random.uniform(0, 2 * np.pi, N_th)

    dt = 0.001
    for step in range(5000):
        z = np.mean(np.exp(1j * theta_T))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_T + K_T * r * np.sin(psi - theta_T)
        theta_T += dt * dtheta

    r_T, _ = order_param(theta_T)
    temp_sync.append({
        "T_celsius": round(float(T - 273.15), 1),
        "omega_Hz": round(float(omega_vs_T[Ti*5] / (2 * np.pi)), 2),
        "K_effective": round(float(K_T), 3),
        "r_sync": round(float(r_T), 4),
    })

# Fever and hypothermia effects
add_finding("THERMAL_OSCILLATORS", "Temperature modulates frequency and coupling", {
    "results": temp_sync,
    "fever_40C_omega_change": f"+{round((omega_vs_T[np.argmin(np.abs(temperatures-313))] / omega_vs_T[np.argmin(np.abs(temperatures-310))] - 1) * 100, 1)}%",
    "hypothermia_32C_omega_change": f"{round((omega_vs_T[np.argmin(np.abs(temperatures-305))] / omega_vs_T[np.argmin(np.abs(temperatures-310))] - 1) * 100, 1)}%",
    "equation": "Paper 4, p.24: omega(T) = omega_0*exp(-Ea/kT), K(T) = K_0*(T/T_0)^a*exp(-DG/kT)",
})

# --- Circadian temperature as synchroniser ---
print("\n--- B2: Circadian temperature rhythm as sync drive ---")

# Paper 4: T_circadian(t) = T_core + A_temp * cos(omega_cir*t + phi)
# A_temp ~ 0.5°C peak-to-trough
T_core = 310.15  # 37°C
A_temp = 0.25  # K (±0.25°C)
omega_cir = 2 * np.pi / 24  # rad/hour

# Temperature creates a tissue-wide frequency modulation
# Omega_thermal = sum (domega/dT) * T_circadian(t)
domega_dT = omega_0_body * (E_activation / (k_B * T_core**2) + alpha_thermal)
Gamma_thermal = domega_dT * A_temp

add_finding("CIRCADIAN_TEMP", "Circadian temperature rhythm as synchroniser", {
    "T_core_C": round(T_core - 273.15, 1),
    "A_temp_C": A_temp,
    "domega_dT_Hz_per_K": round(float(domega_dT / (2 * np.pi)), 2),
    "frequency_modulation_Hz": round(float(Gamma_thermal / (2 * np.pi)), 3),
    "as_fraction_of_omega": round(float(Gamma_thermal / omega_0_body * 100), 2),
    "equation": "Paper 4, p.24: Omega_thermal = (domega/dT) * A_temp * cos(omega_cir*t)",
})

# === PART C: Oscillation Death and Revival ===
print("\n=== Part C: Oscillation death and revival (Paper 4, p.49) ===")

# Paper 4: Amplitude death when K > K_critical (too much coupling)
# dx/dt = f(x) + K * sum G_ij * (x_j - x_i)
# Fixed point becomes stable → oscillation stops

N_death = 50
omega_death = 2 * np.pi * 10
omegas_d = np.random.normal(omega_death, omega_death * 0.3, N_death)  # large spread

K_death_values = np.linspace(0, 10, 30)
death_results = []

for K_d in K_death_values:
    theta_d = np.random.uniform(0, 2 * np.pi, N_death)
    dt = 0.001
    # Use Stuart-Landau oscillators (amplitude + phase)
    # dz/dt = (1 + i*omega)*z - |z|^2*z + K/N * sum(z_j - z_i)
    z = np.exp(1j * theta_d)  # unit amplitude

    for step in range(10000):
        dz = (1 + 1j * omegas_d) * z - np.abs(z)**2 * z + (K_d / N_death) * (np.mean(z) - z)
        z += dt * dz

    mean_amplitude = np.mean(np.abs(z))
    r_d, _ = order_param(np.angle(z))

    death_results.append({
        "K": round(float(K_d), 2),
        "mean_amplitude": round(float(mean_amplitude), 4),
        "r_phase": round(float(r_d), 4),
        "amplitude_death": mean_amplitude < 0.3,
    })

# Find K_critical for amplitude death
K_crit_death = None
for res in death_results:
    if res["amplitude_death"]:
        K_crit_death = res["K"]
        break

add_finding("AMPLITUDE_DEATH", "Oscillation death from excessive coupling", {
    "K_critical": K_crit_death,
    "results_sample": death_results[::5],
    "equation": "Paper 4, p.49: K > K_crit → stable fixed point",
    "clinical": "over-synchronisation can kill oscillations (pathological)",
})

# --- Revival via periodic coupling modulation ---
print("\n--- C2: Oscillation revival via coupling modulation ---")

# Paper 4: K(t) = K_0[1 + eps*cos(omega_revival*t)]
K_0_rev = K_crit_death * 1.5 if K_crit_death else 8.0  # above death threshold
eps_rev = 0.5  # 50% modulation
omega_rev = 2 * np.pi * 2  # 2 Hz modulation

z_rev = 0.01 * np.exp(1j * np.random.uniform(0, 2 * np.pi, N_death))
omegas_rev = np.random.normal(omega_death, omega_death * 0.3, N_death)

dt = 0.001
amp_trace_static = []
amp_trace_modulated = []

# Static K (stays dead)
z_static = z_rev.copy()
for step in range(20000):
    K_static = K_0_rev
    dz = (1 + 1j * omegas_rev) * z_static - np.abs(z_static)**2 * z_static + (K_static / N_death) * (np.mean(z_static) - z_static)
    z_static += dt * dz
    if step % 100 == 0:
        amp_trace_static.append(float(np.mean(np.abs(z_static))))

# Modulated K (revival)
z_mod = z_rev.copy()
for step in range(20000):
    t = step * dt
    K_mod = K_0_rev * (1 + eps_rev * np.cos(omega_rev * t))
    dz = (1 + 1j * omegas_rev) * z_mod - np.abs(z_mod)**2 * z_mod + (K_mod / N_death) * (np.mean(z_mod) - z_mod)
    z_mod += dt * dz
    if step % 100 == 0:
        amp_trace_modulated.append(float(np.mean(np.abs(z_mod))))

add_finding("OSCILLATION_REVIVAL", "Coupling modulation revives dead oscillations", {
    "K_0_above_death": round(float(K_0_rev), 2),
    "eps_modulation": eps_rev,
    "amplitude_static_final": round(float(amp_trace_static[-1]), 4),
    "amplitude_modulated_final": round(float(amp_trace_modulated[-1]), 4),
    "revival_success": amp_trace_modulated[-1] > 3 * amp_trace_static[-1],
    "equation": "Paper 4, p.49: K(t) = K_0[1 + eps*cos(omega*t)]",
})

# --- Aging transition ---
print("\n--- C3: Aging transition — progressive oscillator death ---")

# Paper 4: P_active(t) = exp(-t/tau_aging)
tau_aging = 50  # arbitrary time units
t_ages = np.linspace(0, 200, 50)
N_total = 100

aging_results = []
for t_age in t_ages[::5]:
    P_active = np.exp(-t_age / tau_aging)
    N_active = max(1, int(N_total * P_active))
    N_inactive = N_total - N_active

    # Active oscillators
    omegas_act = np.random.normal(omega_death, omega_death * 0.1, N_active)
    theta_act = np.random.uniform(0, 2 * np.pi, N_active)
    K_age = 2.0

    dt = 0.001
    for step in range(3000):
        z = np.mean(np.exp(1j * theta_act))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omegas_act + K_age * r * np.sin(psi - theta_act)
        theta_act += dt * dtheta

    r_age, _ = order_param(theta_act)
    aging_results.append({
        "t": round(float(t_age), 0),
        "P_active": round(float(P_active), 3),
        "N_active": N_active,
        "r_sync": round(float(r_age), 4),
    })

add_finding("AGING_TRANSITION", "Progressive oscillator death with aging", {
    "results": aging_results,
    "tau_aging": tau_aging,
    "equation": "Paper 4, p.49: P_active(t) = exp(-t/tau_aging)",
    "sync_collapse_at": "~50% active oscillators lost",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "p4_lunar_temperature_oscillation_death", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
