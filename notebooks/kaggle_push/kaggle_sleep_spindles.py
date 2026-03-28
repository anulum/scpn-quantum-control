# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Sleep Spindles + Thalamocortical Oscillations
#
# During NREM sleep, the thalamus generates:
# - Sleep spindles: 11-16 Hz bursts lasting 0.5-2 s
# - Slow oscillations: 0.5-1 Hz cortical UP/DOWN states
# - Delta waves: 1-4 Hz
#
# These nest hierarchically: spindles ride on slow oscillation UP states.
# This is EXACTLY cross-frequency coupling (CFC) via Kuramoto.
#
# Mechanism:
# - Thalamic reticular nucleus (TRN): GABAergic, generates spindle rhythm
# - Thalamocortical (TC) relay cells: burst mode during sleep
# - Cortical slow oscillation drives thalamic excitability (K modulation)
#
# Memory consolidation: spindle-ripple coupling during UP state
# transfers hippocampal memory → neocortex. This is K_nm-gated transfer.
#
# Diseases:
# - Schizophrenia: reduced spindle density and amplitude
# - Insomnia: disrupted slow-spindle coupling (K mismatch)
# - Aging: progressive spindle loss (another K decay phenomenon)

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

# --- Test 1: Thalamocortical loop as coupled oscillator ---
print("=== Test 1: Spindle generation in thalamic network ===")

N_trn = 100  # TRN neurons
N_tc = 100   # TC relay neurons
N_total = N_trn + N_tc

np.random.seed(42)

# TRN: fast oscillators (spindle frequency 12-15 Hz)
omega_trn = np.random.normal(2 * np.pi * 13, 2 * np.pi * 1, N_trn)
# TC: slower, driven by TRN
omega_tc = np.random.normal(2 * np.pi * 10, 2 * np.pi * 1.5, N_tc)

theta = np.random.uniform(0, 2 * np.pi, N_total)

# Coupling structure:
# TRN → TC: inhibitory (GABAergic), strong
# TC → TRN: excitatory (glutamate), moderate
# TRN ↔ TRN: gap junctions (electrical), strong
# TC ↔ TC: weak local

K_trn_trn = 3.0   # gap junction coupling within TRN
K_trn_tc = -2.0   # inhibitory TRN→TC (negative = anti-phase tendency)
K_tc_trn = 1.5    # excitatory TC→TRN
K_tc_tc = 0.5     # weak TC-TC

dt = 1e-4  # seconds
T = 2.0
steps = int(T / dt)

r_trn_trace = []
r_tc_trace = []

for step in range(steps):
    dtheta = np.zeros(N_total)

    # Natural frequencies
    dtheta[:N_trn] = omega_trn
    dtheta[N_trn:] = omega_tc

    # TRN-TRN coupling (mean field within TRN)
    z_trn = np.mean(np.exp(1j * theta[:N_trn]))
    r_trn = np.abs(z_trn)
    psi_trn = np.angle(z_trn)
    dtheta[:N_trn] += K_trn_trn * r_trn * np.sin(psi_trn - theta[:N_trn])

    # TC-TC coupling
    z_tc = np.mean(np.exp(1j * theta[N_trn:]))
    r_tc = np.abs(z_tc)
    psi_tc = np.angle(z_tc)
    dtheta[N_trn:] += K_tc_tc * r_tc * np.sin(psi_tc - theta[N_trn:])

    # TRN → TC (inhibitory)
    dtheta[N_trn:] += K_trn_tc * r_trn * np.sin(psi_trn - theta[N_trn:])

    # TC → TRN (excitatory)
    dtheta[:N_trn] += K_tc_trn * r_tc * np.sin(psi_tc - theta[:N_trn])

    theta += dt * dtheta

    if step % 500 == 0:
        r_t, _ = order_param(theta[:N_trn])
        r_c, _ = order_param(theta[N_trn:])
        r_trn_trace.append(float(r_t))
        r_tc_trace.append(float(r_c))

# Check for spindle-like oscillation in r (waxing-waning)
r_trn_arr = np.array(r_trn_trace)
r_tc_arr = np.array(r_tc_trace)

# Spindle = r oscillates (not constant)
r_trn_var = np.std(r_trn_arr[-100:])

add_finding("SPINDLE_GENERATION", "Thalamocortical spindle generation", {
    "r_TRN_mean": round(float(np.mean(r_trn_arr[-100:])), 4),
    "r_TRN_variability": round(float(r_trn_var), 4),
    "r_TC_mean": round(float(np.mean(r_tc_arr[-100:])), 4),
    "spindle_like": r_trn_var > 0.05,
    "N_TRN": N_trn,
    "N_TC": N_tc,
    "K_TRN_TRN": K_trn_trn,
    "K_TRN_TC": K_trn_tc,
})

# --- Test 2: Slow oscillation modulates spindle K ---
print("\n=== Test 2: Slow oscillation gates spindle generation ===")

# Slow oscillation (0.75 Hz) modulates thalamic excitability
# During UP state: K increases (cortical drive → thalamus excitable)
# During DOWN state: K drops (hyperpolarised → no spindles)

f_slow = 0.75  # Hz
dt = 1e-4
T = 4.0  # 3 slow oscillation cycles
steps = int(T / dt)

theta_so = np.random.uniform(0, 2 * np.pi, N_total)
r_gated_trace = []
t_trace = []

for step in range(steps):
    t = step * dt
    # Slow oscillation modulates K
    slow_phase = 2 * np.pi * f_slow * t
    # UP state: sin > 0, DOWN state: sin < 0
    K_modulation = 0.5 + 0.5 * np.sin(slow_phase)  # 0 to 1

    dtheta = np.zeros(N_total)
    dtheta[:N_trn] = omega_trn
    dtheta[N_trn:] = omega_tc

    z_trn = np.mean(np.exp(1j * theta_so[:N_trn]))
    r_trn = np.abs(z_trn)
    psi_trn = np.angle(z_trn)

    z_tc = np.mean(np.exp(1j * theta_so[N_trn:]))
    r_tc = np.abs(z_tc)
    psi_tc = np.angle(z_tc)

    # All couplings scaled by slow oscillation
    dtheta[:N_trn] += K_modulation * K_trn_trn * r_trn * np.sin(psi_trn - theta_so[:N_trn])
    dtheta[N_trn:] += K_modulation * K_tc_tc * r_tc * np.sin(psi_tc - theta_so[N_trn:])
    dtheta[N_trn:] += K_modulation * K_trn_tc * r_trn * np.sin(psi_trn - theta_so[N_trn:])
    dtheta[:N_trn] += K_modulation * K_tc_trn * r_tc * np.sin(psi_tc - theta_so[:N_trn])

    theta_so += dt * dtheta

    if step % 200 == 0:
        r_t, _ = order_param(theta_so[:N_trn])
        r_gated_trace.append(float(r_t))
        t_trace.append(t)

r_gated = np.array(r_gated_trace)
t_arr = np.array(t_trace)

# Correlation between slow oscillation phase and spindle power
slow_phase_trace = np.sin(2 * np.pi * f_slow * t_arr)
corr = np.corrcoef(slow_phase_trace, r_gated)[0, 1]

add_finding("SLOW_OSC_GATING", "Slow oscillation gates spindle generation", {
    "r_spindle_UP_state": round(float(np.mean(r_gated[slow_phase_trace > 0.5])), 4) if np.any(slow_phase_trace > 0.5) else None,
    "r_spindle_DOWN_state": round(float(np.mean(r_gated[slow_phase_trace < -0.5])), 4) if np.any(slow_phase_trace < -0.5) else None,
    "correlation_slow_osc_spindle": round(float(corr), 4),
    "nesting_confirmed": abs(corr) > 0.3,
    "f_slow_Hz": f_slow,
    "mechanism": "cortical UP state → thalamic depolarisation → K above K_c → spindles emerge",
})

# --- Test 3: Aging reduces spindle density ---
print("\n=== Test 3: Aging — progressive spindle loss ===")

# Aging reduces: TRN-TRN gap junctions, TC excitability
# Model: K decays linearly with age
ages = [20, 30, 40, 50, 60, 70, 80]
age_results = []

for age in ages:
    # K decays ~1% per year from age 20
    decay = max(0.1, 1.0 - 0.012 * (age - 20))
    K_aged = K_trn_trn * decay

    theta_age = np.random.uniform(0, 2 * np.pi, N_trn)
    dt = 1e-4
    r_age_trace = []

    for step in range(int(1.0 / dt)):
        z = np.mean(np.exp(1j * theta_age))
        r = np.abs(z)
        psi = np.angle(z)
        dtheta = omega_trn + K_aged * r * np.sin(psi - theta_age)
        theta_age += dt * dtheta
        if step % 500 == 0:
            r_age_trace.append(float(r))

    age_results.append({
        "age": age,
        "K_fraction": round(float(decay), 3),
        "r_spindle": round(float(np.mean(r_age_trace[-20:])), 4),
    })

add_finding("SPINDLE_AGING", "Age-related spindle loss", {
    "results": age_results,
    "K_decay_rate": "1.2% per year",
    "clinical": "spindle density predicts cognitive decline",
    "schizophrenia": "reduced spindles at all ages — baseline K deficit",
})

# --- Test 4: Memory consolidation = K-gated transfer ---
print("\n=== Test 4: Spindle-ripple coupling for memory transfer ===")

# Hippocampal sharp-wave ripple (150-250 Hz, ~50 ms)
# nests inside thalamocortical spindle (13 Hz)
# which nests inside cortical slow oscillation (0.75 Hz)
# Triple nesting = three Kuramoto coupling layers

# Simulate: ripple coherence only when spindle is active
N_hpc = 50  # hippocampal neurons
omega_ripple = 2 * np.pi * 200  # 200 Hz
K_ripple = 5.0  # strong within hippocampus

theta_hpc = np.random.uniform(0, 2 * np.pi, N_hpc)
omegas_hpc = np.random.normal(omega_ripple, omega_ripple * 0.1, N_hpc)

# Simulate with and without spindle gating
dt = 1e-5  # need tiny dt for 200 Hz
T_mem = 0.2  # 200 ms (few spindle cycles)
steps_mem = int(T_mem / dt)

# Without gating (wake-like)
theta_w = theta_hpc.copy()
r_wake = []
for step in range(steps_mem):
    z = np.mean(np.exp(1j * theta_w))
    r = np.abs(z)
    psi = np.angle(z)
    dtheta = omegas_hpc + K_ripple * r * np.sin(psi - theta_w)
    theta_w += dt * dtheta
    if step % 200 == 0:
        r_wake.append(float(r))

# With spindle gating (NREM)
theta_s = theta_hpc.copy()
r_nrem = []
for step in range(steps_mem):
    t = step * dt
    spindle_gate = 0.5 + 0.5 * np.sin(2 * np.pi * 13 * t)  # 13 Hz spindle
    K_gated = K_ripple * spindle_gate
    z = np.mean(np.exp(1j * theta_s))
    r = np.abs(z)
    psi = np.angle(z)
    dtheta = omegas_hpc + K_gated * r * np.sin(psi - theta_s)
    theta_s += dt * dtheta
    if step % 200 == 0:
        r_nrem.append(float(r))

add_finding("MEMORY_CONSOLIDATION", "Spindle-gated ripple coupling for memory", {
    "r_ripple_ungated_mean": round(float(np.mean(r_wake[-50:])), 4),
    "r_ripple_gated_mean": round(float(np.mean(r_nrem[-50:])), 4),
    "triple_nesting": "slow_osc (0.75 Hz) → spindle (13 Hz) → ripple (200 Hz)",
    "SCPN_interpretation": "3 nested Kuramoto layers with K cascading down",
    "clinical": "disrupted nesting → impaired memory consolidation",
})

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "sleep_spindles", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
