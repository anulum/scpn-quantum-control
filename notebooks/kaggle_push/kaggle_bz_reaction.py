# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Belousov-Zhabotinsky Reaction as Chemical Kuramoto
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


# --- Test 1: Oregonator model reduced to phase oscillator ---
# The Oregonator is the standard BZ model. Near limit cycle,
# reduce to phase oscillator via isochrons.

print("=== Test 1: BZ droplet array as Kuramoto ===")

N = 50  # 50 BZ droplets in a line
spacing_mm = 1.0
lambda_diff = 0.5  # mm diffusion length

# Natural frequencies (concentration variations)
np.random.seed(42)
omega_0 = 2 * np.pi / 60.0  # 60 s period
sigma = omega_0 * 0.1  # 10% variation
omegas = np.random.normal(omega_0, sigma, N)

# Distance-dependent coupling (1D chain)
positions = np.arange(N) * spacing_mm
distances = np.abs(positions[:, None] - positions[None, :])
K_matrix = 0.5 * np.exp(-distances / lambda_diff)
np.fill_diagonal(K_matrix, 0)

theta0 = np.random.uniform(0, 2 * np.pi, N)


def bz_kuramoto(t, theta):
    dtheta = np.copy(omegas)
    for i in range(N):
        dtheta[i] += np.sum(K_matrix[i] * np.sin(theta - theta[i]))
    return dtheta


sol = solve_ivp(
    bz_kuramoto, (0, 600), theta0, t_eval=np.linspace(400, 600, 500), method="RK45", rtol=1e-6
)

r_trace = [order_param(sol.y[:, i])[0] for i in range(sol.y.shape[1])]
r_final = np.mean(r_trace[-100:])

# Local order (nearest neighbours only)
local_r = []
for i in range(1, N - 1):
    z = np.mean(np.exp(1j * sol.y[[i - 1, i, i + 1], -1]))
    local_r.append(np.abs(z))
local_r_mean = np.mean(local_r)

add_finding(
    "BZ_CHAIN",
    "BZ droplet chain synchronisation",
    {
        "global_r": round(float(r_final), 4),
        "local_r_mean": round(float(local_r_mean), 4),
        "N_droplets": N,
        "spacing_mm": spacing_mm,
        "diffusion_length_mm": lambda_diff,
        "period_s": 60,
        "pattern": "local sync with global phase gradient"
        if local_r_mean > 0.8 and r_final < 0.5
        else "global sync",
    },
)

# --- Test 2: 2D BZ spiral wave formation ---
print("\n=== Test 2: 2D BZ grid — spiral wave formation ===")

Nx, Ny = 20, 20
N2d = Nx * Ny
omegas_2d = np.random.normal(omega_0, sigma, N2d)
theta_2d = np.random.uniform(0, 2 * np.pi, N2d)

# 2D nearest-neighbour coupling (4-connected grid)
K_2d = 0.3
dt = 0.5
T = 600
steps = int(T / dt)

# Create a phase singularity (initial condition for spiral)
for iy in range(Ny):
    for ix in range(Nx):
        idx = iy * Nx + ix
        theta_2d[idx] = np.arctan2(iy - Ny / 2, ix - Nx / 2)  # angular initial condition

r_2d_trace = []
# Check for spiral wave by measuring spatial phase gradient
winding_numbers = []

for step in range(steps):
    dtheta = np.copy(omegas_2d)
    for iy in range(Ny):
        for ix in range(Nx):
            idx = iy * Nx + ix
            neighbours = []
            if ix > 0:
                neighbours.append(iy * Nx + (ix - 1))
            if ix < Nx - 1:
                neighbours.append(iy * Nx + (ix + 1))
            if iy > 0:
                neighbours.append((iy - 1) * Nx + ix)
            if iy < Ny - 1:
                neighbours.append((iy + 1) * Nx + ix)
            for j in neighbours:
                dtheta[idx] += K_2d * np.sin(theta_2d[j] - theta_2d[idx])
    theta_2d += dt * dtheta

    if step % 100 == 0:
        r, _ = order_param(theta_2d)
        r_2d_trace.append(r)

        # Winding number around centre
        cx, cy = Nx // 2, Ny // 2
        ring_indices = []
        for dx, dy in [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]:
            ring_indices.append((cy + dy) * Nx + (cx + dx))
        ring_phases = [theta_2d[i] for i in ring_indices]
        dphis = np.diff(ring_phases)
        dphis = (dphis + np.pi) % (2 * np.pi) - np.pi  # wrap
        winding = np.sum(dphis) / (2 * np.pi)
        winding_numbers.append(round(float(winding), 2))

has_spiral = any(abs(w) > 0.5 for w in winding_numbers[-10:])

add_finding(
    "BZ_SPIRAL",
    "2D BZ grid spiral wave detection",
    {
        "global_r_final": round(float(np.mean(r_2d_trace[-5:])), 4),
        "spiral_detected": has_spiral,
        "winding_number_final": winding_numbers[-1] if winding_numbers else None,
        "grid": f"{Nx}x{Ny}",
        "K_coupling": K_2d,
        "note": "low global r + non-zero winding = spiral wave",
    },
)

# --- Test 3: Chimera states in BZ arrays ---
# Abrams & Strogatz (2004): identical oscillators split into
# sync + async groups. Observed in BZ by Tinsley et al. (2012).

print("\n=== Test 3: Chimera states in BZ ring ===")

N_ring = 100
omegas_ring = np.ones(N_ring) * omega_0  # IDENTICAL frequencies (chimera needs this)
theta_ring = np.random.uniform(0, 2 * np.pi, N_ring)
# Add small perturbation to break symmetry
theta_ring[: N_ring // 2] += 0.01 * np.random.randn(N_ring // 2)

# Non-local coupling: each oscillator couples to R nearest neighbours
R = 35  # coupling range
K_chimera = 0.4
alpha = 1.46  # phase lag (critical for chimera formation)

dt = 0.5
r_chimera_trace = []
local_r_left = []
local_r_right = []

for step in range(2000):
    dtheta = np.copy(omegas_ring)
    for i in range(N_ring):
        coupling = 0
        for dr in range(-R, R + 1):
            if dr == 0:
                continue
            j = (i + dr) % N_ring
            coupling += np.sin(theta_ring[j] - theta_ring[i] - alpha)
        dtheta[i] += (K_chimera / (2 * R)) * coupling
    theta_ring += dt * dtheta

    if step % 20 == 0:
        # Measure local order for left and right halves
        r_l, _ = order_param(theta_ring[: N_ring // 2])
        r_r, _ = order_param(theta_ring[N_ring // 2 :])
        local_r_left.append(r_l)
        local_r_right.append(r_r)
        r_g, _ = order_param(theta_ring)
        r_chimera_trace.append(r_g)

chimera_asymmetry = abs(np.mean(local_r_left[-10:]) - np.mean(local_r_right[-10:]))

add_finding(
    "BZ_CHIMERA",
    "Chimera state detection in BZ ring",
    {
        "global_r": round(float(np.mean(r_chimera_trace[-10:])), 4),
        "r_left_half": round(float(np.mean(local_r_left[-10:])), 4),
        "r_right_half": round(float(np.mean(local_r_right[-10:])), 4),
        "chimera_asymmetry": round(float(chimera_asymmetry), 4),
        "chimera_detected": chimera_asymmetry > 0.2,
        "N": N_ring,
        "coupling_range_R": R,
        "phase_lag_alpha": alpha,
    },
)

# --- Test 4: Concentration → K mapping ---
print("\n=== Test 4: BZ chemistry to SCPN K_nm ===")

# Known BZ parameters:
# Diffusion coefficient of activator (HBrO2): D ~ 2e-5 cm^2/s
# Droplet radius: ~100 um
# Oil gap: ~200 um

D_cm2s = 2e-5
droplet_radius_cm = 0.01  # 100 um
gap_cm = 0.02  # 200 um

# Diffusion coupling: K ~ D / gap^2
K_diffusion = D_cm2s / gap_cm**2  # s^-1
# Compare to omega ~ 0.1 rad/s (60 s period)
K_over_omega = K_diffusion / (omega_0 / (2 * np.pi))

add_finding(
    "BZ_COUPLING_PHYSICS",
    "Physical K_nm from BZ diffusion",
    {
        "D_cm2_per_s": D_cm2s,
        "gap_um": 200,
        "K_diffusion_per_s": round(float(K_diffusion), 4),
        "K_over_omega": round(float(K_over_omega), 2),
        "sync_predicted": K_over_omega > 2,
        "note": "K/omega >> 1 means BZ droplets easily sync — consistent with experiments",
    },
)

# --- Output ---
print("\n" + "=" * 60)
print(f"TOTAL FINDINGS: {len(FINDINGS)}")
print("=" * 60)
output = {"notebook": "bz_reaction", "findings": FINDINGS}
print("\n--- JSON OUTPUT ---")
print(json.dumps(output, indent=2, default=str))
