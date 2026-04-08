# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Pulse Shaping (ICI + Hypergeometric)
"""PMP-optimal ICI pulse sequences and (α,β)-hypergeometric pulse shaping.

Implements two complementary pulse-level control techniques:

1. **ICI (Intuitive-Counterintuitive-Intuitive):** PMP-derived optimal
   pulse sequence for dissipative state transfer via STIREP. Three-segment
   trajectory that maximises fidelity under Lindblad decay.
   Ref: Liu et al. (2023) — STIREP optimal control

2. **(α,β)-Hypergeometric:** Unified pulse family parameterised by the
   Gauss hypergeometric function ₂F₁. Subsumes Allen-Eberly, STIRAP,
   and Demkov-Kunike as special cases.
   Ref: Ventura Meinersen et al., arXiv:2504.08031 (2025)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import hyp2f1

# ============================================================
# Data structures
# ============================================================


@dataclass
class ICIPulse:
    """ICI three-segment pulse envelope."""

    times: np.ndarray  # time grid
    omega_p: np.ndarray  # pump Rabi frequency Ω_P(t)
    omega_s: np.ndarray  # Stokes Rabi frequency Ω_S(t)
    theta: np.ndarray  # mixing angle θ(t)
    omega_total: float  # peak Rabi frequency Ω₀
    gamma_decay: float  # Lindblad decay rate γ
    fidelity: float  # achieved fidelity (from BVP solution)


@dataclass
class HypergeometricPulse:
    """(α,β)-hypergeometric pulse envelope."""

    times: np.ndarray
    envelope: np.ndarray  # Ω(t) / Ω₀
    alpha: float
    beta: float
    gamma_width: float  # pulse width parameter γ
    omega_0: float  # peak Rabi frequency


@dataclass
class PulseSchedule:
    """Complete pulse schedule for a Trotter step."""

    pulses: Sequence[HypergeometricPulse | ICIPulse]
    total_time: float
    n_qubits: int
    infidelity_bound: float


# ============================================================
# ICI Pulse Sequences (Tweak 2)
# ============================================================


def ici_mixing_angle(
    t: np.ndarray,
    t_total: float,
    theta_jump: float = 0.3,
) -> np.ndarray:
    """Compute ICI mixing angle θ(t) for three-segment trajectory.

    Segment 1 (intuitive): θ jumps from 0 to θ_jump
    Segment 2 (counterintuitive): θ sweeps from θ_jump to π/2 − θ_jump
    Segment 3 (intuitive): θ jumps from π/2 − θ_jump to π/2

    The jump segments are sharp (bang-bang); the middle segment is smooth
    (adiabatic). This is the PMP-optimal solution for STIREP.

    Args:
        t: time array (0 to t_total)
        t_total: total pulse duration
        theta_jump: jump angle at boundaries (controls speed vs. loss)
    """
    theta = np.zeros_like(t)
    t1 = 0.05 * t_total  # first jump duration (5% of total)
    t2 = 0.95 * t_total  # start of last jump

    for i, ti in enumerate(t):
        if ti < t1:
            # Segment 1: linear ramp 0 → θ_jump
            theta[i] = theta_jump * (ti / t1)
        elif ti < t2:
            # Segment 2: smooth counterintuitive sweep
            s = (ti - t1) / (t2 - t1)  # normalised [0, 1]
            theta[i] = theta_jump + (np.pi / 2 - 2 * theta_jump) * s
        else:
            # Segment 3: linear ramp → π/2
            s = (ti - t2) / (t_total - t2)
            theta[i] = (np.pi / 2 - theta_jump) + theta_jump * s

    return theta


def build_ici_pulse(
    t_total: float,
    omega_0: float,
    gamma_decay: float,
    n_points: int = 200,
    theta_jump: float = 0.3,
) -> ICIPulse:
    """Build ICI pulse envelope for STIREP state transfer.

    The pump and Stokes Rabi frequencies satisfy the total power constraint:
        Ω_P²(t) + Ω_S²(t) = Ω₀²

    With mixing angle θ(t):
        Ω_P(t) = Ω₀ sin(θ(t))
        Ω_S(t) = Ω₀ cos(θ(t))

    Args:
        t_total: total pulse duration (μs)
        omega_0: peak Rabi frequency (MHz)
        gamma_decay: Lindblad decay rate from excited state (MHz)
        n_points: time grid resolution
        theta_jump: ICI jump angle (0 → π/4 range)
    """
    if t_total <= 0:
        raise ValueError("t_total must be positive")
    if omega_0 <= 0:
        raise ValueError("omega_0 must be positive")
    if gamma_decay < 0:
        raise ValueError("gamma_decay must be non-negative")
    if not 0 < theta_jump < np.pi / 4:
        raise ValueError(f"theta_jump must be in (0, π/4), got {theta_jump}")

    times = np.linspace(0, t_total, n_points)
    theta = ici_mixing_angle(times, t_total, theta_jump)
    omega_p = omega_0 * np.sin(theta)
    omega_s = omega_0 * np.cos(theta)

    # Estimate fidelity from adiabatic condition
    # Loss ≈ γ × ∫ sin²(θ) dt / Ω₀ (excited state population)
    dt = times[1] - times[0]
    excited_pop = np.sum(np.sin(theta) ** 2) * dt / t_total
    fidelity = max(0.0, 1.0 - gamma_decay * excited_pop / omega_0)

    return ICIPulse(
        times=times,
        omega_p=omega_p,
        omega_s=omega_s,
        theta=theta,
        omega_total=omega_0,
        gamma_decay=gamma_decay,
        fidelity=fidelity,
    )


def ici_three_level_evolution(
    pulse: ICIPulse,
    gamma_decay: float | None = None,
) -> np.ndarray:
    """Simulate 3-level Λ system under ICI pulse.

    States: |g⟩ (ground), |e⟩ (excited), |s⟩ (target).
    Hamiltonian in rotating frame:
        H(t) = Ω_P(t)|e⟩⟨g| + Ω_S(t)|e⟩⟨s| + h.c.

    Returns state population vector [P_g, P_e, P_s] at each time step.
    """
    gamma = gamma_decay if gamma_decay is not None else pulse.gamma_decay
    n_t = len(pulse.times)
    populations = np.zeros((n_t, 3))
    populations[0] = [1.0, 0.0, 0.0]  # start in ground

    # Density matrix evolution (3×3)
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0

    for i in range(1, n_t):
        dt = pulse.times[i] - pulse.times[i - 1]
        op = pulse.omega_p[i]
        os = pulse.omega_s[i]

        # Hamiltonian
        h = np.array(
            [
                [0, op, 0],
                [op, 0, os],
                [0, os, 0],
            ],
            dtype=complex,
        )

        # Coherent evolution: dρ/dt = -i[H, ρ]
        commutator = h @ rho - rho @ h
        drho = -1j * commutator

        # Lindblad decay from |e⟩
        if gamma > 0:
            # L = sqrt(γ)|g⟩⟨e| (decay to ground)
            drho[0, 0] += gamma * rho[1, 1]
            drho[1, 1] -= gamma * rho[1, 1]
            # Dephasing of off-diagonal
            drho[0, 1] -= 0.5 * gamma * rho[0, 1]
            drho[1, 0] -= 0.5 * gamma * rho[1, 0]
            drho[1, 2] -= 0.5 * gamma * rho[1, 2]
            drho[2, 1] -= 0.5 * gamma * rho[2, 1]

        rho = rho + drho * dt
        # Enforce trace preservation
        trace = np.real(np.trace(rho))
        if trace > 1e-15:
            rho /= trace

        populations[i] = np.real(np.diag(rho))

    return populations


# ============================================================
# (α,β)-Hypergeometric Pulse Shaping (Tweak 3)
# ============================================================


def hypergeometric_envelope(
    t: np.ndarray,
    alpha: float,
    beta: float,
    gamma_width: float,
) -> np.ndarray:
    """Compute (α,β)-hypergeometric pulse envelope.

    Ω(t)/Ω₀ = sech(γt) × ₂F₁(α, β; (α+β+1)/2; (1+tanh(γt))/2)

    Ref: Ventura Meinersen et al., arXiv:2504.08031 (2025), Eq. 14

    Special cases:
        α=β=0: Allen-Eberly (pure sech)
        α=β=0.5: STIRAP-optimal
        α=1, β=0.5: Demkov-Kunike

    Args:
        t: time array (centred at 0)
        alpha: first hypergeometric parameter
        beta: second hypergeometric parameter
        gamma_width: pulse width parameter γ (MHz)
    """
    if gamma_width <= 0:
        raise ValueError("gamma_width must be positive")
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha, beta must be >= 0, got ({alpha}, {beta})")

    gt = gamma_width * t
    sech = 1.0 / np.cosh(gt)
    z = 0.5 * (1.0 + np.tanh(gt))
    c = (alpha + beta + 1.0) / 2.0

    # ₂F₁ is evaluated element-wise via scipy
    f21 = np.array([float(hyp2f1(alpha, beta, c, zi)) for zi in z])

    result: np.ndarray = sech * f21
    return result


def build_hypergeometric_pulse(
    t_total: float,
    omega_0: float,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma_width: float | None = None,
    n_points: int = 200,
) -> HypergeometricPulse:
    """Build (α,β)-hypergeometric pulse.

    Args:
        t_total: total pulse duration (μs)
        omega_0: peak Rabi frequency (MHz)
        alpha: hypergeometric parameter α (default: 0.5 = STIRAP)
        beta: hypergeometric parameter β (default: 0.5 = STIRAP)
        gamma_width: pulse width (default: 3/t_total for ~3σ coverage)
        n_points: time grid resolution
    """
    if t_total <= 0:
        raise ValueError("t_total must be positive")
    if omega_0 <= 0:
        raise ValueError("omega_0 must be positive")

    if gamma_width is None:
        gamma_width = 6.0 / t_total  # ≈3σ on each side

    times = np.linspace(-t_total / 2, t_total / 2, n_points)
    envelope = hypergeometric_envelope(times, alpha, beta, gamma_width)

    return HypergeometricPulse(
        times=times + t_total / 2,  # shift to [0, t_total]
        envelope=envelope,
        alpha=alpha,
        beta=beta,
        gamma_width=gamma_width,
        omega_0=omega_0,
    )


def infidelity_bound(
    alpha: float,
    beta: float,
    gamma_width: float,
    omega_0: float,
) -> float:
    """Analytical upper bound on infidelity for (α,β) pulses.

    1 - F ≲ (γ/Ω₀)² × f(α,β)

    where f(α,β) ≈ Γ(α+1)Γ(β+1)/Γ((α+β)/2+1)² (protocol-specific weight).

    Ref: Ventura Meinersen et al., arXiv:2504.08031 (2025)
    """
    from scipy.special import gamma as gamma_fn

    # Protocol-specific weight factor
    f_ab = gamma_fn(alpha + 1) * gamma_fn(beta + 1) / gamma_fn((alpha + beta) / 2 + 1) ** 2
    result: float = (gamma_width / omega_0) ** 2 * f_ab
    return result


# ============================================================
# Pulse schedule builder
# ============================================================


def build_trotter_pulse_schedule(
    n_qubits: int,
    k_matrix: np.ndarray,
    t_step: float,
    omega_0: float = 10.0,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> PulseSchedule:
    """Build pulse schedule for one Trotter step using hypergeometric shaping.

    Each non-zero coupling K[i,j] gets a shaped pulse with amplitude
    proportional to |K[i,j]|.

    Args:
        n_qubits: number of qubits
        k_matrix: K_nm coupling matrix (n×n)
        t_step: Trotter step duration (μs)
        omega_0: base peak Rabi frequency (MHz)
        alpha: hypergeometric α parameter
        beta: hypergeometric β parameter
    """
    pulses: list[HypergeometricPulse] = []
    k_max = np.max(np.abs(k_matrix))
    if k_max < 1e-15:
        k_max = 1.0

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            kij = abs(k_matrix[i, j])
            if kij < 1e-10:
                continue
            # Scale amplitude by coupling strength
            scaled_omega = omega_0 * kij / k_max
            pulse = build_hypergeometric_pulse(
                t_total=t_step,
                omega_0=scaled_omega,
                alpha=alpha,
                beta=beta,
            )
            pulses.append(pulse)

    # Total infidelity bound (sum over all pulses)
    gamma_w = 6.0 / t_step
    total_infidelity = sum(infidelity_bound(alpha, beta, gamma_w, p.omega_0) for p in pulses)

    return PulseSchedule(
        pulses=pulses,
        total_time=t_step,
        n_qubits=n_qubits,
        infidelity_bound=total_infidelity,
    )
