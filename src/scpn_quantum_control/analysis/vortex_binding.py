# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Vortex Binding
"""Vortex binding energy and Kosterlitz renormalization group.

At the BKT transition, vortex-antivortex pairs unbind when the
entropy gain from separating exceeds the energy cost:

    E_pair(r) = 2*pi*J_eff * ln(r/a)  (logarithmic interaction)
    S_pair(r) = 2*k_B * ln(r/a)        (configurational entropy)

Free energy: F = E - TS = (2*pi*J - 2T) * ln(r/a)

Binding condition: F > 0 → T < pi*J_eff → T_BKT = pi*J_eff

The Kosterlitz RG flow equations (in fugacity y and stiffness K):
    dK^{-1}/dl = 4*pi^3 * y^2
    dy/dl = (2 - pi*K) * y

Fixed point: K* = 2/pi (Nelson-Kosterlitz universal jump).

For the finite graph with coupling matrix K_nm:
    J_eff = lambda_2(L_K) / (2n)  (from bkt_analysis.py)
    E_pair = 2*pi*J_eff * ln(L_eff)
    T_BKT = pi*J_eff

The ratio E_pair / T_BKT = 2*ln(L_eff) is universal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bkt_analysis import bkt_analysis


@dataclass
class VortexBindingResult:
    """Vortex binding energy analysis."""

    j_eff: float  # effective spin stiffness
    e_pair: float  # vortex pair energy at system scale
    t_bkt: float  # BKT transition temperature
    binding_ratio: float  # E_pair / T_BKT = 2*ln(L_eff)
    free_energy_at_bkt: float  # F(T_BKT) — should be ~0
    rg_fixed_point_k: float  # K* = 2/pi
    is_bound: bool  # T < T_BKT → pairs are bound
    n_oscillators: int


def vortex_pair_energy(j_eff: float, system_size: float, cutoff: float = 1.0) -> float:
    """Logarithmic vortex pair interaction energy.

    E = 2*pi*J * ln(L/a) where L = system size, a = lattice cutoff.
    """
    if system_size <= cutoff:
        return 0.0
    return float(2.0 * np.pi * j_eff * np.log(system_size / cutoff))


def vortex_pair_entropy(system_size: float, cutoff: float = 1.0) -> float:
    """Configurational entropy of a vortex pair.

    S = 2 * ln(L/a) (in units where k_B = 1).
    """
    if system_size <= cutoff:
        return 0.0
    return float(2.0 * np.log(system_size / cutoff))


def vortex_free_energy(j_eff: float, temperature: float, system_size: float) -> float:
    """Free energy F = E - T*S for a vortex pair."""
    e = vortex_pair_energy(j_eff, system_size)
    s = vortex_pair_entropy(system_size)
    return e - temperature * s


def kosterlitz_rg_step(
    k_inv: float,
    y: float,
    dl: float = 0.01,
) -> tuple[float, float]:
    """One step of Kosterlitz RG flow.

    dK^{-1}/dl = 4*pi^3 * y^2
    dy/dl = (2 - pi/K^{-1}) * y  (note: pi*K = pi/K^{-1})
    """
    k = 1.0 / max(k_inv, 1e-15)
    dk_inv = 4.0 * np.pi**3 * y**2 * dl
    dy = (2.0 - np.pi * k) * y * dl
    return k_inv + dk_inv, y + dy


def compute_vortex_binding(
    K: np.ndarray,
    omega: np.ndarray,
    temperature: float | None = None,
) -> VortexBindingResult:
    """Full vortex binding analysis.

    Args:
        K: coupling matrix
        omega: natural frequencies
        temperature: effective temperature (default: T_BKT estimate)
    """
    n = K.shape[0]
    bkt = bkt_analysis(K)
    j_eff = bkt.effective_coupling
    t_bkt = bkt.t_bkt_estimate

    if temperature is None:
        temperature = t_bkt

    l_eff = np.sqrt(n)
    e_pair = vortex_pair_energy(j_eff, l_eff)

    binding_ratio = e_pair / max(t_bkt, 1e-15)
    f_at_bkt = vortex_free_energy(j_eff, t_bkt, l_eff)
    is_bound = temperature < t_bkt

    return VortexBindingResult(
        j_eff=j_eff,
        e_pair=e_pair,
        t_bkt=t_bkt,
        binding_ratio=binding_ratio,
        free_energy_at_bkt=f_at_bkt,
        rg_fixed_point_k=2.0 / np.pi,
        is_bound=is_bound,
        n_oscillators=n,
    )
