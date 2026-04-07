# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Symmetry Decay ZNE (GUESS)
"""GUESS: Guiding Extrapolations from Symmetry Decays.

Uses Hamiltonian symmetry observables with known ideal values to guide
zero-noise extrapolation of target observables. Instead of generic
polynomial extrapolation (Richardson ZNE), GUESS learns the noise profile
from observables whose ideal values are known analytically.

For the XY Hamiltonian, total magnetisation S = Σ Z_i commutes with H_XY,
so ⟨S⟩_ideal is known. The deviation under noise reveals the decay profile.

Ref: Oliva del Moral et al., arXiv:2603.13060 (2026)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SymmetryDecayModel:
    """Learned noise decay model from symmetry observable."""

    ideal_symmetry_value: float
    noisy_symmetry_values: list[float]
    noise_scales: list[int]
    alpha: float  # learned scaling exponent
    fit_residual: float


@dataclass
class GUESSResult:
    """GUESS mitigation result."""

    raw_value: float
    mitigated_value: float
    decay_model: SymmetryDecayModel
    correction_factor: float


def learn_symmetry_decay(
    ideal_symmetry_value: float,
    noisy_symmetry_values: list[float],
    noise_scales: list[int],
) -> SymmetryDecayModel:
    """Learn noise scaling exponent α from symmetry observable decay.

    The symmetry observable S satisfies [H, S] = 0, so ⟨S⟩_ideal is
    analytically known. At noise scale g:

        ⟨S⟩_g = ⟨S⟩_ideal × exp(-α × (g - 1))

    We fit α from multiple noise levels via log-linear regression.
    """
    if len(noisy_symmetry_values) != len(noise_scales):
        raise ValueError("Length mismatch: noisy values vs noise scales")
    if len(noise_scales) < 2:
        raise ValueError("Need >= 2 noise scales to fit decay model")
    if abs(ideal_symmetry_value) < 1e-15:
        raise ValueError("ideal_symmetry_value too close to zero")

    # log(⟨S⟩_g / ⟨S⟩_ideal) = -α × (g - 1)
    ratios = np.array(noisy_symmetry_values) / ideal_symmetry_value
    # Clamp ratios to avoid log(0) or log(negative)
    ratios = np.clip(ratios, 1e-15, None)
    log_ratios = np.log(ratios)
    g_shifted = np.array(noise_scales, dtype=float) - 1.0

    # Linear fit: log_ratio = -α × g_shifted + offset
    # offset should be ~0 for g=1, but fit anyway for robustness
    if np.std(g_shifted) < 1e-15:
        alpha = 0.0
        residual = 0.0
    else:
        coeffs = np.polyfit(g_shifted, log_ratios, 1)
        alpha = -float(coeffs[0])
        residual = float(np.sqrt(np.mean((np.polyval(coeffs, g_shifted) - log_ratios) ** 2)))

    return SymmetryDecayModel(
        ideal_symmetry_value=ideal_symmetry_value,
        noisy_symmetry_values=list(noisy_symmetry_values),
        noise_scales=list(noise_scales),
        alpha=alpha,
        fit_residual=residual,
    )


def guess_extrapolate(
    target_noisy_value: float,
    symmetry_noisy_value: float,
    decay_model: SymmetryDecayModel,
) -> GUESSResult:
    """Apply GUESS correction to a target observable.

    Eq. from Oliva del Moral et al. (2026), Eq. 5:
        ⟨O⟩_mitigated ≈ ⟨O⟩_noisy × (⟨S⟩_ideal / ⟨S⟩_noisy)^α

    where α is learned from the symmetry decay model.
    """
    s_ideal = decay_model.ideal_symmetry_value
    s_noisy = symmetry_noisy_value

    if abs(s_noisy) < 1e-15:
        # Symmetry completely decayed — correction undefined
        return GUESSResult(
            raw_value=target_noisy_value,
            mitigated_value=target_noisy_value,
            decay_model=decay_model,
            correction_factor=1.0,
        )

    ratio = abs(s_ideal / s_noisy)
    correction = ratio**decay_model.alpha
    mitigated = target_noisy_value * correction

    return GUESSResult(
        raw_value=target_noisy_value,
        mitigated_value=mitigated,
        decay_model=decay_model,
        correction_factor=correction,
    )


def xy_magnetisation_ideal(n_qubits: int, initial_state: str = "ground") -> float:
    """Ideal total magnetisation ⟨Σ Z_i⟩ for standard initial states.

    For the XY Hamiltonian, Σ Z_i commutes with H, so this is conserved.

    Args:
        n_qubits: number of qubits
        initial_state: "ground" (|00...0⟩, ⟨Z⟩=+1 each) or "neel" (|0101...⟩)
    """
    if initial_state == "ground":
        return float(n_qubits)
    if initial_state == "neel":
        # Neel: alternating +1 and -1
        return float(n_qubits % 2)  # 0 for even, 1 for odd
    raise ValueError(f"Unknown initial state: {initial_state}")
