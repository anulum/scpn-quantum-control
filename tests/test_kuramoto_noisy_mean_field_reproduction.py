# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — noisy Kuramoto mean-field theory reproduction
"""Reproduce the noisy Kuramoto Fokker–Planck mean-field theory against direct simulation.

For the Kuramoto model with independent white noise, ``dθ_i = [ω_i + (K/N) Σ_j sin(θ_j−θ_i)] dt
+ √(2D) dW_i``, the stationary order parameter follows a self-consistent Fokker–Planck mean-field
theory: it is zero below a diffusion-shifted critical coupling (for a Lorentzian half-width ``γ``
the onset is ``K_c = 2(γ + D)``) and rises above it (Sakaguchi, *Prog. Theor. Phys.* **79**, 39
(1988); Strogatz & Mirollo, *J. Stat. Phys.* **63**, 613 (1991)).

The existing suite tests the closed-form theory (:func:`noisy_stationary_order_parameter`) and the
stochastic integrator (:func:`integrate_noisy_kuramoto`) *separately*. This reproduction closes the
loop between them — the analogue of the exact-reduction-vs-ensemble checks for the deterministic
Ott–Antonsen and Watanabe–Strogatz reductions — by integrating a large ensemble and confirming its
settled order parameter matches the closed-form theory above the onset, sits at the finite-``N``
incoherent floor below it (where the theory is exactly zero), and follows the theory across a
diffusion-induced onset shift.

Determinism: the natural frequencies are the Lorentzian inverse-CDF at symmetric quantiles (no
random draws) and the Euler–Maruyama noise is seeded, so each ensemble is reproducible bit-for-bit;
the settled order parameter is self-averaging, so independent seeds agree within the finite-size
fluctuation.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel import (
    integrate_noisy_kuramoto,
    lorentzian_density,
    lorentzian_noisy_critical_coupling,
    mean_field_force,
    noisy_stationary_order_parameter,
)

_N = 800
_HALF_WIDTH = 1.0  # Lorentzian half-width γ
_DT = 0.01
_N_STEPS = 3500
_TAIL = _N_STEPS // 4  # samples averaged for the settled order parameter
_SEED = 11

_DIFFUSION = 0.5  # so the Lorentzian onset sits at K_c = 2(γ + D) = 3.0
_BELOW_ONSET = 1.5
_ABOVE_ONSET: tuple[float, ...] = (4.0, 6.0)

# The diffusion-induced onset shift, probed at a fixed coupling between the two onsets.
_SHIFT_COUPLING = 2.5
_WEAK_DIFFUSION = 0.1  # onset K_c = 2.2 < 2.5  => synchronised
_STRONG_DIFFUSION = 0.5  # onset K_c = 3.0 > 2.5  => incoherent

_TOLERANCE = 0.05  # ensemble-vs-theory agreement band (prototyped |diff| <= 0.013 above onset)
_FLOOR = 0.15  # finite-N incoherent ceiling


def _lorentzian_quantile_frequencies(count: int, half_width: float) -> NDArray[np.float64]:
    """Deterministic Lorentzian frequencies: the inverse CDF at symmetric mid-quantiles."""
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    return half_width * np.tan(np.pi * (quantiles - 0.5))


def _simulated_stationary_order_parameter(coupling: float, diffusion: float, seed: int) -> float:
    """Settled order parameter of a seeded large-N noisy Kuramoto ensemble."""
    omega = _lorentzian_quantile_frequencies(_N, _HALF_WIDTH)
    phases = np.linspace(-np.pi, np.pi, _N, endpoint=False)

    def force(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return mean_field_force(theta, coupling)

    run = integrate_noisy_kuramoto(
        phases,
        omega,
        force,
        diffusion=diffusion,
        dt=_DT,
        n_steps=_N_STEPS,
        seed=seed,
        settle_steps=_TAIL,
    )
    return float(run.mean_order_parameter)


def _theory(coupling: float, diffusion: float) -> float:
    return float(
        noisy_stationary_order_parameter(coupling, lorentzian_density(_HALF_WIDTH), diffusion)
    )


@pytest.fixture(scope="module")
def stationary() -> dict[float, float]:
    """Settled ensemble order parameter at ``D = 0.5`` for the below/above couplings (once)."""
    couplings = (_BELOW_ONSET, *_ABOVE_ONSET)
    return {k: _simulated_stationary_order_parameter(k, _DIFFUSION, _SEED) for k in couplings}


@pytest.mark.parametrize("coupling", _ABOVE_ONSET)
def test_stationary_order_parameter_matches_mean_field_theory_above_onset(
    stationary: dict[float, float], coupling: float
) -> None:
    """Above the onset the ensemble settles onto the closed-form Fokker–Planck theory."""
    simulated = stationary[coupling]
    predicted = _theory(coupling, _DIFFUSION)
    assert predicted > 0.3, "the chosen coupling must be above the noisy onset"
    assert abs(simulated - predicted) < _TOLERANCE, (
        f"K={coupling}: ensemble {simulated:.3f} vs theory {predicted:.3f}"
    )


def test_below_onset_is_incoherent_and_theory_is_zero(stationary: dict[float, float]) -> None:
    """Below the noisy onset the ensemble sits at the finite-N floor and the theory is zero."""
    assert _theory(_BELOW_ONSET, _DIFFUSION) == 0.0
    assert stationary[_BELOW_ONSET] < _FLOOR, (
        f"below onset the ensemble should be incoherent, got r={stationary[_BELOW_ONSET]:.3f}"
    )


def test_order_parameter_rises_with_coupling_above_onset(stationary: dict[float, float]) -> None:
    """Coherence increases with coupling once past the onset."""
    assert stationary[_BELOW_ONSET] < stationary[_ABOVE_ONSET[0]] < stationary[_ABOVE_ONSET[1]]


def test_diffusion_raises_the_onset() -> None:
    """At a fixed coupling, stronger diffusion pushes the ensemble across the onset — as the theory does.

    At K=2.5 weak diffusion keeps the onset below the coupling (synchronised, matching the theory);
    strong diffusion lifts the onset above it (incoherent, with the theory exactly zero).
    """
    weak_onset = lorentzian_noisy_critical_coupling(_HALF_WIDTH, _WEAK_DIFFUSION)
    strong_onset = lorentzian_noisy_critical_coupling(_HALF_WIDTH, _STRONG_DIFFUSION)
    assert weak_onset < _SHIFT_COUPLING < strong_onset, "the probe must straddle the two onsets"

    weak = _simulated_stationary_order_parameter(_SHIFT_COUPLING, _WEAK_DIFFUSION, _SEED)
    strong = _simulated_stationary_order_parameter(_SHIFT_COUPLING, _STRONG_DIFFUSION, _SEED)
    assert abs(weak - _theory(_SHIFT_COUPLING, _WEAK_DIFFUSION)) < _TOLERANCE
    assert _theory(_SHIFT_COUPLING, _STRONG_DIFFUSION) == 0.0
    assert strong < _FLOOR < weak, (
        f"diffusion failed to cross the onset: weak r={weak:.3f}, strong r={strong:.3f}"
    )


def test_stationary_value_is_self_averaging(stationary: dict[float, float]) -> None:
    """The settled order parameter is self-averaging — an independent seed agrees within fluctuations."""
    other_seed = _simulated_stationary_order_parameter(_ABOVE_ONSET[1], _DIFFUSION, seed=29)
    assert abs(other_seed - stationary[_ABOVE_ONSET[1]]) < _TOLERANCE
