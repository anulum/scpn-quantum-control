# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Microscopic synchronisation-onset witness for the analytical critical coupling
r"""Microscopic-ensemble witness that the analytical critical coupling predicts a real onset.

``tests/test_kuramoto_critical_coupling.py`` checks the ``K_c = 2/(π g(0))`` layer only against its
own closed forms — analytical against analytical. That leaves the load-bearing physical claim
unwitnessed: that ``K_c`` is where a *microscopic* ensemble of ``θ̇_i = ω_i + (K/N) Σ_j sin(θ_j − θ_i)``
oscillators actually begins to synchronise. This module supplies that second, independent witness.

For a chosen frequency density ``g`` it builds an ``N``-oscillator ensemble whose natural frequencies
are the deterministic quantiles of ``g`` (inverse-CDF sampling, which reproduces the ideal density far
more faithfully than random draws at finite ``N`` and so sharpens the transition), integrates the
mean-field Kuramoto model with the production RK4 trajectory
(:func:`~oscillatools.accel.diff_kuramoto_rk4.kuramoto_rk4_trajectory`), and reads the
stationary Kuramoto order parameter with the production observable
(:func:`~oscillatools.accel.order_parameter_observables.order_parameter`) averaged over the
tail of the trajectory. Sweeping the global coupling ``K`` then anchors the simulation against the
production analysis layer three ways:

* **below** the analytical ``K_c`` the ensemble stays incoherent — the order parameter sits at the
  ``~1/√N`` finite-size floor;
* **above** it the order parameter follows the exact Lorentzian branch
  ``r = √(1 − K_c/K)`` produced by :func:`lorentzian_order_parameter`, to a few parts in a hundred;
* the empirical transition **brackets** the analytical ``K_c`` for both a Lorentzian and a Gaussian
  density (the latter fixing ``K_c`` through :func:`critical_coupling` / :func:`gaussian_critical_coupling`),
  so the anchor is not a Lorentzian-specific coincidence.

The finite-``N`` transition has a physical width and sits a little below the thermodynamic ``K_c``;
the brackets below are the honest expression of that width, not slack tolerances. Everything is
seeded and deterministic (the RK4 floor and the Rust tier integrate the identical scheme), so the
witness is reproducible to floating-point rounding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.special import erfinv

from oscillatools.accel import (
    critical_coupling,
    gaussian_critical_coupling,
    gaussian_density,
    lorentzian_critical_coupling,
    lorentzian_density,
    lorentzian_order_parameter,
    order_parameter,
)
from oscillatools.accel.diff_kuramoto_rk4 import kuramoto_rk4_trajectory

# Ensemble size, integration grid and averaging window. N is large enough that the finite-size floor
# ~1/√N ≈ 0.088 sits well below the synchronised branch, yet small enough that the whole sweep runs in
# a few seconds on the vectorised Python floor as well as the Rust tier.
_N = 128
_DT = 0.05
_N_STEPS = 800  # integration horizon T = 40
_TAIL = 200  # average the order parameter over the final T = 10
_SEED = 20260702
_FINITE_SIZE_FLOOR = 1.0 / math.sqrt(_N)


def _lorentzian_quantile_frequencies(count: int, half_width: float) -> NDArray[np.float64]:
    r"""Deterministic quantiles of the Lorentzian ``g(ω) = (γ/π)/(ω² + γ²)``.

    The inverse CDF is ``ω = γ tan(π(u − ½))``; sampling it on the symmetric midpoint grid
    ``u_i = (i + ½)/count`` gives an exactly zero-mean, symmetric frequency set whose realised density
    matches the ideal Lorentzian, so the synchronisation transition is sharp at finite ``count``.
    """
    quantiles = (np.arange(count) + 0.5) / count
    return np.asarray(half_width * np.tan(math.pi * (quantiles - 0.5)), dtype=np.float64)


def _gaussian_quantile_frequencies(count: int, std: float) -> NDArray[np.float64]:
    r"""Deterministic quantiles of the Gaussian ``g(ω) = exp(−ω²/2σ²)/(σ√(2π))``.

    The inverse CDF is ``ω = σ√2 · erf⁻¹(2u − 1)`` on the same symmetric midpoint grid, giving a
    zero-mean, light-tailed frequency set with a sharp onset.
    """
    quantiles = (np.arange(count) + 0.5) / count
    return np.asarray(std * math.sqrt(2.0) * erfinv(2.0 * quantiles - 1.0), dtype=np.float64)


def _stationary_order_parameter(
    omega: NDArray[np.float64],
    theta0: NDArray[np.float64],
    coupling_strength: float,
) -> float:
    r"""Tail-averaged Kuramoto order parameter of the mean-field ensemble at global coupling ``K``.

    Integrates ``θ̇_i = ω_i + (K/N) Σ_j sin(θ_j − θ_i)`` — the all-to-all coupling matrix ``K/N`` with
    a zero diagonal — with the production RK4 trajectory, then averages the production order parameter
    over the tail of the trajectory to wash out the finite-size fluctuation of the stationary state.
    """
    count = omega.size
    coupling = np.full((count, count), coupling_strength / count, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    trajectory = kuramoto_rk4_trajectory(theta0, omega, coupling, _DT, _N_STEPS)
    tail = trajectory[-_TAIL:]
    return float(np.mean([order_parameter(row) for row in tail]))


def _sweep(
    omega: NDArray[np.float64],
    theta0: NDArray[np.float64],
    couplings: tuple[float, ...],
) -> dict[float, float]:
    return {k: _stationary_order_parameter(omega, theta0, k) for k in couplings}


# Lorentzian ensemble: γ = ½ ⇒ analytical K_c = 2γ = 1.0. The grid straddles the onset with two
# clearly sub-critical points, the near-onset knee and four points on the synchronised branch.
_LORENTZIAN_GAMMA = 0.5
_LORENTZIAN_COUPLINGS = (0.5, 0.8, 1.1, 1.3, 1.6, 2.0, 3.0)

# Gaussian ensemble: σ = 0.6 ⇒ analytical K_c = σ√(8/π) ≈ 0.958. A leaner grid — one sub-critical and
# three synchronised points — is enough to bracket the onset for a second, non-Lorentzian density.
_GAUSSIAN_STD = 0.6
_GAUSSIAN_COUPLINGS = (0.5, 0.8, 1.3, 2.0)


@pytest.fixture(scope="module")
def lorentzian_sweep() -> dict[float, float]:
    rng = np.random.default_rng(_SEED)
    theta0 = rng.uniform(-math.pi, math.pi, _N)
    omega = _lorentzian_quantile_frequencies(_N, _LORENTZIAN_GAMMA)
    return _sweep(omega, theta0, _LORENTZIAN_COUPLINGS)


@pytest.fixture(scope="module")
def gaussian_sweep() -> dict[float, float]:
    rng = np.random.default_rng(_SEED + 1)
    theta0 = rng.uniform(-math.pi, math.pi, _N)
    omega = _gaussian_quantile_frequencies(_N, _GAUSSIAN_STD)
    return _sweep(omega, theta0, _GAUSSIAN_COUPLINGS)


class TestLorentzianMicroscopicOnset:
    def test_critical_coupling_is_the_general_and_closed_form_value(self) -> None:
        # The onset the microscopic sweep is anchored against is the production K_c, taken two ways.
        general = critical_coupling(lorentzian_density(_LORENTZIAN_GAMMA))
        closed = lorentzian_critical_coupling(_LORENTZIAN_GAMMA)
        assert general == pytest.approx(closed)
        assert closed == pytest.approx(2.0 * _LORENTZIAN_GAMMA)

    def test_incoherent_below_the_critical_coupling(
        self, lorentzian_sweep: dict[float, float]
    ) -> None:
        # Below K_c = 1.0 the ensemble does not synchronise: the order parameter stays near the
        # ~1/√N finite-size floor, an order of magnitude under the synchronised branch.
        assert lorentzian_sweep[0.5] < 2.0 * _FINITE_SIZE_FLOOR
        assert lorentzian_sweep[0.8] < 2.0 * _FINITE_SIZE_FLOOR

    def test_synchronised_branch_matches_the_production_closed_form(
        self, lorentzian_sweep: dict[float, float]
    ) -> None:
        # Above K_c the tail-averaged order parameter tracks the exact Lorentzian branch
        # r = √(1 − K_c/K) produced by the analysis layer, to a few parts in a hundred at N = 128.
        for coupling in (1.3, 1.6, 2.0, 3.0):
            predicted = lorentzian_order_parameter(coupling, _LORENTZIAN_GAMMA)
            assert lorentzian_sweep[coupling] == pytest.approx(predicted, abs=0.05)

    def test_onset_brackets_the_critical_coupling_and_rises_monotonically(
        self, lorentzian_sweep: dict[float, float]
    ) -> None:
        k_c = lorentzian_critical_coupling(_LORENTZIAN_GAMMA)
        # The empirical transition brackets the analytical K_c = 1.0: incoherent at 0.8·K_c, an
        # unmistakable synchronised cluster (r > ½ the asymptotic branch) by 1.3·K_c.
        assert lorentzian_sweep[0.8 * k_c] < 0.15
        assert lorentzian_sweep[1.3 * k_c] > 0.4
        # The order parameter is monotone in the coupling across the sweep.
        ordered = [lorentzian_sweep[k] for k in (0.5, 0.8, 1.1, 1.3, 1.6, 2.0, 3.0)]
        assert all(earlier < later for earlier, later in zip(ordered, ordered[1:], strict=False))


class TestGaussianMicroscopicOnset:
    def test_critical_coupling_is_the_general_and_closed_form_value(self) -> None:
        general = critical_coupling(gaussian_density(_GAUSSIAN_STD))
        closed = gaussian_critical_coupling(_GAUSSIAN_STD)
        assert general == pytest.approx(closed)
        assert closed == pytest.approx(_GAUSSIAN_STD * math.sqrt(8.0 / math.pi))

    def test_onset_brackets_the_critical_coupling(
        self, gaussian_sweep: dict[float, float]
    ) -> None:
        k_c = gaussian_critical_coupling(_GAUSSIAN_STD)  # ≈ 0.958
        # A second, non-Lorentzian density: the analytical onset is bracketed by the sweep's 0.8 and
        # 1.3 points — incoherent below K_c, strongly synchronised above it — so the anchor holds for
        # a different g as well.
        assert 0.8 < k_c < 1.3
        assert gaussian_sweep[0.5] < 2.0 * _FINITE_SIZE_FLOOR
        assert gaussian_sweep[0.8] < 0.3
        assert gaussian_sweep[1.3] > 0.6
        assert gaussian_sweep[2.0] > 0.9

    def test_order_parameter_rises_monotonically_through_the_onset(
        self, gaussian_sweep: dict[float, float]
    ) -> None:
        ordered = [gaussian_sweep[k] for k in _GAUSSIAN_COUPLINGS]
        assert all(earlier < later for earlier, later in zip(ordered, ordered[1:], strict=False))
