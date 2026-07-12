# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — kuramoto noisy mean field tests
"""Multi-angle tests for the noisy Kuramoto Fokker–Planck mean-field theory.

Covers the noisy critical coupling (the closed-form Lorentzian shift ``2(γ + D)``, the
quadrature onset against it, the ``D → 0`` reduction to the deterministic Kuramoto value, the
Gaussian density), the self-consistent stationary order parameter (zero below the noisy onset, a
positive monotone branch above it, agreement with a finite-population Euler–Maruyama simulation,
the onset recovered from where the branch leaves zero, the ``D → 0`` reduction to the
deterministic self-consistency) and the input validation of every entry point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from oscillatools.accel import (
    gaussian_density,
    integrate_noisy_kuramoto,
    lorentzian_density,
    lorentzian_noisy_critical_coupling,
    mean_field_force,
    noisy_critical_coupling,
    noisy_stationary_order_parameter,
    synchronised_order_parameter,
)


class TestNoisyCriticalCoupling:
    @pytest.mark.parametrize("diffusion", [0.0, 0.25, 0.5, 1.0, 2.0])
    def test_lorentzian_closed_form(self, diffusion: float) -> None:
        assert lorentzian_noisy_critical_coupling(1.5, diffusion) == pytest.approx(
            2.0 * (1.5 + diffusion)
        )

    @pytest.mark.parametrize("diffusion", [0.25, 0.5, 1.0])
    def test_quadrature_matches_closed_form(self, diffusion: float) -> None:
        # K_c = 2/∫ g(ω) D/(D²+ω²) dω equals 2(γ + D) for a Lorentzian.
        value = noisy_critical_coupling(
            lorentzian_density(1.0), diffusion, frequency_limit=200.0, n_frequency=4001
        )
        assert value == pytest.approx(2.0 * (1.0 + diffusion), rel=2e-3)

    def test_zero_diffusion_reduces_to_deterministic(self) -> None:
        # D → 0 returns the deterministic Kuramoto onset 2/(π g(0)).
        assert noisy_critical_coupling(lorentzian_density(1.0), 0.0) == pytest.approx(2.0)
        assert noisy_critical_coupling(gaussian_density(1.0), 0.0) == pytest.approx(
            2.0 / (math.pi * gaussian_density(1.0)(0.0))
        )

    def test_degenerate_density_rejected(self) -> None:
        # A density with no mass on the frequency grid leaves the kernel integral at zero.
        with pytest.raises(ValueError, match="kernel integral must be positive"):
            noisy_critical_coupling(lambda omega: 0.0, 0.5)

    def test_gaussian_onset_rises_with_noise(self) -> None:
        quiet = noisy_critical_coupling(gaussian_density(1.0), 0.0)
        loud = noisy_critical_coupling(
            gaussian_density(1.0), 0.5, frequency_limit=60.0, n_frequency=2001
        )
        assert loud > quiet

    def test_lorentzian_closed_form_validation(self) -> None:
        with pytest.raises(ValueError, match="half_width must be positive"):
            lorentzian_noisy_critical_coupling(0.0, 0.5)
        with pytest.raises(ValueError, match="diffusion must be non-negative"):
            lorentzian_noisy_critical_coupling(1.0, -0.1)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"diffusion": -0.1}, "diffusion must be non-negative"),
            ({"diffusion": 0.5, "frequency_limit": 0.0}, "frequency_limit must be positive"),
            ({"diffusion": 0.5, "n_frequency": 2}, "n_frequency must be at least 3"),
        ],
    )
    def test_quadrature_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            noisy_critical_coupling(lorentzian_density(1.0), **kwargs)


class TestNoisyStationaryOrderParameter:
    def test_zero_below_onset(self) -> None:
        # K below K_c(D) = 2(1 + 0.5) = 3 gives the incoherent state.
        r = noisy_stationary_order_parameter(
            2.5, lorentzian_density(1.0), 0.5, n_phase=128, frequency_limit=40.0, n_frequency=161
        )
        assert r == 0.0

    def test_positive_and_monotone_above_onset(self) -> None:
        density = lorentzian_density(1.0)
        radii = [
            noisy_stationary_order_parameter(
                K, density, 0.5, n_phase=160, frequency_limit=40.0, n_frequency=161
            )
            for K in (3.3, 4.0, 5.0, 6.0)
        ]
        assert all(r > 0.0 for r in radii)
        assert radii[0] < radii[1] < radii[2] < radii[3]
        assert all(r <= 1.0 for r in radii)

    def test_matches_simulation(self) -> None:
        # The Fokker–Planck stationary r is the N → ∞ mean-field value; a large finite-population
        # Euler–Maruyama run (biased slightly low by finite N / dt) sits just below it.
        density = lorentzian_density(1.0)
        diffusion, coupling = 0.5, 5.0
        r_theory = noisy_stationary_order_parameter(
            coupling, density, diffusion, n_phase=200, frequency_limit=45.0, n_frequency=181
        )
        rng = np.random.default_rng(0)
        n = 8000
        omega = np.tan(np.pi * (rng.uniform(size=n) - 0.5))
        omega -= omega.mean()
        run = integrate_noisy_kuramoto(
            rng.uniform(-np.pi, np.pi, n),
            omega,
            lambda theta: mean_field_force(theta, coupling),
            diffusion=diffusion,
            dt=0.01,
            n_steps=6000,
            seed=1,
        )
        assert abs(r_theory - run.mean_order_parameter) < 0.04

    def test_onset_recovered_from_branch(self) -> None:
        # The branch leaves zero at the noisy onset: just below K_c = 3 → 0, just above → positive.
        density = lorentzian_density(1.0)
        below = noisy_stationary_order_parameter(
            2.9, density, 0.5, n_phase=160, frequency_limit=40.0, n_frequency=161
        )
        above = noisy_stationary_order_parameter(
            3.2, density, 0.5, n_phase=160, frequency_limit=40.0, n_frequency=161
        )
        assert below == 0.0
        assert above > 0.0

    def test_zero_diffusion_reduces_to_deterministic(self) -> None:
        density = lorentzian_density(1.0)
        for coupling in (3.0, 5.0):
            assert noisy_stationary_order_parameter(coupling, density, 0.0) == pytest.approx(
                synchronised_order_parameter(coupling, density)
            )

    def test_strong_coupling_approaches_but_stays_below_one(self) -> None:
        # With diffusion the noise always broadens the stationary density, so a Lorentzian never
        # locks fully: strong coupling drives r high yet strictly below one.
        density = lorentzian_density(1.0)
        moderate = noisy_stationary_order_parameter(
            8.0, density, 0.5, n_phase=128, frequency_limit=40.0, n_frequency=161
        )
        strong = noisy_stationary_order_parameter(
            40.0, density, 0.5, n_phase=128, frequency_limit=40.0, n_frequency=161
        )
        assert moderate < strong < 1.0
        assert strong > 0.9

    def test_rejects_non_positive_coupling(self) -> None:
        with pytest.raises(ValueError, match="coupling must be positive"):
            noisy_stationary_order_parameter(0.0, lorentzian_density(1.0), 0.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"diffusion": -0.1}, "diffusion must be non-negative"),
            ({"diffusion": 0.5, "frequency_limit": 0.0}, "frequency_limit must be positive"),
            ({"diffusion": 0.5, "n_frequency": 2}, "n_frequency must be at least 3"),
            ({"diffusion": 0.5, "n_phase": 4}, "n_phase must be at least 8"),
        ],
    )
    def test_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            noisy_stationary_order_parameter(4.0, lorentzian_density(1.0), **kwargs)


def test_public_symbols_exported() -> None:
    import oscillatools.accel as accel

    for symbol in (
        "FrequencyDensity",
        "lorentzian_noisy_critical_coupling",
        "noisy_critical_coupling",
        "noisy_stationary_order_parameter",
    ):
        assert symbol in accel.__all__
        assert hasattr(accel, symbol)
