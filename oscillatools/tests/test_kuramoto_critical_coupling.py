# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Mean-field Kuramoto critical coupling tests
"""Multi-angle tests for the mean-field Kuramoto critical coupling and order parameter.

Covers the ``K_c = 2/(π g(0))`` onset against the Lorentzian (``2γ``) and Gaussian (``σ√(8/π)``)
closed forms, the normalisation and peak of the density factories, the self-consistent order
parameter against the exact Lorentzian branch ``√(1 − K_c/K)`` (and its monotone rise for a
Gaussian), the incoherent solution below the critical coupling, the over-coherent ``r = 1`` clamp
for a flat density, and the input validation of every function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from oscillatools.accel import (
    critical_coupling,
    gaussian_critical_coupling,
    gaussian_density,
    lorentzian_critical_coupling,
    lorentzian_density,
    lorentzian_order_parameter,
    synchronised_order_parameter,
)


class TestCriticalCoupling:
    def test_lorentzian_matches_closed_form(self) -> None:
        for gamma in (0.3, 0.7, 1.5):
            general = critical_coupling(lorentzian_density(gamma))
            assert general == pytest.approx(lorentzian_critical_coupling(gamma))
            assert general == pytest.approx(2.0 * gamma)

    def test_gaussian_matches_closed_form(self) -> None:
        for sigma in (0.5, 1.0, 2.0):
            general = critical_coupling(gaussian_density(sigma))
            assert general == pytest.approx(gaussian_critical_coupling(sigma))
            assert general == pytest.approx(sigma * math.sqrt(8.0 / math.pi))

    def test_critical_coupling_rejects_non_positive_density(self) -> None:
        with pytest.raises(ValueError, match="density at the centre must be positive"):
            critical_coupling(lambda _omega: 0.0)


class TestDensityFactories:
    def test_lorentzian_normalised_and_peaked(self) -> None:
        gamma = 0.6
        density = lorentzian_density(gamma)
        assert density(0.0) == pytest.approx(1.0 / (math.pi * gamma))
        mass, _ = quad(density, -np.inf, np.inf)
        assert mass == pytest.approx(1.0, abs=1e-8)

    def test_gaussian_normalised_and_peaked(self) -> None:
        sigma = 0.8
        density = gaussian_density(sigma)
        assert density(0.0) == pytest.approx(1.0 / (sigma * math.sqrt(2.0 * math.pi)))
        mass, _ = quad(density, -np.inf, np.inf)
        assert mass == pytest.approx(1.0, abs=1e-8)

    def test_density_factories_reject_non_positive_scale(self) -> None:
        with pytest.raises(ValueError, match="half_width must be positive"):
            lorentzian_density(0.0)
        with pytest.raises(ValueError, match="std must be positive"):
            gaussian_density(-1.0)


class TestClosedFormCriticalCoupling:
    def test_lorentzian_and_gaussian_values(self) -> None:
        assert lorentzian_critical_coupling(0.9) == pytest.approx(1.8)
        assert gaussian_critical_coupling(1.0) == pytest.approx(math.sqrt(8.0 / math.pi))

    def test_reject_non_positive_scale(self) -> None:
        with pytest.raises(ValueError, match="half_width must be positive"):
            lorentzian_critical_coupling(0.0)
        with pytest.raises(ValueError, match="std must be positive"):
            gaussian_critical_coupling(0.0)


class TestSelfConsistentOrderParameter:
    def test_numerical_matches_lorentzian_closed_form(self) -> None:
        gamma = 0.5  # K_c = 1.0
        for coupling in (1.5, 2.0, 4.0, 8.0):
            numeric = synchronised_order_parameter(coupling, lorentzian_density(gamma))
            closed = lorentzian_order_parameter(coupling, gamma)
            assert numeric == pytest.approx(closed, abs=1e-6)
            assert closed == pytest.approx(math.sqrt(1.0 - 2.0 * gamma / coupling))

    def test_incoherent_below_critical(self) -> None:
        gamma = 0.5  # K_c = 1.0
        assert synchronised_order_parameter(0.9, lorentzian_density(gamma)) == 0.0
        assert synchronised_order_parameter(1.0, lorentzian_density(gamma)) == 0.0
        sigma = 1.0  # K_c = sqrt(8/pi) ≈ 1.5958
        assert synchronised_order_parameter(1.0, gaussian_density(sigma)) == 0.0

    def test_gaussian_order_parameter_rises_with_coupling(self) -> None:
        sigma = 0.7
        critical = gaussian_critical_coupling(sigma)
        radii = [
            synchronised_order_parameter(critical * factor, gaussian_density(sigma))
            for factor in (1.1, 1.5, 2.0, 4.0)
        ]
        assert all(
            0.0 < earlier < later <= 1.0 for earlier, later in zip(radii, radii[1:], strict=False)
        )

    def test_flat_density_clamps_to_full_coherence(self) -> None:
        # A constant density has no sub-unity self-consistent root above its onset, so the
        # coherence saturates at the r = 1 ceiling.
        def flat_density(_omega: float) -> float:
            return 0.5

        coupling = 2.0 * critical_coupling(flat_density)
        assert synchronised_order_parameter(coupling, flat_density) == 1.0

    def test_rejects_non_positive_coupling(self) -> None:
        with pytest.raises(ValueError, match="coupling must be positive"):
            synchronised_order_parameter(0.0, lorentzian_density(0.5))

    def test_propagates_density_validation(self) -> None:
        with pytest.raises(ValueError, match="density at the centre must be positive"):
            synchronised_order_parameter(2.0, lambda _omega: 0.0)


class TestLorentzianOrderParameter:
    def test_branch_above_and_below_critical(self) -> None:
        gamma = 0.5  # K_c = 1.0
        assert lorentzian_order_parameter(0.8, gamma) == 0.0
        assert lorentzian_order_parameter(1.0, gamma) == 0.0
        assert lorentzian_order_parameter(2.0, gamma) == pytest.approx(math.sqrt(0.5))

    def test_rejects_non_positive_inputs(self) -> None:
        with pytest.raises(ValueError, match="coupling must be positive"):
            lorentzian_order_parameter(0.0, 0.5)
        with pytest.raises(ValueError, match="half_width must be positive"):
            lorentzian_order_parameter(2.0, 0.0)
