# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the thermodynamic-witness observable
"""Tests for the calibrated thermodynamic-work witness observable.

Covers the fail-closed input contract (no invented work), scalar vs sample
inputs, the empty/non-finite rejections, and the optional dissipated-work and
Jarzynski free-energy branches including their finiteness/positivity guards.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.analysis.thermodynamic_witness import ThermodynamicWitness


class TestInputContract:
    """The witness refuses to invent work and validates its samples."""

    def test_requires_work_input(self) -> None:
        """With neither samples nor a scalar, the witness fails closed."""
        with pytest.raises(ValueError, match="calibrated work protocol"):
            ThermodynamicWitness()(counts={"0": 8})

    def test_rejects_empty_samples(self) -> None:
        """An empty work-sample sequence is rejected."""
        with pytest.raises(ValueError, match="at least one sample"):
            ThermodynamicWitness()(work_samples_joule=[])

    def test_rejects_non_finite_samples(self) -> None:
        """Non-finite work samples are rejected."""
        with pytest.raises(ValueError, match="finite joule"):
            ThermodynamicWitness()(work_samples_joule=[1.0, math.inf])


class TestWorkStatistics:
    """Mean and variance over scalar and multi-sample inputs."""

    def test_scalar_work_has_zero_variance(self) -> None:
        """A single scalar work value yields zero variance and one sample."""
        result = ThermodynamicWitness()(work_joule=2.5)
        assert result["mean_work_joule"] == pytest.approx(2.5)
        assert result["work_variance_joule2"] == pytest.approx(0.0)
        assert result["n_work_samples"] == pytest.approx(1.0)

    def test_sample_mean_and_unbiased_variance(self) -> None:
        """Multiple samples report the mean and the ddof=1 sample variance."""
        samples = [1.0, 2.0, 3.0]
        result = ThermodynamicWitness()(work_samples_joule=samples)
        assert result["mean_work_joule"] == pytest.approx(2.0)
        assert result["work_variance_joule2"] == pytest.approx(float(np.var(samples, ddof=1)))
        assert result["n_work_samples"] == pytest.approx(3.0)

    def test_counts_are_ignored(self) -> None:
        """Bitstring counts never contribute to the work estimate."""
        with_counts = ThermodynamicWitness()(counts={"0": 100}, work_joule=1.0)
        assert with_counts["mean_work_joule"] == pytest.approx(1.0)


class TestDissipatedWork:
    """Optional free-energy difference yields dissipated work."""

    def test_dissipated_work_is_mean_minus_delta_f(self) -> None:
        """Dissipated work equals mean work minus the supplied free-energy gap."""
        result = ThermodynamicWitness()(work_samples_joule=[2.0, 4.0], delta_free_energy_joule=1.0)
        assert result["delta_free_energy_joule"] == pytest.approx(1.0)
        assert result["dissipated_work_joule"] == pytest.approx(3.0 - 1.0)

    def test_rejects_non_finite_delta_f(self) -> None:
        """A non-finite free-energy difference is rejected."""
        with pytest.raises(ValueError, match="delta_free_energy_joule must be finite"):
            ThermodynamicWitness()(work_joule=1.0, delta_free_energy_joule=math.nan)


class TestJarzynski:
    """Optional inverse temperature yields the Jarzynski estimate."""

    def test_jarzynski_estimate_for_constant_work(self) -> None:
        """For constant work the Jarzynski free energy equals that work."""
        result = ThermodynamicWitness()(work_samples_joule=[3.0, 3.0], beta_per_joule=2.0)
        assert result["beta_per_joule"] == pytest.approx(2.0)
        assert result["jarzynski_delta_free_energy_joule"] == pytest.approx(3.0)

    def test_jarzynski_residual_against_supplied_delta_f(self) -> None:
        """Supplying both delta_f and beta reports the Jarzynski residual."""
        result = ThermodynamicWitness()(
            work_samples_joule=[3.0, 3.0],
            beta_per_joule=2.0,
            delta_free_energy_joule=2.5,
        )
        assert result["jarzynski_residual_joule"] == pytest.approx(3.0 - 2.5)

    @pytest.mark.parametrize("beta", [0.0, -1.0, math.inf])
    def test_rejects_non_positive_or_non_finite_beta(self, beta: float) -> None:
        """beta must be finite and strictly positive."""
        with pytest.raises(ValueError, match="beta_per_joule must be finite and positive"):
            ThermodynamicWitness()(work_joule=1.0, beta_per_joule=beta)
