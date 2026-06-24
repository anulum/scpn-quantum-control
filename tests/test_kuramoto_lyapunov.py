# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Networked Kuramoto Lyapunov spectrum tests
"""Multi-angle tests for the networked Kuramoto Lyapunov spectrum.

Covers the sum rule (the exponents sum to the time-averaged Jacobian trace) at a fixed point, the
zero Goldstone exponent of a marginally stable synchronised state, the known positive exponent of
the unstable anti-phase pair, the agreement of the maximal exponent with the spectrum and a leading
subset, the descending order, the reorthonormalisation-interval and transient branches, and the
input validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scpn_quantum_control.accel import (
    lyapunov_spectrum,
    maximal_lyapunov_exponent,
    networked_kuramoto_jacobian,
)


def _random_symmetric_coupling(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coupling = rng.uniform(0.3, 1.0, size=(n, n))
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    return coupling


class TestSumRuleAndGoldstone:
    def test_sum_equals_jacobian_trace_at_fixed_point(self) -> None:
        # Identical frequencies and equal phases form a fixed point, so the Jacobian (hence its
        # trace) is constant and the exponents must sum to it.
        coupling = _random_symmetric_coupling(5, 7)
        theta0 = np.zeros(5)
        omega = np.zeros(5)
        spectrum = lyapunov_spectrum(theta0, omega, coupling, 0.01, 3000)
        trace = np.trace(networked_kuramoto_jacobian(theta0, coupling))
        assert spectrum.sum() == pytest.approx(float(trace), abs=0.05)

    def test_goldstone_exponent_is_zero(self) -> None:
        # The synchronised fixed point is marginally stable: the largest exponent is the zero
        # Goldstone mode and every exponent is non-positive.
        coupling = _random_symmetric_coupling(5, 3)
        theta0 = np.zeros(5)
        omega = np.zeros(5)
        spectrum = lyapunov_spectrum(theta0, omega, coupling, 0.01, 3000)
        assert spectrum[0] == pytest.approx(0.0, abs=0.03)
        assert np.all(spectrum <= 0.05)

    def test_descending_order(self) -> None:
        coupling = _random_symmetric_coupling(4, 11)
        spectrum = lyapunov_spectrum(np.zeros(4), np.zeros(4), coupling, 0.01, 1500)
        assert np.all(np.diff(spectrum) <= 1e-9)


class TestKnownUnstableExponent:
    def test_anti_phase_pair_has_known_positive_exponent(self) -> None:
        # Two positively coupled oscillators in anti-phase form an unstable fixed point whose
        # transverse mode grows at 2k; the maximal Lyapunov exponent recovers it.
        k = 0.5
        coupling = np.array([[0.0, k], [k, 0.0]])
        theta0 = np.array([0.0, math.pi])
        omega = np.zeros(2)
        spectrum = lyapunov_spectrum(theta0, omega, coupling, 0.01, 2500)
        assert spectrum[0] == pytest.approx(2.0 * k, abs=0.05)
        # The other exponent is the zero Goldstone mode, so the pair sums to the trace 2k.
        assert spectrum.sum() == pytest.approx(2.0 * k, abs=0.05)


class TestMaximalExponent:
    def test_maximal_matches_spectrum_head(self) -> None:
        coupling = _random_symmetric_coupling(4, 5)
        theta0 = np.zeros(4)
        omega = np.zeros(4)
        spectrum = lyapunov_spectrum(theta0, omega, coupling, 0.01, 2000)
        maximal = maximal_lyapunov_exponent(theta0, omega, coupling, 0.01, 2000)
        assert maximal == pytest.approx(spectrum[0], abs=1e-9)

    def test_leading_subset_matches_full_spectrum(self) -> None:
        coupling = _random_symmetric_coupling(4, 9)
        theta0 = np.zeros(4)
        omega = np.zeros(4)
        full = lyapunov_spectrum(theta0, omega, coupling, 0.01, 2500)
        subset = lyapunov_spectrum(theta0, omega, coupling, 0.01, 2500, num_exponents=2)
        assert subset.size == 2
        np.testing.assert_allclose(subset, full[:2], atol=0.05)


class TestIntegrationControls:
    def test_reorth_interval_and_transient(self) -> None:
        # A wider reorthonormalisation interval and a discarded transient still recover the sum
        # rule at the fixed point.
        coupling = _random_symmetric_coupling(4, 2)
        theta0 = np.zeros(4)
        omega = np.zeros(4)
        spectrum = lyapunov_spectrum(
            theta0, omega, coupling, 0.01, 2000, reorth_interval=5, transient_steps=400
        )
        trace = np.trace(networked_kuramoto_jacobian(theta0, coupling))
        assert spectrum.sum() == pytest.approx(float(trace), abs=0.05)


class TestValidation:
    def test_shape_and_count_validation(self) -> None:
        coupling = _random_symmetric_coupling(3, 1)
        with pytest.raises(ValueError, match="at least one oscillator"):
            lyapunov_spectrum(np.zeros(0), np.zeros(0), np.zeros((0, 0)), 0.01, 10)
        with pytest.raises(ValueError, match="omega must have shape"):
            lyapunov_spectrum(np.zeros(3), np.zeros(2), coupling, 0.01, 10)
        with pytest.raises(ValueError, match="coupling must have shape"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), np.zeros((3, 2)), 0.01, 10)

    def test_numeric_parameter_validation(self) -> None:
        coupling = _random_symmetric_coupling(3, 4)
        with pytest.raises(ValueError, match="dt must be positive"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.0, 10)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.01, 0)
        with pytest.raises(ValueError, match=r"num_exponents must be in \[1, 3\]"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.01, 10, num_exponents=4)
        with pytest.raises(ValueError, match=r"reorth_interval must be in \[1, 10\]"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.01, 10, reorth_interval=11)
        with pytest.raises(ValueError, match="transient_steps must be non-negative"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.01, 10, transient_steps=-1)

    def test_transient_leaves_no_reorthonormalisation(self) -> None:
        coupling = _random_symmetric_coupling(3, 6)
        with pytest.raises(ValueError, match="no post-transient reorthonormalisation"):
            lyapunov_spectrum(np.zeros(3), np.zeros(3), coupling, 0.01, 10, transient_steps=10)
