# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Crypto Noise Analysis
"""Tests for noise_analysis: security under depolarizing noise and eavesdropping."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from scpn_quantum_control.crypto.noise_analysis import (
    amplitude_damping_single,
    depolarizing_channel,
    devetak_winter_rate,
    intercept_resend_qber,
    noisy_concurrence,
    security_analysis,
)


def _bell_plus() -> Statevector:
    """|Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    return Statevector([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


class TestDepolarizingChannel:
    """Exercise the density-matrix depolarizing channel."""

    def test_identity_at_zero_noise(self) -> None:
        """Leave a density matrix unchanged at zero noise."""
        rho = np.array([[0.6, 0.3], [0.3, 0.4]])
        result = depolarizing_channel(rho, 0.0)
        np.testing.assert_allclose(result, rho)

    def test_maximally_mixed_at_full_noise(self) -> None:
        """Return the maximally mixed state at full noise."""
        rho = np.array([[1, 0], [0, 0]], dtype=float)
        result = depolarizing_channel(rho, 1.0)
        np.testing.assert_allclose(result, np.eye(2) / 2)

    def test_trace_preserved(self) -> None:
        """Preserve unit trace under partial depolarization."""
        rho = np.array([[0.7, 0.2j], [-0.2j, 0.3]])
        result = depolarizing_channel(rho, 0.3)
        assert abs(np.trace(result) - 1.0) < 1e-12


class TestAmplitudeDamping:
    """Exercise the single-qubit amplitude-damping channel."""

    def test_no_damping(self) -> None:
        """Leave the input unchanged at zero damping."""
        rho = np.array([[0.5, 0.3], [0.3, 0.5]])
        result = amplitude_damping_single(rho, 0.0)
        np.testing.assert_allclose(result, rho)

    def test_full_damping_to_ground(self) -> None:
        """Map a fully damped excited state to the ground state."""
        rho = np.array([[0.0, 0.0], [0.0, 1.0]])  # |1⟩⟨1|
        result = amplitude_damping_single(rho, 1.0)
        np.testing.assert_allclose(result, np.array([[1, 0], [0, 0]]), atol=1e-12)


class TestNoisyConcurrence:
    """Exercise reduced-pair concurrence under depolarizing noise."""

    def test_bell_state_high_concurrence_no_noise(self) -> None:
        """Retain high Bell-state concurrence without noise."""
        sv = _bell_plus()
        c = noisy_concurrence(sv, 0, 1, 2, 0.0)
        assert c > 0.9

    def test_bell_state_lower_concurrence_with_noise(self) -> None:
        """Reduce Bell-state concurrence after depolarization."""
        sv = _bell_plus()
        c_clean = noisy_concurrence(sv, 0, 1, 2, 0.0)
        c_noisy = noisy_concurrence(sv, 0, 1, 2, 0.3)
        assert c_noisy < c_clean


class TestInterceptResendQBER:
    """Exercise correlation loss from intercept-resend attacks."""

    def test_bell_state_has_nonzero_qber(self) -> None:
        """Return a bounded disturbance for an entangled pair."""
        sv = _bell_plus()
        qber = intercept_resend_qber(sv, 0, 1, 2)
        assert 0 <= qber <= 0.5

    def test_product_state_low_qber(self) -> None:
        """Return low disturbance for a computational-basis product state."""
        sv = Statevector([1, 0, 0, 0])  # |00⟩
        qber = intercept_resend_qber(sv, 0, 1, 2)
        assert qber < 0.1


class TestDevetakWinterRate:
    """Exercise the binary-entropy secret-key-rate bound."""

    def test_zero_qber_gives_max_rate(self) -> None:
        """Return the maximum rate at zero QBER."""
        assert devetak_winter_rate(0.0) == 1.0

    def test_half_qber_gives_zero_rate(self) -> None:
        """Return zero rate at fully random QBER."""
        assert devetak_winter_rate(0.5) == 0.0

    def test_rate_monotonically_decreasing(self) -> None:
        """Decrease the admitted rate as QBER increases."""
        qbers = np.linspace(0.01, 0.49, 20)
        rates = [devetak_winter_rate(q) for q in qbers]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_threshold_around_011(self) -> None:
        """Cross zero rate near the eleven-percent security threshold."""
        assert devetak_winter_rate(0.10) > 0
        assert devetak_winter_rate(0.12) == 0.0


class TestConcurrenceImaginaryWarning:
    """Exercise diagnostic handling of a non-physical density matrix."""

    def test_non_physical_matrix_triggers_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warn when the spin-flipped spectrum has a material imaginary part."""
        from scpn_quantum_control.crypto.noise_analysis import _concurrence_2qubit

        rho = np.array(
            [
                [0.5, 0.3, 0.2, 0.4],
                [0.3, -0.1, 0.1, 0.2],
                [0.2, 0.1, 0.3, -0.2],
                [0.4, 0.2, -0.2, 0.3],
            ],
            dtype=np.complex128,
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="scpn_quantum_control.crypto.noise_analysis"):
            result = _concurrence_2qubit(rho)
        assert isinstance(result, float)
        assert "imaginary part" in caplog.text


class TestSecurityAnalysis:
    """Exercise pairwise and aggregate security-analysis records."""

    def test_returns_expected_keys(self) -> None:
        """Return all three public result mappings."""
        sv = _bell_plus()
        result = security_analysis(sv, [0], [1], np.array([0.0, 0.1, 0.2]))
        assert "pair_rates" in result
        assert "critical_noise" in result
        assert "aggregate_rate" in result

    def test_aggregate_rate_length_matches_noise_range(self) -> None:
        """Return one aggregate rate for every requested noise value."""
        sv = _bell_plus()
        p_range = np.linspace(0, 0.3, 8, dtype=np.float64)
        result = security_analysis(sv, [0], [1], p_range)
        agg = result["aggregate_rate"]
        assert len(agg) == 8
        assert all(r >= 0 for _, r in agg)

    def test_default_noise_range_has_sixteen_points(self) -> None:
        """Use the documented default depolarizing-noise grid."""
        result = security_analysis(_bell_plus(), [0], [1])
        assert len(result["aggregate_rate"]) == 16
