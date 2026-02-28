"""Tests for noise_analysis: security under depolarizing noise and eavesdropping."""

from __future__ import annotations

import numpy as np
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
    def test_identity_at_zero_noise(self):
        rho = np.array([[0.6, 0.3], [0.3, 0.4]])
        result = depolarizing_channel(rho, 0.0)
        np.testing.assert_allclose(result, rho)

    def test_maximally_mixed_at_full_noise(self):
        rho = np.array([[1, 0], [0, 0]], dtype=float)
        result = depolarizing_channel(rho, 1.0)
        np.testing.assert_allclose(result, np.eye(2) / 2)

    def test_trace_preserved(self):
        rho = np.array([[0.7, 0.2j], [-0.2j, 0.3]])
        result = depolarizing_channel(rho, 0.3)
        assert abs(np.trace(result) - 1.0) < 1e-12


class TestAmplitudeDamping:
    def test_no_damping(self):
        rho = np.array([[0.5, 0.3], [0.3, 0.5]])
        result = amplitude_damping_single(rho, 0.0)
        np.testing.assert_allclose(result, rho)

    def test_full_damping_to_ground(self):
        rho = np.array([[0.0, 0.0], [0.0, 1.0]])  # |1⟩⟨1|
        result = amplitude_damping_single(rho, 1.0)
        np.testing.assert_allclose(result, np.array([[1, 0], [0, 0]]), atol=1e-12)


class TestNoisyConcurrence:
    def test_bell_state_high_concurrence_no_noise(self):
        sv = _bell_plus()
        c = noisy_concurrence(sv, 0, 1, 2, 0.0)
        assert c > 0.9

    def test_bell_state_lower_concurrence_with_noise(self):
        sv = _bell_plus()
        c_clean = noisy_concurrence(sv, 0, 1, 2, 0.0)
        c_noisy = noisy_concurrence(sv, 0, 1, 2, 0.3)
        assert c_noisy < c_clean


class TestInterceptResendQBER:
    def test_bell_state_has_nonzero_qber(self):
        sv = _bell_plus()
        qber = intercept_resend_qber(sv, 0, 1, 2)
        assert 0 <= qber <= 0.5

    def test_product_state_low_qber(self):
        sv = Statevector([1, 0, 0, 0])  # |00⟩
        qber = intercept_resend_qber(sv, 0, 1, 2)
        assert qber < 0.1


class TestDevetakWinterRate:
    def test_zero_qber_gives_max_rate(self):
        assert devetak_winter_rate(0.0) == 1.0

    def test_half_qber_gives_zero_rate(self):
        assert devetak_winter_rate(0.5) == 0.0

    def test_rate_monotonically_decreasing(self):
        qbers = np.linspace(0.01, 0.49, 20)
        rates = [devetak_winter_rate(q) for q in qbers]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_threshold_around_011(self):
        assert devetak_winter_rate(0.10) > 0
        assert devetak_winter_rate(0.12) == 0.0


class TestSecurityAnalysis:
    def test_returns_expected_keys(self):
        sv = _bell_plus()
        result = security_analysis(sv, [0], [1], np.array([0.0, 0.1, 0.2]))
        assert "pair_rates" in result
        assert "critical_noise" in result
        assert "aggregate_rate" in result

    def test_aggregate_rate_length_matches_noise_range(self):
        sv = _bell_plus()
        p_range = np.linspace(0, 0.3, 8)
        result = security_analysis(sv, [0], [1], p_range)
        agg = result["aggregate_rate"]
        assert len(agg) == 8
        assert all(r >= 0 for _, r in agg)
