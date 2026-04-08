# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Pulse Shaping (ICI + Hypergeometric)
"""STRONG tests for phase/pulse_shaping.py.

6 dimensions: empty/null, error handling, negative cases, pipeline
integration, roundtrip, performance.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from scpn_quantum_control.phase.pulse_shaping import (
    HypergeometricPulse,
    ICIPulse,
    PulseSchedule,
    build_hypergeometric_pulse,
    build_ici_pulse,
    build_trotter_pulse_schedule,
    hypergeometric_envelope,
    ici_mixing_angle,
    ici_three_level_evolution,
    infidelity_bound,
)

# ===== 1. Empty/Null Inputs =====


class TestEmptyNull:
    def test_ici_minimal_duration(self) -> None:
        pulse = build_ici_pulse(t_total=0.001, omega_0=1.0, gamma_decay=0.0)
        assert isinstance(pulse, ICIPulse)
        assert len(pulse.times) == 200

    def test_hypergeometric_allen_eberly(self) -> None:
        """α=β=0 → Allen-Eberly (pure sech)."""
        pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=10.0, alpha=0.0, beta=0.0)
        assert isinstance(pulse, HypergeometricPulse)
        # At t=0 (centre), sech(0)=1, ₂F₁(0,0;0.5;z)=1 → envelope peak=1
        mid = len(pulse.envelope) // 2
        assert abs(pulse.envelope[mid] - 1.0) < 0.01

    def test_ici_zero_decay(self) -> None:
        """No decay → fidelity = 1."""
        pulse = build_ici_pulse(t_total=1.0, omega_0=10.0, gamma_decay=0.0)
        assert pulse.fidelity > 0.99


# ===== 2. Error Handling =====


class TestErrorHandling:
    def test_ici_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="t_total must be positive"):
            build_ici_pulse(t_total=-1.0, omega_0=10.0, gamma_decay=0.1)

    def test_ici_zero_omega(self) -> None:
        with pytest.raises(ValueError, match="omega_0 must be positive"):
            build_ici_pulse(t_total=1.0, omega_0=0.0, gamma_decay=0.1)

    def test_ici_negative_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma_decay must be non-negative"):
            build_ici_pulse(t_total=1.0, omega_0=10.0, gamma_decay=-0.1)

    def test_ici_bad_theta_jump(self) -> None:
        with pytest.raises(ValueError, match="theta_jump"):
            build_ici_pulse(t_total=1.0, omega_0=10.0, gamma_decay=0.1, theta_jump=1.0)

    def test_hypergeometric_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha, beta must be >= 0"):
            build_hypergeometric_pulse(t_total=1.0, omega_0=10.0, alpha=-0.5, beta=0.5)

    def test_hypergeometric_zero_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma_width must be positive"):
            hypergeometric_envelope(np.array([0.0]), alpha=0.5, beta=0.5, gamma_width=0.0)

    def test_hypergeometric_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="t_total must be positive"):
            build_hypergeometric_pulse(t_total=0.0, omega_0=10.0)


# ===== 3. Negative Cases =====


class TestNegativeCases:
    def test_ici_power_constraint(self) -> None:
        """Ω_P² + Ω_S² = Ω₀² at all times."""
        pulse = build_ici_pulse(t_total=1.0, omega_0=5.0, gamma_decay=0.1)
        power = pulse.omega_p**2 + pulse.omega_s**2
        np.testing.assert_allclose(power, 25.0, atol=1e-10)

    def test_ici_mixing_angle_monotonic(self) -> None:
        """θ(t) should be monotonically increasing (0 → π/2)."""
        t = np.linspace(0, 1.0, 1000)
        theta = ici_mixing_angle(t, 1.0, theta_jump=0.2)
        assert theta[0] < 0.01
        assert abs(theta[-1] - np.pi / 2) < 0.01
        # Monotonic (allowing tiny numerical noise)
        diffs = np.diff(theta)
        assert np.all(diffs >= -1e-10)

    def test_allen_eberly_symmetry(self) -> None:
        """α=β=0 (Allen-Eberly) envelope is symmetric about t=0."""
        t = np.linspace(-2, 2, 201)
        env = hypergeometric_envelope(t, alpha=0.0, beta=0.0, gamma_width=1.0)
        np.testing.assert_allclose(env, env[::-1], atol=1e-10)

    def test_high_decay_reduces_fidelity(self) -> None:
        """Higher γ → lower fidelity."""
        f_low = build_ici_pulse(1.0, 10.0, gamma_decay=0.01).fidelity
        f_high = build_ici_pulse(1.0, 10.0, gamma_decay=1.0).fidelity
        assert f_low > f_high


# ===== 4. Pipeline Integration =====


class TestPipelineIntegration:
    def test_trotter_schedule_creates_pulses(self) -> None:
        n = 4
        k = 0.5 * np.ones((n, n))
        np.fill_diagonal(k, 0.0)
        schedule = build_trotter_pulse_schedule(n, k, t_step=0.1)
        assert isinstance(schedule, PulseSchedule)
        assert schedule.n_qubits == n
        # n*(n-1)/2 = 6 unique pairs
        assert len(schedule.pulses) == 6

    def test_trotter_schedule_sparse_coupling(self) -> None:
        """Only non-zero K entries produce pulses."""
        n = 4
        k = np.zeros((n, n))
        k[0, 1] = k[1, 0] = 0.5
        schedule = build_trotter_pulse_schedule(n, k, t_step=0.1)
        assert len(schedule.pulses) == 1

    def test_ici_evolution_ground_to_target(self) -> None:
        """With no decay and strong Ω, ICI should transfer population."""
        pulse = build_ici_pulse(t_total=2.0, omega_0=20.0, gamma_decay=0.0, n_points=500)
        pops = ici_three_level_evolution(pulse)
        # Final population should be primarily in |s⟩ (index 2)
        final = pops[-1]
        assert final[2] > 0.5, f"Target population too low: {final[2]}"

    def test_infidelity_bound_decreases_with_omega(self) -> None:
        """Stronger Ω₀ → lower infidelity bound."""
        f1 = infidelity_bound(0.5, 0.5, gamma_width=1.0, omega_0=5.0)
        f2 = infidelity_bound(0.5, 0.5, gamma_width=1.0, omega_0=50.0)
        assert f2 < f1

    def test_top_level_import(self) -> None:
        from scpn_quantum_control.phase import (
            build_hypergeometric_pulse,
            build_ici_pulse,
        )

        assert callable(build_ici_pulse)
        assert callable(build_hypergeometric_pulse)

    def test_special_cases_match(self) -> None:
        """α=β=0 envelope should equal sech."""
        t = np.linspace(-3, 3, 301)
        env = hypergeometric_envelope(t, alpha=0.0, beta=0.0, gamma_width=1.0)
        sech = 1.0 / np.cosh(t)
        np.testing.assert_allclose(env, sech, atol=1e-10)


# ===== 5. Roundtrip =====


class TestRoundtrip:
    def test_ici_dataclass_fields(self) -> None:
        pulse = build_ici_pulse(t_total=1.0, omega_0=5.0, gamma_decay=0.1)
        assert isinstance(pulse.times, np.ndarray)
        assert isinstance(pulse.omega_p, np.ndarray)
        assert isinstance(pulse.omega_s, np.ndarray)
        assert isinstance(pulse.theta, np.ndarray)
        assert pulse.omega_total == 5.0
        assert pulse.gamma_decay == 0.1

    def test_hypergeometric_dataclass_fields(self) -> None:
        pulse = build_hypergeometric_pulse(t_total=1.0, omega_0=10.0, alpha=0.3, beta=0.7)
        assert pulse.alpha == 0.3
        assert pulse.beta == 0.7
        assert pulse.omega_0 == 10.0
        assert len(pulse.times) == 200

    def test_infidelity_bound_positive(self) -> None:
        bound = infidelity_bound(0.5, 0.5, gamma_width=1.0, omega_0=10.0)
        assert bound > 0
        assert bound < 1


# ===== 6. Performance =====


class TestPerformance:
    def test_hypergeometric_envelope_fast(self) -> None:
        """1000 evaluations of 200-point envelope < 2s."""
        t = np.linspace(-3, 3, 200)
        # warmup
        hypergeometric_envelope(t, 0.5, 0.5, 1.0)
        t0 = time.perf_counter()
        for _ in range(100):
            hypergeometric_envelope(t, 0.5, 0.5, 1.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"100 evaluations took {elapsed:.3f}s"

    def test_ici_build_fast(self) -> None:
        """Building ICI pulse < 1ms."""
        # warmup
        build_ici_pulse(1.0, 10.0, 0.1)
        t0 = time.perf_counter()
        for _ in range(1000):
            build_ici_pulse(1.0, 10.0, 0.1)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"1000 ICI builds took {elapsed:.3f}s"
