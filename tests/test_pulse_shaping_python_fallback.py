# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — pulse_shaping Python fallback parity tests
"""Tests for the pure-Python fallback branches in `phase.pulse_shaping`.

The module runs on the Rust accelerator when the ``scpn_quantum_engine``
wheel is importable and falls back to equivalent Python implementations
otherwise. The fast path is exercised by the existing
``tests/test_pulse_shaping.py`` suite; this file deliberately
monkey-patches ``_HAS_RUST = False`` so the Python fallback branches
are executed and their numerical agreement with the Rust result is
verified.

Also covers the remaining no-Rust validators and edge cases that the
existing suite did not reach:

* ``build_hypergeometric_pulse`` rejects ``omega_0 <= 0``.
* ``build_trotter_pulse_schedule`` handles a zero coupling matrix
  without a divide-by-zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import pulse_shaping

_NEEDS_RUST = pytest.mark.skipif(
    not pulse_shaping._HAS_RUST,
    reason="Rust accel wheel not importable — fallback is the default, no parity to check",
)


def _with_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pulse_shaping, "_HAS_RUST", False)


# ---------------------------------------------------------------------------
# ici_mixing_angle — Python fallback vs Rust parity
# ---------------------------------------------------------------------------


class TestIciMixingAnglePythonFallback:
    @_NEEDS_RUST
    @pytest.mark.parametrize("theta_jump", [0.1, 0.3, 0.5, np.pi / 5])
    @pytest.mark.parametrize("t_total", [0.1, 1.0, 7.5])
    def test_parity_with_rust(
        self,
        monkeypatch: pytest.MonkeyPatch,
        t_total: float,
        theta_jump: float,
    ) -> None:
        t = np.linspace(0.0, t_total, 64)
        rust_result = pulse_shaping.ici_mixing_angle(t, t_total, theta_jump)
        _with_python_fallback(monkeypatch)
        python_result = pulse_shaping.ici_mixing_angle(t, t_total, theta_jump)
        np.testing.assert_allclose(python_result, rust_result, atol=1e-9)

    def test_fallback_boundary_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _with_python_fallback(monkeypatch)
        t = np.linspace(0.0, 1.0, 101)
        theta = pulse_shaping.ici_mixing_angle(t, 1.0, theta_jump=0.3)
        # Segment 1 start: θ(0) = 0 exactly.
        assert theta[0] == pytest.approx(0.0, abs=1e-12)
        # Segment 3 end: θ(t_total) = π/2 exactly.
        assert theta[-1] == pytest.approx(np.pi / 2, abs=1e-12)

    def test_fallback_shape_preserved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _with_python_fallback(monkeypatch)
        t = np.linspace(0.0, 2.0, 32)
        theta = pulse_shaping.ici_mixing_angle(t, 2.0, theta_jump=0.25)
        assert theta.shape == t.shape


# ---------------------------------------------------------------------------
# ici_three_level_evolution — Python fallback vs Rust parity
# ---------------------------------------------------------------------------


class TestIciThreeLevelEvolutionPythonFallback:
    @_NEEDS_RUST
    @pytest.mark.parametrize("gamma", [0.0, 0.05, 0.2])
    def test_parity_with_rust(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gamma: float,
    ) -> None:
        pulse = pulse_shaping.build_ici_pulse(
            t_total=1.0,
            omega_0=10.0,
            gamma_decay=gamma,
            n_points=64,
            theta_jump=0.3,
        )
        rust = pulse_shaping.ici_three_level_evolution(pulse, gamma_decay=gamma)
        _with_python_fallback(monkeypatch)
        python = pulse_shaping.ici_three_level_evolution(pulse, gamma_decay=gamma)
        # Forward-Euler Python fallback is first-order accurate; agreement to
        # ~1 % is expected for moderate step counts, not bit-exact.
        np.testing.assert_allclose(python, rust, atol=5e-2)

    def test_fallback_populations_start_in_ground(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _with_python_fallback(monkeypatch)
        pulse = pulse_shaping.build_ici_pulse(
            t_total=0.5,
            omega_0=8.0,
            gamma_decay=0.1,
            n_points=32,
        )
        populations = pulse_shaping.ici_three_level_evolution(pulse)
        # Row 0 is the initial state: all probability in |g⟩.
        np.testing.assert_allclose(populations[0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_fallback_populations_trace_is_renormalised(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _with_python_fallback(monkeypatch)
        pulse = pulse_shaping.build_ici_pulse(
            t_total=1.0,
            omega_0=10.0,
            gamma_decay=0.0,
            n_points=200,
        )
        populations = pulse_shaping.ici_three_level_evolution(pulse)
        # The Python fallback is forward-Euler plus a trace re-normalisation
        # at every step — so ∑p_i ≡ 1 exactly, even though individual p_i
        # can briefly violate the 0 ≤ p ≤ 1 bound when the step size is
        # coarse. The Rust path is a higher-order integrator and is the
        # production choice; the Python branch exists as a correctness
        # fallback only.
        sums = populations.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-9)


# ---------------------------------------------------------------------------
# hypergeometric_envelope — Python fallback vs Rust parity
# ---------------------------------------------------------------------------


class TestHypergeometricEnvelopePythonFallback:
    @_NEEDS_RUST
    def test_parity_with_rust_pure_sech(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # α = β = 0: both paths reduce to pure sech, agreement is bit-exact.
        t = np.linspace(-1.0, 1.0, 64)
        rust = pulse_shaping.hypergeometric_envelope(t, 0.0, 0.0, gamma_width=3.0)
        _with_python_fallback(monkeypatch)
        python = pulse_shaping.hypergeometric_envelope(t, 0.0, 0.0, gamma_width=3.0)
        np.testing.assert_allclose(python, rust, atol=1e-10)

    @_NEEDS_RUST
    @pytest.mark.parametrize(("alpha", "beta"), [(0.5, 0.5), (1.0, 0.5)])
    def test_fallback_runs_with_nontrivial_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
        alpha: float,
        beta: float,
    ) -> None:
        # For non-zero α, β the Rust and scipy hyp2f1 implementations
        # follow different truncation schemes; they agree on the overall
        # shape but not bit-for-bit. We assert the Python path produces a
        # well-behaved envelope (not NaN, bounded, symmetric about t=0
        # for symmetric parameters) rather than bit-exact equality.
        _with_python_fallback(monkeypatch)
        t = np.linspace(-1.0, 1.0, 64)
        python = pulse_shaping.hypergeometric_envelope(t, alpha, beta, gamma_width=3.0)
        # Fallback returns a well-behaved real envelope — the formula uses
        # ``(1 + tanh(γt))/2`` which breaks left-right symmetry of t, so
        # we only assert shape + finiteness + positivity, not symmetry.
        assert python.shape == t.shape
        assert np.all(np.isfinite(python))
        assert np.all(python > 0)

    def test_fallback_pure_sech_for_zero_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _with_python_fallback(monkeypatch)
        t = np.linspace(-2.0, 2.0, 41)
        env = pulse_shaping.hypergeometric_envelope(t, 0.0, 0.0, gamma_width=1.5)
        # α = β = 0 → Allen-Eberly → pure sech(γt).
        expected = 1.0 / np.cosh(1.5 * t)
        np.testing.assert_allclose(env, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Remaining validators and edge cases
# ---------------------------------------------------------------------------


class TestBuildHypergeometricPulseValidators:
    def test_rejects_nonpositive_omega_0(self) -> None:
        with pytest.raises(ValueError, match="omega_0 must be positive"):
            pulse_shaping.build_hypergeometric_pulse(
                t_total=1.0,
                omega_0=0.0,
            )

    def test_rejects_negative_omega_0(self) -> None:
        with pytest.raises(ValueError, match="omega_0 must be positive"):
            pulse_shaping.build_hypergeometric_pulse(
                t_total=1.0,
                omega_0=-3.0,
            )


class TestBuildTrotterPulseScheduleZeroK:
    def test_zero_k_matrix_does_not_divide_by_zero(self) -> None:
        # k_max < 1e-15 triggers the `k_max = 1.0` safety branch at line 406.
        # Behaviour: every |K[i,j]| < 1e-10 is skipped, so no pulses are
        # produced and the schedule is empty.
        k_zero = np.zeros((3, 3))
        schedule = pulse_shaping.build_trotter_pulse_schedule(
            n_qubits=3,
            k_matrix=k_zero,
            t_step=0.3,
            omega_0=5.0,
        )
        assert schedule.n_qubits == 3
        assert len(schedule.pulses) == 0
