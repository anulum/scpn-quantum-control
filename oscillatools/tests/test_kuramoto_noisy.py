# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li — Stochastic (noisy) Kuramoto Euler–Maruyama integration tests
"""Multi-angle tests for the noisy Kuramoto Euler–Maruyama integrator.

Covers the single Euler–Maruyama step (its value against the closed-form drift-plus-diffusion
update, the deterministic zero-noise limit), the seeded integrator (reproducibility under a shared
seed, divergence under different seeds, the deterministic zero-diffusion limit, the settle-window
statistics and default), the physical raising of the onset by noise (more diffusion desynchronises
at fixed coupling) and the input validation of both entry points.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel import (
    NoisyKuramotoRun,
    StochasticForce,
    integrate_noisy_kuramoto,
    mean_field_force,
    networked_kuramoto_force,
    noisy_kuramoto_step,
)


def _bound_force(coupling: float) -> StochasticForce:
    return lambda theta: mean_field_force(theta, coupling)


_N = 8


class TestNoisyKuramotoStep:
    def test_matches_euler_maruyama_update(self) -> None:
        rng = np.random.default_rng(0)
        theta, omega, noise = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )
        diffusion, dt = 0.4, 0.02
        result = noisy_kuramoto_step(theta, omega, _bound_force(1.5), diffusion, dt, noise)
        expected = (
            theta
            + (omega + mean_field_force(theta, 1.5)) * dt
            + np.sqrt(2.0 * diffusion * dt) * noise
        )
        np.testing.assert_allclose(result, expected)

    def test_zero_noise_is_deterministic_drift(self) -> None:
        rng = np.random.default_rng(1)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        result = noisy_kuramoto_step(
            theta, omega, _bound_force(2.0), 0.0, 0.01, rng.standard_normal(_N)
        )
        expected = theta + (omega + mean_field_force(theta, 2.0)) * 0.01
        np.testing.assert_allclose(result, expected)

    def test_accepts_networked_force(self) -> None:
        rng = np.random.default_rng(2)
        coupling = rng.standard_normal((_N, _N))
        coupling = 0.5 * (coupling + coupling.T)
        np.fill_diagonal(coupling, 0.0)
        theta, omega, noise = (
            rng.uniform(-np.pi, np.pi, _N),
            rng.standard_normal(_N),
            rng.standard_normal(_N),
        )

        def force(phases: np.ndarray) -> np.ndarray:
            return networked_kuramoto_force(phases, coupling)

        result = noisy_kuramoto_step(theta, omega, force, 0.1, 0.01, noise)
        expected = (
            theta
            + (omega + networked_kuramoto_force(theta, coupling)) * 0.01
            + np.sqrt(2.0 * 0.1 * 0.01) * noise
        )
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        ("phases", "omega", "noise", "match"),
        [
            (np.zeros(3), np.zeros((3, 1)), np.zeros(3), "non-empty one-dimensional"),
            (np.zeros(3), np.zeros(0), np.zeros(0), "non-empty one-dimensional"),
            (np.zeros(4), np.zeros(3), np.zeros(3), "phases must have shape"),
            (np.zeros(3), np.zeros(3), np.zeros(4), "noise must have shape"),
        ],
    )
    def test_shape_validation(
        self, phases: np.ndarray, omega: np.ndarray, noise: np.ndarray, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            noisy_kuramoto_step(phases, omega, _bound_force(1.0), 0.1, 0.01, noise)

    def test_rejects_negative_diffusion(self) -> None:
        with pytest.raises(ValueError, match="diffusion must be non-negative"):
            noisy_kuramoto_step(
                np.zeros(3), np.zeros(3), _bound_force(1.0), -0.1, 0.01, np.zeros(3)
            )

    def test_rejects_non_positive_dt(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            noisy_kuramoto_step(np.zeros(3), np.zeros(3), _bound_force(1.0), 0.1, 0.0, np.zeros(3))


class TestIntegrateNoisyKuramoto:
    def test_reproducible_under_shared_seed(self) -> None:
        rng = np.random.default_rng(3)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        kwargs = dict(diffusion=0.3, dt=0.01, n_steps=200, seed=42)
        first = integrate_noisy_kuramoto(theta, omega, _bound_force(2.0), **kwargs)
        second = integrate_noisy_kuramoto(theta, omega, _bound_force(2.0), **kwargs)
        np.testing.assert_array_equal(first.order_parameter_series, second.order_parameter_series)
        np.testing.assert_array_equal(first.terminal_phases, second.terminal_phases)
        assert first.mean_order_parameter == second.mean_order_parameter

    def test_different_seed_diverges(self) -> None:
        rng = np.random.default_rng(4)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        a = integrate_noisy_kuramoto(
            theta, omega, _bound_force(2.0), diffusion=0.3, dt=0.01, n_steps=200, seed=1
        )
        b = integrate_noisy_kuramoto(
            theta, omega, _bound_force(2.0), diffusion=0.3, dt=0.01, n_steps=200, seed=2
        )
        assert not np.array_equal(a.terminal_phases, b.terminal_phases)

    def test_zero_diffusion_is_deterministic_euler(self) -> None:
        rng = np.random.default_rng(5)
        theta0, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        run = integrate_noisy_kuramoto(
            theta0, omega, _bound_force(2.5), diffusion=0.0, dt=0.01, n_steps=50, seed=999
        )
        theta = theta0.copy()
        for _ in range(50):
            theta = theta + (omega + mean_field_force(theta, 2.5)) * 0.01
        np.testing.assert_allclose(run.terminal_phases, theta)

    def test_run_structure_and_settle_default(self) -> None:
        rng = np.random.default_rng(6)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        run = integrate_noisy_kuramoto(
            theta, omega, _bound_force(2.0), diffusion=0.2, dt=0.01, n_steps=120, seed=7
        )
        assert isinstance(run, NoisyKuramotoRun)
        assert run.order_parameter_series.shape == (120,)
        assert run.terminal_phases.shape == (_N,)
        assert run.settle_steps == 60  # the default trailing half
        assert run.diffusion == 0.2
        assert 0.0 <= run.mean_order_parameter <= 1.0
        assert run.order_parameter_std >= 0.0
        expected_mean = float(run.order_parameter_series[-60:].mean())
        assert run.mean_order_parameter == pytest.approx(expected_mean)

    def test_explicit_settle_window(self) -> None:
        rng = np.random.default_rng(8)
        theta, omega = rng.uniform(-np.pi, np.pi, _N), rng.standard_normal(_N)
        run = integrate_noisy_kuramoto(
            theta,
            omega,
            _bound_force(2.0),
            diffusion=0.2,
            dt=0.01,
            n_steps=80,
            seed=3,
            settle_steps=10,
        )
        assert run.settle_steps == 10
        assert run.mean_order_parameter == pytest.approx(
            float(run.order_parameter_series[-10:].mean())
        )

    def test_noise_raises_the_onset(self) -> None:
        # At a fixed coupling above the deterministic onset, stronger noise lowers the stationary
        # coherence (the synchronisation threshold rises with diffusion).
        rng = np.random.default_rng(9)
        n = 400
        omega = np.tan(np.pi * (rng.uniform(size=n) - 0.5))  # Lorentzian, half-width 1
        omega -= omega.mean()
        theta0 = rng.uniform(-np.pi, np.pi, n)
        quiet = integrate_noisy_kuramoto(
            theta0, omega, _bound_force(3.0), diffusion=0.05, dt=0.01, n_steps=4000, seed=11
        )
        loud = integrate_noisy_kuramoto(
            theta0, omega, _bound_force(3.0), diffusion=1.5, dt=0.01, n_steps=4000, seed=11
        )
        assert loud.mean_order_parameter < quiet.mean_order_parameter

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"diffusion": 0.1, "dt": 0.0, "n_steps": 10, "seed": 0}, "dt must be positive"),
            ({"diffusion": 0.1, "dt": 0.01, "n_steps": 0, "seed": 0}, "n_steps must be positive"),
            (
                {"diffusion": -0.1, "dt": 0.01, "n_steps": 10, "seed": 0},
                "diffusion must be non-negative",
            ),
            (
                {"diffusion": 0.1, "dt": 0.01, "n_steps": 10, "seed": 0, "settle_steps": 11},
                "settle_steps must be in",
            ),
            (
                {"diffusion": 0.1, "dt": 0.01, "n_steps": 10, "seed": 0, "settle_steps": 0},
                "settle_steps must be in",
            ),
        ],
    )
    def test_run_validation(self, kwargs: dict[str, float], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            integrate_noisy_kuramoto(np.zeros(3), np.zeros(3), _bound_force(1.0), **kwargs)

    def test_rejects_mismatched_phases(self) -> None:
        with pytest.raises(ValueError, match="phases must have shape"):
            integrate_noisy_kuramoto(
                np.zeros(4),
                np.zeros(3),
                _bound_force(1.0),
                diffusion=0.1,
                dt=0.01,
                n_steps=5,
                seed=0,
            )


def test_public_symbols_exported() -> None:
    import oscillatools.accel as accel

    for symbol in (
        "NoisyKuramotoRun",
        "StochasticForce",
        "integrate_noisy_kuramoto",
        "noisy_kuramoto_step",
    ):
        assert symbol in accel.__all__
        assert hasattr(accel, symbol)
