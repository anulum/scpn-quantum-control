# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the polyglot stochastic (Euler–Maruyama) networked-Kuramoto trajectory
r"""Tests for :mod:`oscillatools.accel.networked_noisy`.

The accelerated networked-force Euler–Maruyama integrator is checked in the deterministic limit
(zero noise recovers the plain Euler flow, and additionally the free rotor at zero coupling — both to
machine precision), for reproducibility under a fixed seed, for the physical noise-raises-decoherence
ordering, and for bit-identity with the general
:func:`~oscillatools.accel.kuramoto_noisy.integrate_noisy_kuramoto` on the same networked force and seed.
The Rust → Julia → Python floor tier chain is checked for tolerance-parity (all tiers consume the same
pre-generated Wiener increments) and its fail-closed fall-through, and every input-validation branch is
covered.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import oscillatools.accel.dispatcher as dispatcher_module
import oscillatools.accel.networked_noisy as noisy_module
from oscillatools.accel.kuramoto_noisy import NoisyKuramotoRun, integrate_noisy_kuramoto
from oscillatools.accel.networked_noisy import (
    _force,
    last_networked_noisy_trajectory_tier_used,
    networked_noisy_trajectory,
)


def _problem(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(theta0, omega, coupling)`` with a symmetric zero-diagonal coupling."""
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    omega = rng.normal(0.0, 1.0, count)
    coupling = rng.normal(0.0, 0.4, (count, count))
    coupling = coupling + coupling.T
    np.fill_diagonal(coupling, 0.0)
    return theta0, omega, coupling


# --------------------------------------------------------------------------- ground-truth physics


def test_zero_noise_recovers_the_deterministic_euler_flow() -> None:
    """With diffusion ``D = 0`` the scheme is the plain Euler map ``θ ← θ + (ω + F(θ)) dt``."""
    theta0, omega, coupling = _problem(8, 0)
    dt, n_steps = 0.01, 300
    run = networked_noisy_trajectory(theta0, omega, coupling, 0.0, dt=dt, n_steps=n_steps, seed=1)
    theta = theta0.astype(np.float64)
    expected_series = np.empty(n_steps)
    for step in range(n_steps):
        theta = theta + (omega + _force(theta, coupling)) * dt
        expected_series[step] = np.abs(np.mean(np.exp(1j * theta)))
    np.testing.assert_allclose(run.terminal_phases, theta, atol=1e-12)
    np.testing.assert_allclose(run.order_parameter_series, expected_series, atol=1e-12)


def test_zero_coupling_zero_noise_is_the_free_rotor() -> None:
    """With no coupling and no noise each oscillator advances as ``θ_j ← θ_j + ω_j dt`` exactly."""
    count = 5
    rng = np.random.default_rng(3)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    omega = rng.normal(0.0, 1.0, count)
    coupling = np.zeros((count, count))
    dt, n_steps = 0.02, 150
    run = networked_noisy_trajectory(theta0, omega, coupling, 0.0, dt=dt, n_steps=n_steps, seed=2)
    expected_terminal = theta0 + omega * (n_steps * dt)
    np.testing.assert_allclose(run.terminal_phases, expected_terminal, atol=1e-11)


def test_same_seed_reproduces_the_run() -> None:
    """Identical seeds reproduce the whole run bit-for-bit; different seeds do not."""
    theta0, omega, coupling = _problem(10, 4)
    kwargs = {"dt": 0.01, "n_steps": 200}
    first = networked_noisy_trajectory(theta0, omega, coupling, 0.5, seed=11, **kwargs)
    same = networked_noisy_trajectory(theta0, omega, coupling, 0.5, seed=11, **kwargs)
    other = networked_noisy_trajectory(theta0, omega, coupling, 0.5, seed=12, **kwargs)
    np.testing.assert_array_equal(first.order_parameter_series, same.order_parameter_series)
    np.testing.assert_array_equal(first.terminal_phases, same.terminal_phases)
    assert not np.array_equal(first.terminal_phases, other.terminal_phases)


def test_noise_lowers_the_stationary_coherence() -> None:
    """Stronger diffusion decoheres a synchronised population: mean order parameter drops."""
    count = 24
    rng = np.random.default_rng(5)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, count)
    omega = rng.normal(0.0, 0.2, count)
    coupling = np.full((count, count), 4.0 / count)
    np.fill_diagonal(coupling, 0.0)
    kwargs = {"dt": 0.01, "n_steps": 600, "seed": 7}
    quiet = networked_noisy_trajectory(theta0, omega, coupling, 0.0, **kwargs)
    loud = networked_noisy_trajectory(theta0, omega, coupling, 3.0, **kwargs)
    assert quiet.mean_order_parameter > loud.mean_order_parameter + 0.1
    assert np.all(loud.order_parameter_series >= 0.0)
    assert np.all(loud.order_parameter_series <= 1.0 + 1e-12)


# --------------------------------------------------------------------------- floor equivalence


def test_floor_is_bit_identical_to_the_general_integrator() -> None:
    """The Python floor equals :func:`integrate_noisy_kuramoto` on the same force and seed."""
    theta0, omega, coupling = _problem(9, 6)
    diffusion, dt, n_steps, seed = 0.4, 0.01, 250, 21
    noise = np.random.default_rng(seed).standard_normal((n_steps, omega.size))
    floor_series, floor_terminal = noisy_module._python_networked_noisy_trajectory(
        theta0, omega, coupling, diffusion, dt, noise
    )
    reference = integrate_noisy_kuramoto(
        theta0,
        omega,
        lambda theta: _force(theta, coupling),
        diffusion=diffusion,
        dt=dt,
        n_steps=n_steps,
        seed=seed,
    )
    np.testing.assert_array_equal(floor_series, reference.order_parameter_series)
    np.testing.assert_array_equal(floor_terminal, reference.terminal_phases)


def test_run_structure_and_metadata() -> None:
    """The returned run has the documented shapes and settle-window metadata."""
    theta0, omega, coupling = _problem(7, 8)
    n_steps = 120
    run = networked_noisy_trajectory(
        theta0, omega, coupling, 0.3, dt=0.01, n_steps=n_steps, seed=3, settle_steps=40
    )
    assert isinstance(run, NoisyKuramotoRun)
    assert run.order_parameter_series.shape == (n_steps,)
    assert run.terminal_phases.shape == (7,)
    assert run.diffusion == 0.3
    assert run.settle_steps == 40
    settle = run.order_parameter_series[n_steps - 40 :]
    assert run.mean_order_parameter == pytest.approx(float(settle.mean()))
    assert run.order_parameter_std == pytest.approx(float(settle.std()))


def test_default_settle_window_is_the_final_half() -> None:
    """When ``settle_steps`` is omitted the trailing half of the run is averaged."""
    theta0, omega, coupling = _problem(6, 9)
    run = networked_noisy_trajectory(theta0, omega, coupling, 0.2, dt=0.01, n_steps=101, seed=4)
    assert run.settle_steps == 50


# --------------------------------------------------------------------------- validation branches


def test_rejects_non_vector_omega() -> None:
    theta0, _, coupling = _problem(4, 10)
    with pytest.raises(ValueError, match="omega must be a non-empty one-dimensional array"):
        networked_noisy_trajectory(
            theta0, np.zeros((4, 1)), coupling, 0.1, dt=0.01, n_steps=10, seed=0
        )


def test_rejects_mismatched_theta0() -> None:
    _, omega, coupling = _problem(5, 11)
    with pytest.raises(ValueError, match="theta0 must match omega shape"):
        networked_noisy_trajectory(np.zeros(4), omega, coupling, 0.1, dt=0.01, n_steps=10, seed=0)


def test_rejects_non_square_coupling() -> None:
    theta0, omega, _ = _problem(5, 12)
    with pytest.raises(ValueError, match="coupling must be a square matrix"):
        networked_noisy_trajectory(
            theta0, omega, np.zeros((5, 4)), 0.1, dt=0.01, n_steps=10, seed=0
        )


def test_rejects_negative_diffusion() -> None:
    theta0, omega, coupling = _problem(4, 13)
    with pytest.raises(ValueError, match="diffusion must be non-negative"):
        networked_noisy_trajectory(theta0, omega, coupling, -0.1, dt=0.01, n_steps=10, seed=0)


def test_rejects_non_positive_dt() -> None:
    theta0, omega, coupling = _problem(4, 14)
    with pytest.raises(ValueError, match="dt must be positive"):
        networked_noisy_trajectory(theta0, omega, coupling, 0.1, dt=0.0, n_steps=10, seed=0)


def test_rejects_non_positive_n_steps() -> None:
    theta0, omega, coupling = _problem(4, 15)
    with pytest.raises(ValueError, match="n_steps must be positive"):
        networked_noisy_trajectory(theta0, omega, coupling, 0.1, dt=0.01, n_steps=0, seed=0)


def test_rejects_settle_steps_out_of_range() -> None:
    theta0, omega, coupling = _problem(4, 16)
    with pytest.raises(ValueError, match="settle_steps must be in"):
        networked_noisy_trajectory(
            theta0, omega, coupling, 0.1, dt=0.01, n_steps=10, seed=0, settle_steps=11
        )


# --------------------------------------------------------------------------- tier dispatch


class TestNetworkedNoisyTierDispatch:
    """The Rust → Julia → Python floor tier chain for the noisy forward trajectory."""

    def test_public_call_records_a_served_tier(self) -> None:
        theta0, omega, coupling = _problem(16, 20)
        networked_noisy_trajectory(theta0, omega, coupling, 0.5, dt=0.01, n_steps=100, seed=1)
        assert last_networked_noisy_trajectory_tier_used() in {"rust", "julia", "python"}

    def test_python_floor_tier_direct(self) -> None:
        theta0, omega, coupling = _problem(14, 21)
        noise = np.random.default_rng(1).standard_normal((120, 14))
        series, terminal = noisy_module._python_networked_noisy_trajectory(
            theta0, omega, coupling, 0.5, 0.01, noise
        )
        assert series.shape == (120,)
        assert terminal.shape == (14,)

    def test_rust_tier_matches_python_floor(self) -> None:
        engine = pytest.importorskip("scpn_quantum_engine")
        assert hasattr(engine, "kuramoto_noisy_trajectory")
        theta0, omega, coupling = _problem(32, 22)
        noise = np.random.default_rng(3).standard_normal((300, 32))
        rust = noisy_module._rust_networked_noisy_trajectory(
            theta0, omega, coupling, 0.4, 0.01, noise
        )
        floor = noisy_module._python_networked_noisy_trajectory(
            theta0, omega, coupling, 0.4, 0.01, noise
        )
        np.testing.assert_allclose(rust[0], floor[0], atol=1e-11)
        np.testing.assert_allclose(rust[1], floor[1], atol=1e-11)

    def test_julia_tier_matches_python_floor(self) -> None:
        pytest.importorskip("juliacall")
        theta0, omega, coupling = _problem(24, 23)
        noise = np.random.default_rng(4).standard_normal((250, 24))
        julia = noisy_module._julia_networked_noisy_trajectory(
            theta0, omega, coupling, 0.3, 0.01, noise
        )
        floor = noisy_module._python_networked_noisy_trajectory(
            theta0, omega, coupling, 0.3, 0.01, noise
        )
        np.testing.assert_allclose(julia[0], floor[0], atol=1e-10)
        np.testing.assert_allclose(julia[1], floor[1], atol=1e-10)

    def test_rust_tier_raises_when_engine_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        theta0, omega, coupling = _problem(8, 24)
        noise = np.random.default_rng(0).standard_normal((50, 8))
        with pytest.raises(ModuleNotFoundError):
            noisy_module._rust_networked_noisy_trajectory(
                theta0, omega, coupling, 0.5, 0.01, noise
            )

    def test_rust_tier_raises_when_symbol_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: object())
        theta0, omega, coupling = _problem(8, 25)
        noise = np.random.default_rng(0).standard_normal((50, 8))
        with pytest.raises(ImportError):
            noisy_module._rust_networked_noisy_trajectory(
                theta0, omega, coupling, 0.5, 0.01, noise
            )

    def test_fall_through_to_floor_when_engine_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dispatcher_module, "optional_rust_engine", lambda: None)
        monkeypatch.setitem(sys.modules, "oscillatools.accel.julia", None)
        theta0, omega, coupling = _problem(10, 26)
        run = networked_noisy_trajectory(theta0, omega, coupling, 0.5, dt=0.01, n_steps=80, seed=1)
        assert last_networked_noisy_trajectory_tier_used() == "python"
        assert run.order_parameter_series.shape == (80,)
