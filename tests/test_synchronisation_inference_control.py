# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — synchronisation inference and control under held-out disturbances
"""Inference reports what synchronised data cannot decide; control reports where it loses.

Oracles: synthetic two-oscillator ground truth, a phase-locked observation whose
unidentified direction follows from the design, an RK4 integration written here,
and held-out plant disturbances the control was never designed for. Nothing here
supports a biological or plant efficacy claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_bayesian_inference import infer_network_bayesian
from oscillatools.accel.kuramoto_network_control import (
    ControlDisturbance,
    evaluate_control_under_disturbances,
    optimise_network_control,
)

_DT = 0.01
_NOISE = 0.04


def _pair(n_steps: int, seed: int) -> NDArray[np.float64]:
    """Simulate two noisy oscillators with directed couplings ``K_01 = 0.6`` and ``K_10 = 0.3``."""
    rng = np.random.default_rng(seed)
    omega = np.array([0.4, -0.3])
    coupling = np.array([[0.0, 0.6], [0.3, 0.0]])
    phases = np.empty((n_steps + 1, 2))
    phases[0] = rng.uniform(0.0, 2.0 * np.pi, size=2)
    scale = np.sqrt(_NOISE * _DT)
    for step in range(n_steps):
        difference = phases[step][None, :] - phases[step][:, None]
        drift = omega + np.sum(coupling * np.sin(difference), axis=1)
        phases[step + 1] = phases[step] + _DT * drift + scale * rng.standard_normal(2)
    return phases


def test_synchronisation_inference_control_01_identifiable_pair_is_recovered() -> None:
    """Both frequencies and both directed couplings lie within four declared deviations."""
    posterior = infer_network_bayesian(_pair(60000, seed=1), _DT)
    assert posterior.identifiable
    truth_coupling = np.array([[0.0, 0.6], [0.3, 0.0]])
    off = ~np.eye(2, dtype=bool)
    assert np.all(
        np.abs(posterior.coupling - truth_coupling)[off] <= 4.0 * posterior.coupling_std[off]
    )
    assert np.all(
        np.abs(posterior.frequencies - np.array([0.4, -0.3])) <= 4.0 * posterior.frequency_std
    )


def test_synchronisation_inference_control_02_phase_locked_data_cannot_separate_k_from_omega() -> (
    None
):
    """With a constant phase difference only ``ω_0 + K_01 sin δ`` is observed.

    The design ``[1, sin δ]`` is constant, so the direction ``(sin δ, -1)`` in
    ``[ω_0, K_01]`` is unidentified and keeps its prior; no unique coupling is
    claimed, while the observed combination is still determined.
    """
    rng = np.random.default_rng(2)
    steps, delta = 20000, 0.7
    phases = np.zeros((steps + 1, 2))
    increments = _DT * 0.5 + np.sqrt(_NOISE * _DT) * rng.standard_normal(steps)
    phases[1:, 0] = np.cumsum(increments)
    phases[:, 1] = phases[:, 0] + delta
    posterior = infer_network_bayesian(phases, _DT)
    assert not posterior.identifiable and posterior.data_rank[0] == 1
    direction = posterior.unidentified_directions[0][:, 0]
    expected = np.array([np.sin(delta), -1.0]) / np.hypot(np.sin(delta), 1.0)
    assert abs(abs(float(direction @ expected)) - 1.0) < 1e-8
    covariance = posterior.parameter_covariances[0]
    assert float(direction @ covariance @ direction) > 0.9e6
    observed = np.array([1.0, np.sin(delta)])
    combination = observed @ np.array([posterior.frequencies[0], posterior.coupling[0, 1]])
    assert combination == pytest.approx(0.5, abs=4.0 * np.sqrt(observed @ covariance @ observed))


def _rk4_order(
    phases: NDArray[np.float64],
    control: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Integrate the controlled network with RK4 written here and return ``r`` after each step."""

    def field(theta: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``ω + Σ K sin(θ_j - θ_i) + u``."""
        return np.asarray(
            omega + np.sum(coupling * np.sin(theta[None, :] - theta[:, None]), axis=1) + u,
            dtype=np.float64,
        )

    theta = phases.copy()
    orders = []
    for u in control:
        k1 = field(theta, u)
        k2 = field(theta + 0.5 * _STEP * k1, u)
        k3 = field(theta + 0.5 * _STEP * k2, u)
        k4 = field(theta + _STEP * k3, u)
        theta = theta + _STEP / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        orders.append(abs(np.mean(np.exp(1j * theta))))
    return np.array(orders)


_STEP = 0.05
_PHASES = np.array([0.0, 0.4, 0.9, 1.3])
_OMEGA = np.array([0.1, -0.05, 0.02, -0.08])
_COUPLING = np.full((4, 4), 0.5) - 0.5 * np.eye(4)


def test_synchronisation_inference_control_03_held_out_disturbances_report_losing_cases() -> None:
    """A desynchronising control keeps its amplitude bound on every plant and names its losses."""
    control, history = optimise_network_control(
        _PHASES,
        _OMEGA,
        _COUPLING,
        _STEP,
        80,
        control_weight=0.01,
        learning_rate=5.0,
        n_iterations=400,
    )
    assert history[-1] < 0.5 * history[0]
    rng = np.random.default_rng(3)
    off = 1.0 - np.eye(4)
    disturbances = [
        ControlDisturbance(
            rng.normal(0.0, 0.05, 4), rng.normal(0.0, 0.05, (4, 4)) * off, rng.normal(0.0, 0.05, 4)
        )
        for _ in range(5)
    ]
    disturbances.append(ControlDisturbance(np.zeros(4), 6.0 * off, np.zeros(4)))
    report = evaluate_control_under_disturbances(
        _PHASES,
        control,
        _OMEGA,
        _COUPLING,
        _STEP,
        disturbances=disturbances,
        amplitude_bound=8.0,
        target_order=0.5,
    )
    assert report.constraint_satisfied and report.max_amplitude <= 8.0
    assert report.nominal.succeeded
    assert len(report.held_out) == len(disturbances)
    # Every outcome is recomputed with an independent RK4; losing cases are exactly the failures.
    expected_losses = []
    for index, item in enumerate(disturbances):
        start = _PHASES + item.phase_shift
        omega = _OMEGA + item.omega_shift
        coupling = _COUPLING + item.coupling_shift
        controlled = _rk4_order(start, control, omega, coupling)
        free = _rk4_order(start, np.zeros_like(control), omega, coupling)
        outcome = report.held_out[index]
        assert outcome.mean_order_squared == pytest.approx(np.mean(controlled**2), rel=1e-10)
        assert outcome.uncontrolled_mean_order_squared == pytest.approx(
            np.mean(free**2), rel=1e-10
        )
        if not (np.mean(controlled**2) < np.mean(free**2) and controlled[-1] <= 0.5):
            expected_losses.append(index)
    assert list(report.losing_cases) == expected_losses
    assert 5 in report.losing_cases  # strong extra coupling resynchronises the network
    # A bound below the series' amplitude is reported as violated, not hidden.
    tight = evaluate_control_under_disturbances(
        _PHASES,
        control,
        _OMEGA,
        _COUPLING,
        _STEP,
        disturbances=(),
        amplitude_bound=0.1,
        target_order=0.5,
    )
    assert not tight.constraint_satisfied and tight.held_out == () and tight.losing_cases == ()


def test_control_evaluation_rejects_malformed_inputs() -> None:
    """Bounds, targets and disturbance shapes are validated before any integration."""
    control = np.zeros((5, 4))
    common = {"amplitude_bound": 1.0, "target_order": 0.5}
    with pytest.raises(ValueError, match="amplitude_bound"):
        evaluate_control_under_disturbances(
            _PHASES,
            control,
            _OMEGA,
            _COUPLING,
            _STEP,
            disturbances=(),
            amplitude_bound=0.0,
            target_order=0.5,
        )
    with pytest.raises(ValueError, match="target_order"):
        evaluate_control_under_disturbances(
            _PHASES,
            control,
            _OMEGA,
            _COUPLING,
            _STEP,
            disturbances=(),
            amplitude_bound=1.0,
            target_order=1.5,
        )
    with pytest.raises(ValueError, match="disturbance 0"):
        evaluate_control_under_disturbances(
            _PHASES,
            control,
            _OMEGA,
            _COUPLING,
            _STEP,
            disturbances=[ControlDisturbance(np.zeros(3), np.zeros((4, 4)), np.zeros(4))],
            amplitude_bound=common["amplitude_bound"],
            target_order=common["target_order"],
        )
