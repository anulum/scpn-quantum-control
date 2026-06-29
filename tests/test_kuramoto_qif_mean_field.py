# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the QIF / theta next-generation neural-mass mean field
"""Module-specific tests for :mod:`kuramoto_qif_mean_field`.

The exact reduction is validated on its analytic contracts — the fixed point annihilates
the field to machine precision, the Jacobian and the differentiable form match central
finite differences, the theta-QIF map and the conformal Kuramoto relation round-trip — and,
as the physics validation, the conformally mapped order parameter of the reduced ``(r, V)``
trajectory is checked against the order parameter of a direct large-N theta-neuron ensemble
over the whole run (residual at the finite-N sampling floor).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.kuramoto_qif_mean_field import (
    QifMeanFieldGradients,
    QifMeanFieldTrajectory,
    integrate_qif_mean_field,
    kuramoto_order_parameter_from_macro,
    macro_from_kuramoto_order_parameter,
    qif_mean_field_fixed_point,
    qif_mean_field_jacobian,
    qif_mean_field_rates,
    qif_mean_field_terminal_value_and_grad,
    qif_potential_from_theta,
    theta_from_qif_potential,
)

# Representative excitatory parameters (Montbrió–Pazó–Roxin regime).
_ETA_BAR = -5.0
_DELTA = 1.0
_COUPLING = 15.0
_CURRENT = 0.0


def _field(rate: float, potential: float) -> NDArray[np.float64]:
    return np.array(
        qif_mean_field_rates(
            rate, potential, eta_bar=_ETA_BAR, delta=_DELTA, coupling=_COUPLING, current=_CURRENT
        ),
        dtype=np.float64,
    )


def test_rates_match_the_published_equations() -> None:
    r, v = 0.8, -0.3
    dr, dv = qif_mean_field_rates(
        r, v, eta_bar=_ETA_BAR, delta=_DELTA, coupling=_COUPLING, current=_CURRENT
    )
    assert dr == pytest.approx(_DELTA / np.pi + 2.0 * r * v)
    assert dv == pytest.approx(v * v + _ETA_BAR + _COUPLING * r + _CURRENT - np.pi**2 * r * r)


def test_rates_reject_negative_delta() -> None:
    with pytest.raises(ValueError, match="delta must be non-negative"):
        qif_mean_field_rates(1.0, 0.0, eta_bar=0.0, delta=-1.0, coupling=0.0)


def test_jacobian_matches_finite_difference() -> None:
    r, v = 0.8, -0.3
    eps = 1e-7
    numerical = np.array(
        [
            (_field(r + eps, v) - _field(r - eps, v)) / (2.0 * eps),
            (_field(r, v + eps) - _field(r, v - eps)) / (2.0 * eps),
        ]
    ).T
    assert qif_mean_field_jacobian(r, v, coupling=_COUPLING) == pytest.approx(numerical, abs=1e-6)


def test_fixed_point_annihilates_the_field() -> None:
    r_star, v_star = qif_mean_field_fixed_point(
        eta_bar=_ETA_BAR, delta=_DELTA, coupling=_COUPLING, current=_CURRENT
    )
    assert r_star > 0.0
    assert _field(r_star, v_star) == pytest.approx(np.zeros(2), abs=1e-12)
    assert v_star == pytest.approx(-_DELTA / (2.0 * np.pi * r_star))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": 0.0}, "delta must be positive"),
        ({"max_rate": 0.0}, "max_rate must be positive"),
    ],
)
def test_fixed_point_validation(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "eta_bar": _ETA_BAR,
        "delta": _DELTA,
        "coupling": _COUPLING,
        "current": _CURRENT,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        qif_mean_field_fixed_point(**call)


def test_fixed_point_unbracketed_raises() -> None:
    # Strong negative excitability with no coupling has no positive-rate fixed point in range.
    with pytest.raises(ValueError, match="no positive-rate fixed point"):
        qif_mean_field_fixed_point(eta_bar=-1.0, delta=1.0, coupling=0.0, max_rate=0.01)


def test_theta_qif_map_round_trips_and_matches_tangent() -> None:
    theta = np.array([0.3, -1.2, 2.1], dtype=np.float64)
    assert qif_potential_from_theta(theta) == pytest.approx(np.tan(theta / 2.0))
    assert theta_from_qif_potential(qif_potential_from_theta(theta)) == pytest.approx(theta)


def test_conformal_kuramoto_relation_round_trips() -> None:
    r, v = 0.9, -0.2
    order_parameter = kuramoto_order_parameter_from_macro(r, v)
    assert abs(order_parameter) <= 1.0 + 1e-12
    r_back, v_back = macro_from_kuramoto_order_parameter(order_parameter)
    assert r_back == pytest.approx(r)
    assert v_back == pytest.approx(v)


def test_integrator_settles_to_the_fixed_point() -> None:
    r_star, v_star = qif_mean_field_fixed_point(
        eta_bar=_ETA_BAR, delta=_DELTA, coupling=_COUPLING, current=_CURRENT
    )
    trajectory = integrate_qif_mean_field(
        0.5 * r_star,
        v_star,
        eta_bar=_ETA_BAR,
        delta=_DELTA,
        coupling=_COUPLING,
        current=_CURRENT,
        dt=0.005,
        n_steps=6000,
    )
    assert isinstance(trajectory, QifMeanFieldTrajectory)
    assert trajectory.firing_rate.shape == (6001,)
    assert trajectory.terminal_firing_rate == pytest.approx(r_star, abs=1e-3)
    assert trajectory.terminal_mean_potential == pytest.approx(v_star, abs=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": -1.0}, "delta must be non-negative"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"n_steps": 0}, "n_steps must be positive"),
    ],
)
def test_integrator_validation(kwargs: dict[str, Any], message: str) -> None:
    call: dict[str, Any] = {
        "eta_bar": _ETA_BAR,
        "delta": _DELTA,
        "coupling": _COUPLING,
        "current": _CURRENT,
        "dt": 0.01,
        "n_steps": 10,
    }
    call.update(kwargs)
    with pytest.raises(ValueError, match=message):
        integrate_qif_mean_field(0.5, -0.2, **call)


def test_differentiable_form_matches_finite_difference() -> None:
    dt, n_steps = 0.01, 40
    r0, v0 = 0.5, -0.2

    def run(rate: float, potential: float, eta: float, delta: float, j: float, i: float) -> float:
        return integrate_qif_mean_field(
            rate,
            potential,
            eta_bar=eta,
            delta=delta,
            coupling=j,
            current=i,
            dt=dt,
            n_steps=n_steps,
        ).terminal_firing_rate

    value, grads = qif_mean_field_terminal_value_and_grad(
        r0,
        v0,
        eta_bar=_ETA_BAR,
        delta=_DELTA,
        coupling=_COUPLING,
        current=_CURRENT,
        dt=dt,
        n_steps=n_steps,
        objective=lambda r, v: r,
        objective_grad=lambda r, v: (1.0, 0.0),
    )
    assert value == pytest.approx(run(r0, v0, _ETA_BAR, _DELTA, _COUPLING, _CURRENT))

    h = 1e-6
    base = (r0, v0, _ETA_BAR, _DELTA, _COUPLING, _CURRENT)

    def fd(index: int) -> float:
        high = list(base)
        low = list(base)
        high[index] += h
        low[index] -= h
        return (run(*high) - run(*low)) / (2.0 * h)

    assert grads.initial_firing_rate == pytest.approx(fd(0), abs=1e-7)
    assert grads.initial_mean_potential == pytest.approx(fd(1), abs=1e-7)
    assert grads.eta_bar == pytest.approx(fd(2), abs=1e-7)
    assert grads.delta == pytest.approx(fd(3), abs=1e-7)
    assert grads.coupling == pytest.approx(fd(4), abs=1e-7)
    assert grads.current == pytest.approx(fd(5), abs=1e-7)
    assert isinstance(grads, QifMeanFieldGradients)


def _theta_ensemble_order_parameter(
    rate0: float, potential0: float, dt: float, n_steps: int, n_neurons: int
) -> NDArray[np.complex128]:
    """A direct large-N theta-neuron ensemble, returning the per-step order parameter ``⟨e^{iθ}⟩``.

    The excitabilities are placed on Lorentzian quantiles and the membrane potentials are
    initialised Lorentzian with centre ``potential0`` and half-width ``π rate0`` — the macroscopic
    state on the Ott–Antonsen manifold. Each neuron is driven by ``η_j + J r(t)`` with the
    self-consistent population firing rate ``r = π^{-1}\\operatorname{Re} W`` recovered from the
    ensemble's own order parameter, so the ensemble is closed; the order parameter is a smooth
    population average (no spike binning), which makes the comparison robust at finite ``N``.
    """
    quantiles = (np.arange(1, n_neurons + 1) - 0.5) / n_neurons
    lorentzian = np.tan(np.pi * (quantiles - 0.5))
    excitability = _ETA_BAR + _DELTA * lorentzian
    theta = 2.0 * np.arctan(potential0 + np.pi * rate0 * lorentzian)
    rate = rate0
    series = np.empty(n_steps, dtype=np.complex128)

    def derivative(phase: NDArray[np.float64], drive: NDArray[np.float64]) -> NDArray[np.float64]:
        return (1.0 - np.cos(phase)) + (1.0 + np.cos(phase)) * drive

    for step in range(n_steps):
        drive = excitability + _COUPLING * rate
        k1 = derivative(theta, drive)
        k2 = derivative(theta + 0.5 * dt * k1, drive)
        k3 = derivative(theta + 0.5 * dt * k2, drive)
        k4 = derivative(theta + dt * k3, drive)
        theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        order_parameter = np.mean(np.exp(1j * theta))
        series[step] = order_parameter
        conformal = (1.0 - np.conj(order_parameter)) / (1.0 + np.conj(order_parameter))
        rate = float(conformal.real / np.pi)
    return series


def test_order_parameter_trajectory_matches_theta_ensemble() -> None:
    r_star, v_star = qif_mean_field_fixed_point(
        eta_bar=_ETA_BAR, delta=_DELTA, coupling=_COUPLING, current=_CURRENT
    )
    dt, n_steps, n_neurons = 0.005, 1200, 20000
    reduced = integrate_qif_mean_field(
        0.5 * r_star,
        v_star,
        eta_bar=_ETA_BAR,
        delta=_DELTA,
        coupling=_COUPLING,
        current=_CURRENT,
        dt=dt,
        n_steps=n_steps,
    )
    reduced_order_parameter = np.array(
        [
            kuramoto_order_parameter_from_macro(
                float(reduced.firing_rate[step + 1]), float(reduced.mean_potential[step + 1])
            )
            for step in range(n_steps)
        ]
    )
    ensemble = _theta_ensemble_order_parameter(0.5 * r_star, v_star, dt, n_steps, n_neurons)
    # The reduction is the theta-population order parameter conformally mapped; the residual is the
    # finite-N sampling error (~1/sqrt(N) ≈ 0.007 at N = 20000) over the whole trajectory.
    assert float(np.max(np.abs(ensemble - reduced_order_parameter))) < 0.03
