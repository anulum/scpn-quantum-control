# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — inertial Kuramoto first-order transition reproduction
"""Reproduce the first-order (hysteretic) synchronisation transition of the inertial model.

The second-order ("inertial") Kuramoto model ``m θ̈ + γ θ̇ = ω + (K/N) Σ_j sin(θ_j − θ_i)``
undergoes a *discontinuous, hysteretic* synchronisation transition once the inertia ``m``
is large enough — in contrast to the continuous transition of the first-order model. Sweeping
the coupling ``K`` upward from an incoherent state, the ensemble stays desynchronised until a
forward critical coupling and then jumps to synchrony; sweeping ``K`` back down from the
synchronised state, it stays synchronised until a lower backward critical coupling. Between the
two lies a bistable window, so the forward and backward branches trace a hysteresis loop
(Tanaka, Lichtenberg & Oishi, *Phys. Rev. Lett.* **78**, 2104 (1997); the finite-``N`` picture
is Olmi, Navas, Boccaletti & Torcini, *Phys. Rev. E* **90**, 042905 (2014)).

This drives the production inertial integrator :func:`integrate_inertial` with the production
mean-field :func:`mean_field_force` and reads the production :func:`order_parameter`. The two
branches are obtained by *adiabatic continuation* — the terminal ``(θ, θ̇)`` at one coupling
seeds the next — which is what exposes the bistability. Everything is deterministic: the natural
frequencies are the Lorentzian inverse-CDF at symmetric quantiles (no random draws), the initial
conditions are fixed, and the RK4 flow is fixed-step, so the loop reproduces bit-for-bit. A weak
inertia (``m = 1``) contrast confirms the loop is inertia-induced, not an integration artefact.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import integrate_inertial, mean_field_force, order_parameter

_N = 64
_HALF_WIDTH = 1.0  # Lorentzian half-width γ; the overdamped onset would sit at K_c = 2γ = 2.
_DAMPING = 1.0
_DT = 0.05
_N_STEPS = 3000
_TAIL = 600  # samples tail-averaged for the settled order parameter

_STRONG_INERTIA = 3.0  # above the first-order threshold — a wide hysteresis loop
_WEAK_INERTIA = 1.0  # near-continuous — the inertia-induced control

_COUPLINGS: tuple[float, ...] = (1.5, 2.5, 3.0, 3.5, 4.5, 5.5)
_LOW_COUPLING = 1.5
_BISTABLE_COUPLING = 3.0
_HIGH_COUPLING = 5.5
_BISTABLE_WINDOW: tuple[float, ...] = (2.5, 3.0, 3.5)

_Branch = dict[float, float]


def _lorentzian_quantile_frequencies(count: int, half_width: float) -> NDArray[np.float64]:
    """Deterministic Lorentzian frequencies: the inverse CDF at symmetric mid-quantiles."""
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    return half_width * np.tan(np.pi * (quantiles - 0.5))


def _settled_order_parameter(
    phases: NDArray[np.float64],
    velocities: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: float,
    mass: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Integrate one coupling and return the tail-averaged ``r`` plus the terminal state."""

    def force(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return mean_field_force(theta, coupling)

    trajectory = integrate_inertial(
        phases, velocities, omega, force, mass, damping=_DAMPING, dt=_DT, n_steps=_N_STEPS
    )
    tail = trajectory.phases[_N_STEPS + 1 - _TAIL :]
    settled = float(np.mean([order_parameter(row) for row in tail]))
    return settled, trajectory.terminal_phases, trajectory.terminal_velocities


def _adiabatic_sweep(
    omega: NDArray[np.float64], mass: float, *, from_synchronised: bool
) -> _Branch:
    """Sweep the coupling with adiabatic continuation, returning ``{K: r}``.

    ``from_synchronised`` seeds a phase-locked state and sweeps ``K`` downward (the upper
    branch); otherwise it seeds a spread state and sweeps upward (the lower branch).
    """
    if from_synchronised:
        phases = np.zeros(_N, dtype=np.float64)
        order = tuple(reversed(_COUPLINGS))
    else:
        phases = np.linspace(-np.pi, np.pi, _N, endpoint=False)
        order = _COUPLINGS
    velocities = np.zeros(_N, dtype=np.float64)
    branch: _Branch = {}
    for coupling in order:
        branch[coupling], phases, velocities = _settled_order_parameter(
            phases, velocities, omega, coupling, mass
        )
    return branch


def _loop_area(forward: _Branch, backward: _Branch) -> float:
    """Summed forward↔backward gap over the coupling grid — the hysteresis loop 'area'."""
    return float(sum(backward[k] - forward[k] for k in _COUPLINGS))


@pytest.fixture(scope="module")
def branches() -> dict[str, _Branch]:
    """Compute the four branches once (strong/weak inertia × forward/backward)."""
    omega = _lorentzian_quantile_frequencies(_N, _HALF_WIDTH)
    return {
        "strong_forward": _adiabatic_sweep(omega, _STRONG_INERTIA, from_synchronised=False),
        "strong_backward": _adiabatic_sweep(omega, _STRONG_INERTIA, from_synchronised=True),
        "weak_forward": _adiabatic_sweep(omega, _WEAK_INERTIA, from_synchronised=False),
        "weak_backward": _adiabatic_sweep(omega, _WEAK_INERTIA, from_synchronised=True),
    }


def test_loop_is_closed_at_low_coupling(branches: dict[str, _Branch]) -> None:
    """Below the backward critical coupling both branches are incoherent — the loop closes."""
    forward = branches["strong_forward"][_LOW_COUPLING]
    backward = branches["strong_backward"][_LOW_COUPLING]
    assert forward < 0.20, f"forward branch already coherent at K={_LOW_COUPLING}: r={forward:.3f}"
    assert backward < 0.20, (
        f"backward branch still coherent at K={_LOW_COUPLING}: r={backward:.3f}"
    )
    assert abs(backward - forward) < 0.05, "the branches must coincide where the loop closes"


def test_bistable_window_shows_hysteresis(branches: dict[str, _Branch]) -> None:
    """Inside the window the incoherent branch stays low while the synchronised branch holds."""
    forward = branches["strong_forward"][_BISTABLE_COUPLING]
    backward = branches["strong_backward"][_BISTABLE_COUPLING]
    assert forward < 0.25, f"forward branch synchronised too early: r={forward:.3f}"
    assert backward > 0.40, f"backward branch failed to hold synchrony: r={backward:.3f}"
    assert backward - forward > 0.30, f"no hysteresis gap at K={_BISTABLE_COUPLING}"


def test_backward_branch_dominates_across_the_window(branches: dict[str, _Branch]) -> None:
    """The upper (backward) branch stays at or above the lower branch, strictly in the window."""
    forward = branches["strong_forward"]
    backward = branches["strong_backward"]
    for coupling in _COUPLINGS:
        assert backward[coupling] >= forward[coupling] - 0.05, (
            f"branch ordering violated at K={coupling}"
        )
    for coupling in _BISTABLE_WINDOW:
        assert backward[coupling] - forward[coupling] > 0.12, (
            f"hysteresis gap collapsed at K={coupling}"
        )


def test_synchronised_branch_persists_at_high_coupling(branches: dict[str, _Branch]) -> None:
    """Well above the forward critical coupling the upper branch is strongly synchronised."""
    backward = branches["strong_backward"][_HIGH_COUPLING]
    assert backward > 0.55, (
        f"upper branch not synchronised at K={_HIGH_COUPLING}: r={backward:.3f}"
    )


def test_hysteresis_is_inertia_induced(branches: dict[str, _Branch]) -> None:
    """A weak inertia traces a far smaller loop — the hysteresis comes from the inertia."""
    strong_area = _loop_area(branches["strong_forward"], branches["strong_backward"])
    weak_area = _loop_area(branches["weak_forward"], branches["weak_backward"])
    assert strong_area > 0.0 and weak_area >= 0.0
    assert strong_area > 3.0 * weak_area, (
        f"loop not dominated by inertia: strong={strong_area:.3f}, weak={weak_area:.3f}"
    )


def test_sweep_is_deterministic() -> None:
    """The forward sweep is a fixed-step deterministic flow — recomputation is bit-identical."""
    omega = _lorentzian_quantile_frequencies(_N, _HALF_WIDTH)
    first = _adiabatic_sweep(omega, _STRONG_INERTIA, from_synchronised=False)
    second = _adiabatic_sweep(omega, _STRONG_INERTIA, from_synchronised=False)
    assert [first[k] for k in _COUPLINGS] == [second[k] for k in _COUPLINGS]


def test_quantile_frequencies_are_symmetric() -> None:
    """The deterministic Lorentzian construction is symmetric (zero-sum mean field driver)."""
    omega = _lorentzian_quantile_frequencies(_N, _HALF_WIDTH)
    assert omega.shape == (_N,)
    np.testing.assert_allclose(omega, -omega[::-1], atol=1e-12)
    assert abs(float(np.sum(omega))) < 1e-9
