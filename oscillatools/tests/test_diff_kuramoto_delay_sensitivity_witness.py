# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Independent-autodiff witness for the Kuramoto delay sensitivity dtheta_N/dtau
r"""Independent forward-mode autodiff witness for the time-delayed Kuramoto delay sensitivity.

:func:`~oscillatools.accel.diff_kuramoto_delay_sensitivity.delayed_delay_sensitivity`
co-integrates a hand-derived tangent to obtain ``∂θ_N/∂τ`` and validates it against a central finite
difference over ``τ`` — an independent *numerical* witness, but only to roughly ``√eps`` precision, so
the production suite asserts it at ``1e-6``. These tests add the missing *second* witness: they
differentiate the identical continuous-delay forward map with JAX forward-mode autodiff
(:func:`jax.jvp` along the scalar ``τ``) and require the hand-co-integrated ``σ_N`` to agree with the
JAX tangent at (double-precision) machine tolerance — three to five orders tighter than the
finite-difference check. The JAX forward map in :mod:`tests.kuramoto_delay_sensitivity_witness`
re-implements only the integrator (reading the lagged phase by the same linear interpolation), so the
sole quantity under test is the tangent recurrence.

Covered: the forward map reproduces the production terminal state (only the differentiation differs);
the delay sensitivity matches the JAX tangent across seeds and interior delays; the JAX tangent
matches a central finite difference of the same map; the objective delay gradient ``dL/dτ`` matches
the witnessed cotangent contraction; and the analytic zero-coupling case has a witnessed-zero tangent.

JAX is an optional dependency; importing the witness module skips this file when JAX is absent, and
the ``differentiable-frameworks`` overlay lane provisions ``jax[cpu]``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from oscillatools.accel.diff_kuramoto_delay_sensitivity import (
    delayed_delay_gradient,
    delayed_delay_sensitivity,
)
from tests import kuramoto_delay_sensitivity_witness as witness

_DT = 0.02
_N_STEPS = 100

# Machine-tolerance parity: the JAX tangent and the hand-co-integrated sensitivity differentiate the
# byte-identical float64 forward map, so they agree far tighter than the finite-difference check the
# production suite uses (which sits at 1e-6).
_ATOL = 1e-9
_RTOL = 1e-7

# Interior delays that are not (half-)integer multiples of dt=0.02, where the interpolant is smooth in
# tau and the production sensitivity is defined.
_INTERIOR_DELAYS = (0.366, 0.5123, 0.913)


def _problem(
    count: int = 4, seed: int = 20260703
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return a reproducible constant-history ``(history, omega, coupling)`` delayed-Kuramoto problem."""
    rng = np.random.default_rng(seed)
    history = rng.uniform(-math.pi, math.pi, count)
    omega = rng.uniform(-0.5, 0.5, count)
    coupling = rng.uniform(0.3, 0.8, (count, count))
    np.fill_diagonal(coupling, 0.0)
    return history, omega, coupling


@pytest.mark.parametrize("delay", _INTERIOR_DELAYS)
@pytest.mark.parametrize("seed", [20260703, 7, 101])
def test_forward_map_reproduces_the_production_delay_sensitivity_forward(
    delay: float, seed: int
) -> None:
    """The JAX continuous-delay forward reproduces the production terminal phases (only diff differs)."""
    history, omega, coupling = _problem(seed=seed)
    production_phases, _ = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_N_STEPS
    )
    witness_phases, _ = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, delay, _DT, _N_STEPS
    )
    assert np.allclose(np.asarray(witness_phases), production_phases, atol=1e-11, rtol=0.0)


@pytest.mark.parametrize("delay", _INTERIOR_DELAYS)
@pytest.mark.parametrize("seed", [20260703, 7, 101])
def test_delay_sensitivity_matches_jax_forward_mode_at_machine_tolerance(
    delay: float, seed: int
) -> None:
    """The hand-co-integrated ``∂θ_N/∂τ`` equals the JAX forward-mode tangent of the same map."""
    history, omega, coupling = _problem(seed=seed)
    _, production_sensitivity = delayed_delay_sensitivity(
        history, omega, coupling, delay=delay, dt=_DT, n_steps=_N_STEPS
    )
    _, witness_tangent = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, delay, _DT, _N_STEPS
    )
    tangent = np.asarray(witness_tangent)
    # A meaningful, non-degenerate sensitivity — the witness exercises real signal, not near-zero.
    assert np.max(np.abs(production_sensitivity)) > 1e-2
    assert np.allclose(production_sensitivity, tangent, atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("delay", _INTERIOR_DELAYS)
def test_jax_forward_mode_matches_central_finite_difference(delay: float) -> None:
    """The JAX tangent equals a central finite difference of the same forward map (it is its derivative)."""
    history, omega, coupling = _problem()
    prefix_length = witness._history_prefix_length(delay, _DT)
    history_j = jnp.asarray(history)
    omega_j = jnp.asarray(omega)
    coupling_j = jnp.asarray(coupling)

    def terminal(tau: float) -> NDArray[np.float64]:
        return np.asarray(
            witness.delayed_terminal_continuous_delay(
                history_j, omega_j, coupling_j, tau, _DT, _N_STEPS, prefix_length
            )
        )

    _, witness_tangent = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, delay, _DT, _N_STEPS
    )
    epsilon = 1e-6
    finite_difference = (terminal(delay + epsilon) - terminal(delay - epsilon)) / (2.0 * epsilon)
    assert np.allclose(np.asarray(witness_tangent), finite_difference, atol=1e-6, rtol=1e-5)


def test_objective_delay_gradient_matches_the_witnessed_cotangent_contraction() -> None:
    """The objective ``dL/dτ`` equals the terminal cotangent contracted with the JAX tangent."""
    history, omega, coupling = _problem()
    delay = 0.417
    weights = np.linspace(0.5, 2.0, history.size)

    def objective(theta: NDArray[np.float64]) -> float:
        return float(weights @ theta)

    def objective_grad(_theta: NDArray[np.float64]) -> NDArray[np.float64]:
        return weights

    _, production_gradient = delayed_delay_gradient(
        history,
        omega,
        coupling,
        delay=delay,
        dt=_DT,
        n_steps=_N_STEPS,
        objective=objective,
        objective_grad=objective_grad,
    )
    _, witness_tangent = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, delay, _DT, _N_STEPS
    )
    witnessed_gradient = float(weights @ np.asarray(witness_tangent))
    assert production_gradient == pytest.approx(witnessed_gradient, abs=_ATOL, rel=_RTOL)


def test_zero_coupling_sensitivity_is_witnessed_zero() -> None:
    """With no coupling the terminal state is independent of ``τ``: both tangents are exactly zero."""
    history, omega, _ = _problem()
    coupling = np.zeros((history.size, history.size), dtype=np.float64)
    _, production_sensitivity = delayed_delay_sensitivity(
        history, omega, coupling, delay=0.313, dt=_DT, n_steps=_N_STEPS
    )
    _, witness_tangent = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, 0.313, _DT, _N_STEPS
    )
    assert np.allclose(production_sensitivity, 0.0, atol=1e-14)
    assert np.allclose(np.asarray(witness_tangent), 0.0, atol=1e-14)


def test_witness_tangent_is_resident_on_a_jax_device() -> None:
    """The witness tangent is a JAX array on an available device (GPU when present, else CPU)."""
    history, omega, coupling = _problem()
    _, witness_tangent = witness.delay_sensitivity_via_jvp(
        history, omega, coupling, 0.366, _DT, _N_STEPS
    )
    assert isinstance(witness_tangent, jax.Array)
    assert witness_tangent.shape == (history.size,)
    assert len(witness_tangent.devices()) >= 1
