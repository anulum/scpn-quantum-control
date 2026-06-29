# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for data-driven coupling-function inference
"""Module-specific tests for :mod:`kuramoto_coupling_function_inference`.

The contracts: the Fourier coupling function reproduces the canonical sinusoidal and Sakaguchi
interactions; the physics-informed collocation estimator recovers a diverse (two-harmonic) coupling
function and the natural frequencies exactly from clean snapshot/derivative data; the trajectory-match
loss gradient matches finite differences to machine precision; gradient descent on a sampled
trajectory recovers the true coupling function; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.kuramoto_coupling_function_inference import (
    CouplingFunctionEstimate,
    coupling_function_trajectory_value_and_grad,
    coupling_function_value,
    infer_coupling_function,
    refine_coupling_function,
)

_DT = 0.05


def _coupling(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    coupling = 0.4 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return coupling


def _field(
    phases: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    difference = phases[None, :] - phases[:, None]
    return np.asarray(
        omega + np.sum(coupling * coupling_function_value(difference, sine, cosine), axis=1)
    )


def _rk4(
    phases: NDArray[np.float64],
    sine: NDArray[np.float64],
    cosine: NDArray[np.float64],
    omega: NDArray[np.float64],
    coupling: NDArray[np.float64],
) -> NDArray[np.float64]:
    k1 = _field(phases, sine, cosine, omega, coupling)
    k2 = _field(phases + 0.5 * _DT * k1, sine, cosine, omega, coupling)
    k3 = _field(phases + 0.5 * _DT * k2, sine, cosine, omega, coupling)
    k4 = _field(phases + _DT * k3, sine, cosine, omega, coupling)
    return np.asarray(phases + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def test_coupling_function_reproduces_canonical_interactions() -> None:
    grid = np.linspace(-np.pi, np.pi, 17, dtype=np.float64)
    classic = coupling_function_value(grid, np.array([1.0]), np.array([0.0]))
    assert classic == pytest.approx(np.sin(grid))
    beta = 0.4
    sakaguchi = coupling_function_value(grid, np.array([np.cos(beta)]), np.array([-np.sin(beta)]))
    assert sakaguchi == pytest.approx(np.sin(grid - beta))


def test_collocation_recovers_diverse_coupling_and_frequencies_exactly() -> None:
    rng = np.random.default_rng(0)
    n, harmonics = 6, 2
    coupling = _coupling(rng, n)
    omega = rng.standard_normal(n) * 0.2
    sine_true = np.array([1.0, 0.3])
    cosine_true = np.array([-0.4, 0.15])
    snapshots = rng.uniform(0.0, 2.0 * np.pi, size=(20, n))
    derivatives = np.array(
        [_field(snapshots[k], sine_true, cosine_true, omega, coupling) for k in range(20)]
    )

    estimate = infer_coupling_function(snapshots, derivatives, coupling, harmonics)
    assert estimate.sine_coefficients == pytest.approx(sine_true, abs=1e-9)
    assert estimate.cosine_coefficients == pytest.approx(cosine_true, abs=1e-9)
    assert estimate.frequencies == pytest.approx(omega, abs=1e-9)
    assert estimate.residual < 1e-9


def test_trajectory_match_gradient_matches_finite_differences() -> None:
    rng = np.random.default_rng(0)
    n, harmonics = 6, 2
    coupling = _coupling(rng, n)
    omega = rng.standard_normal(n) * 0.2
    sine_true = np.array([1.0, 0.3])
    cosine_true = np.array([-0.4, 0.15])
    phases0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    observed = [phases0]
    for _ in range(12):
        observed.append(_rk4(observed[-1], sine_true, cosine_true, omega, coupling))
    trajectory = np.array(observed)

    sine = np.array([0.7, 0.0])
    cosine = np.array([0.0, 0.0])
    grads = coupling_function_trajectory_value_and_grad(
        phases0, trajectory, sine, cosine, omega, coupling, _DT
    )
    analytic = np.concatenate([grads.sine_gradient, grads.cosine_gradient])

    eps = 1e-6
    base = np.concatenate([sine, cosine])
    finite = np.zeros(2 * harmonics, dtype=np.float64)

    def loss(packed: NDArray[np.float64]) -> float:
        return coupling_function_trajectory_value_and_grad(
            phases0, trajectory, packed[:harmonics], packed[harmonics:], omega, coupling, _DT
        ).loss

    for index in range(2 * harmonics):
        plus = base.copy()
        minus = base.copy()
        plus[index] += eps
        minus[index] -= eps
        finite[index] = (loss(plus) - loss(minus)) / (2.0 * eps)
    assert analytic == pytest.approx(finite, abs=1e-6)


def test_trajectory_match_recovers_the_coupling_function() -> None:
    rng = np.random.default_rng(7)
    n, harmonics = 5, 2
    coupling = _coupling(rng, n)
    omega = rng.standard_normal(n) * 0.2
    sine_true = np.array([1.0, 0.3])
    cosine_true = np.array([-0.4, 0.15])
    phases0 = rng.uniform(0.0, 2.0 * np.pi, size=n)
    observed = [phases0]
    for _ in range(10):
        observed.append(_rk4(observed[-1], sine_true, cosine_true, omega, coupling))
    trajectory = np.array(observed)

    estimate, history = refine_coupling_function(
        phases0,
        trajectory,
        omega,
        coupling,
        _DT,
        harmonics,
        learning_rate=0.03,
        n_iterations=2000,
    )
    assert isinstance(estimate, CouplingFunctionEstimate)
    assert history[-1] < 1e-6 * history[0]
    assert estimate.sine_coefficients == pytest.approx(sine_true, abs=1e-3)
    assert estimate.cosine_coefficients == pytest.approx(cosine_true, abs=1e-3)
    # the estimate evaluates the recovered coupling function
    grid = np.array([0.0, 0.5, 1.0])
    assert estimate.evaluate(grid) == pytest.approx(
        coupling_function_value(grid, sine_true, cosine_true), abs=1e-3
    )


def test_coupling_function_value_rejects_mismatched_coefficients() -> None:
    with pytest.raises(ValueError, match="equal-length non-empty"):
        coupling_function_value(np.array([0.0]), np.array([1.0]), np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("infer", {"phases": np.zeros(4)}, "phases must be a"),
        ("infer", {"derivatives": np.zeros((3, 4))}, "same shape as phases"),
        ("infer", {"coupling": np.zeros((5, 6))}, "coupling must have shape"),
        ("infer", {"n_harmonics": 0}, "n_harmonics must be positive"),
        ("traj", {"initial_phases": np.zeros((2, 2))}, "initial_phases must be a"),
        ("traj", {"observations": np.zeros((1, 5))}, "observations must have shape"),
        ("traj", {"sine": np.array([0.5])}, "equal-length non-empty"),
        ("traj", {"omega": np.zeros(4)}, "omega must have shape"),
        ("traj", {"coupling": np.zeros((5, 6))}, "coupling must have shape"),
        ("traj", {"dt": 0.0}, "dt must be positive"),
        ("refine", {"n_harmonics": 0}, "n_harmonics must be positive"),
        ("refine", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("refine", {"n_iterations": 0}, "n_iterations must be positive"),
        ("refine", {"initial_sine": np.zeros(3)}, "initial coefficients must have shape"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    rng = np.random.default_rng(1)
    n = 5
    coupling = _coupling(rng, n)
    omega = rng.standard_normal(n) * 0.2
    snapshots = rng.uniform(0.0, 2.0 * np.pi, size=(6, n))
    derivatives = rng.standard_normal((6, n))
    trajectory = rng.uniform(0.0, 2.0 * np.pi, size=(5, n))
    with pytest.raises(ValueError, match=message):
        if call == "infer":
            args: dict[str, Any] = {
                "phases": snapshots,
                "derivatives": derivatives,
                "coupling": coupling,
                "n_harmonics": 2,
            }
            args.update(kwargs)
            infer_coupling_function(
                args["phases"], args["derivatives"], args["coupling"], args["n_harmonics"]
            )
        elif call == "traj":
            args = {
                "initial_phases": trajectory[0],
                "observations": trajectory,
                "sine": np.array([0.5, 0.0]),
                "cosine": np.array([0.0, 0.0]),
                "omega": omega,
                "coupling": coupling,
                "dt": _DT,
            }
            args.update(kwargs)
            coupling_function_trajectory_value_and_grad(
                args["initial_phases"],
                args["observations"],
                args["sine"],
                args["cosine"],
                args["omega"],
                args["coupling"],
                args["dt"],
            )
        else:
            args = {
                "n_harmonics": 2,
                "learning_rate": 0.03,
                "n_iterations": 10,
                "initial_sine": None,
            }
            args.update(kwargs)
            refine_coupling_function(
                trajectory[0],
                trajectory,
                omega,
                coupling,
                _DT,
                args["n_harmonics"],
                learning_rate=args["learning_rate"],
                n_iterations=args["n_iterations"],
                initial_sine=args["initial_sine"],
            )
