# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for differentiable-simulation policy learning
"""Module-specific tests for :mod:`kuramoto_policy_learning`.

The contracts: the analytic policy gradient (forward-mode sensitivity through the closed-loop rollout)
matches finite differences to machine precision; gradient descent over a batch of initial conditions
learns a repulsive feedback policy that desynchronises a held-out initial condition (generalisation,
the property a feedback policy has over open-loop control); the learned policy evaluates its control;
and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.kuramoto_policy_learning import (
    DesynchronisingPolicy,
    PolicyRolloutGradients,
    learn_desynchronising_policy,
    policy_rollout_value_and_grad,
)
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force
from oscillatools.accel.order_parameter_observables import order_parameter

_N = 8
_DT = 0.05
_STEPS = 60


def _network(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(_N, _N))
    coupling = 0.6 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {"omega": rng.standard_normal(_N) * 0.2, "coupling": coupling, "rng": rng}


def _final_order_parameter(
    initial: NDArray[np.float64], policy: DesynchronisingPolicy, network: dict[str, Any]
) -> float:
    phases = initial.copy()
    for _ in range(_STEPS):
        phases = phases + _DT * (
            network["omega"]
            + networked_kuramoto_force(phases, network["coupling"])
            + policy.control(phases)
        )
    return order_parameter(phases)


def test_policy_gradient_matches_finite_differences() -> None:
    network = _network()
    initial = network["rng"].uniform(0.0, 2.0 * np.pi, size=_N)
    sine = np.array([0.3, -0.1])
    cosine = np.array([0.0, 0.2])
    gradients = policy_rollout_value_and_grad(
        initial,
        sine,
        cosine,
        network["omega"],
        network["coupling"],
        _DT,
        _STEPS,
        parameter_penalty=0.01,
    )
    assert isinstance(gradients, PolicyRolloutGradients)
    analytic = np.concatenate([gradients.sine_gradient, gradients.cosine_gradient])

    eps = 1e-6
    base = np.concatenate([sine, cosine])
    finite = np.zeros(4, dtype=np.float64)

    def cost(packed: NDArray[np.float64]) -> float:
        return policy_rollout_value_and_grad(
            initial,
            packed[:2],
            packed[2:],
            network["omega"],
            network["coupling"],
            _DT,
            _STEPS,
            parameter_penalty=0.01,
        ).cost

    for index in range(4):
        plus = base.copy()
        minus = base.copy()
        plus[index] += eps
        minus[index] -= eps
        finite[index] = (cost(plus) - cost(minus)) / (2.0 * eps)
    assert analytic == pytest.approx(finite, abs=1e-6)


def test_learned_policy_desynchronises_a_held_out_initial_condition() -> None:
    network = _network()
    batch = np.array([network["rng"].uniform(0.0, 2.0 * np.pi, size=_N) for _ in range(5)])
    policy, history = learn_desynchronising_policy(
        batch,
        network["omega"],
        network["coupling"],
        _DT,
        _STEPS,
        1,
        parameter_penalty=0.01,
        learning_rate=0.5,
        n_iterations=150,
    )
    assert isinstance(policy, DesynchronisingPolicy)
    # the learned first-harmonic gain is repulsive (negative)
    assert policy.sine_gains[0] < 0.0
    # the batch-averaged rollout cost falls
    assert history[-1] < history[0]
    # the policy desynchronises a held-out initial condition it never trained on
    held_out = network["rng"].uniform(0.0, 2.0 * np.pi, size=_N)
    uncontrolled = _final_order_parameter(
        held_out, DesynchronisingPolicy(np.zeros(1), np.zeros(1)), network
    )
    controlled = _final_order_parameter(held_out, policy, network)
    assert uncontrolled > 0.8
    assert controlled < 0.3


def test_policy_control_evaluation() -> None:
    policy = DesynchronisingPolicy(np.array([-1.0]), np.array([0.0]))
    phases = np.linspace(0.0, 1.0, _N, dtype=np.float64)
    control = policy.control(phases)
    # u_i = (a/N) sum_l sin(theta_l - theta_i) = a * mean-field force / coupling==1
    difference = phases[None, :] - phases[:, None]
    expected = -1.0 * np.mean(np.sin(difference), axis=1)
    assert control == pytest.approx(expected)
    # a zero policy applies no control
    zero = DesynchronisingPolicy(np.zeros(2), np.zeros(2))
    assert zero.control(phases) == pytest.approx(np.zeros(_N))


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("rollout", {"initial_phases": np.zeros(1)}, "initial_phases must be a one-dimensional"),
        ("rollout", {"sine_gains": np.zeros(2), "cosine_gains": np.zeros(3)}, "equal-length"),
        ("rollout", {"omega": np.zeros(_N + 1)}, "omega must have shape"),
        ("rollout", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("rollout", {"initial_phases": np.full(_N, np.nan)}, "initial_phases must be finite"),
        ("rollout", {"omega": np.full(_N, np.inf)}, "omega and coupling must be finite"),
        ("rollout", {"dt": 0.0}, "dt must be positive"),
        ("rollout", {"n_steps": 0}, "n_steps must be positive"),
        ("rollout", {"parameter_penalty": -1.0}, "parameter_penalty must be non-negative"),
        ("learn", {"initial_phases_batch": np.zeros((2, 1))}, "initial_phases_batch must be"),
        ("learn", {"n_harmonics": 0}, "n_harmonics must be positive"),
        ("learn", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("learn", {"n_iterations": 0}, "n_iterations must be positive"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    network = _network()
    initial = network["rng"].uniform(0.0, 2.0 * np.pi, size=_N)
    with pytest.raises(ValueError, match=message):
        if call == "rollout":
            args: dict[str, Any] = {
                "initial_phases": initial,
                "sine_gains": np.array([0.1]),
                "cosine_gains": np.array([0.0]),
                "omega": network["omega"],
                "coupling": network["coupling"],
                "dt": _DT,
                "n_steps": 10,
                "parameter_penalty": 0.01,
            }
            args.update(kwargs)
            policy_rollout_value_and_grad(
                args["initial_phases"],
                args["sine_gains"],
                args["cosine_gains"],
                args["omega"],
                args["coupling"],
                args["dt"],
                args["n_steps"],
                parameter_penalty=args["parameter_penalty"],
            )
        else:
            args = {
                "initial_phases_batch": initial[None, :],
                "n_harmonics": 1,
                "learning_rate": 0.5,
                "n_iterations": 5,
            }
            args.update(kwargs)
            learn_desynchronising_policy(
                args["initial_phases_batch"],
                network["omega"],
                network["coupling"],
                _DT,
                10,
                args["n_harmonics"],
                parameter_penalty=0.01,
                learning_rate=args["learning_rate"],
                n_iterations=args["n_iterations"],
            )
