# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the Kuramoto neural-operator surrogate
"""Module-specific tests for :mod:`kuramoto_neural_operator`.

The contracts: the dataset builder produces correctly shaped branch/trunk/target tensors whose initial
sample is exactly the embedded initial condition; training a DeepONet on them drives the loss down
substantially (it learns the flow map); the trained operator forecasts a held-out initial condition
more accurately than a persistence baseline (it generalises); training is deterministic under a seed;
and the input contract is enforced. Requires the optional PyTorch extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

from oscillatools.accel.networked_kuramoto import networked_kuramoto_force  # noqa: E402
from oscillatools.neural_operator import (  # noqa: E402
    KuramotoOperatorDataset,
    TrainedKuramotoOperator,
    simulate_operator_dataset,
    train_kuramoto_neural_operator,
)

_N = 6
_DT = 0.05
_STEPS = 16


def _network(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.4, 1.0, size=(_N, _N))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    return {"omega": rng.standard_normal(_N) * 0.2, "coupling": coupling, "rng": rng}


def _true_rollout(initial: np.ndarray[Any, Any], network: dict[str, Any]) -> np.ndarray[Any, Any]:
    def field(angle: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(network["omega"] + networked_kuramoto_force(angle, network["coupling"]))

    state = initial.copy()
    trajectory = [state.copy()]
    for _ in range(_STEPS):
        k1 = field(state)
        k2 = field(state + 0.5 * _DT * k1)
        k3 = field(state + 0.5 * _DT * k2)
        k4 = field(state + _DT * k3)
        state = state + (_DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(state.copy())
    return np.array(trajectory)


def _angle_error(predicted: np.ndarray[Any, Any], truth: np.ndarray[Any, Any]) -> float:
    return float(np.mean(np.abs((predicted - truth + np.pi) % (2.0 * np.pi) - np.pi)))


def test_dataset_shapes_and_initial_sample() -> None:
    network = _network()
    dataset = simulate_operator_dataset(
        network["omega"], network["coupling"], n_trajectories=4, dt=_DT, n_steps=_STEPS, seed=1
    )
    assert isinstance(dataset, KuramotoOperatorDataset)
    samples = 4 * (_STEPS + 1)
    assert dataset.branch_inputs.shape == (samples, 2 * _N)
    assert dataset.trunk_inputs.shape == (samples, 1)
    assert dataset.targets.shape == (samples, 2 * _N)
    assert dataset.horizon == pytest.approx(_DT * _STEPS)
    # the first sample of each trajectory targets the initial condition itself (t = 0)
    assert dataset.trunk_inputs[0, 0] == pytest.approx(0.0)
    assert dataset.targets[0] == pytest.approx(dataset.branch_inputs[0])


def test_training_learns_and_forecast_beats_persistence() -> None:
    network = _network(2)
    dataset = simulate_operator_dataset(
        network["omega"], network["coupling"], n_trajectories=120, dt=_DT, n_steps=_STEPS, seed=3
    )
    operator = train_kuramoto_neural_operator(dataset, epochs=250, seed=0)
    assert isinstance(operator, TrainedKuramotoOperator)
    assert operator.loss_history.shape == (250,)
    # the loss falls substantially: the operator learns the flow map
    assert operator.loss_history[-1] < 0.4 * operator.loss_history[0]

    # on a held-out initial condition the operator beats the persistence baseline
    held_out = network["rng"].uniform(0.0, 2.0 * np.pi, size=_N)
    truth = _true_rollout(held_out, network)
    times = _DT * np.arange(_STEPS + 1, dtype=np.float64)
    forecast = operator.forecast(held_out, times)
    assert forecast.shape == (_STEPS + 1, _N)
    persistence = np.tile(truth[0], (_STEPS + 1, 1))
    assert _angle_error(forecast, truth) < _angle_error(persistence, truth)


def test_training_is_deterministic() -> None:
    network = _network(4)
    dataset = simulate_operator_dataset(
        network["omega"], network["coupling"], n_trajectories=30, dt=_DT, n_steps=_STEPS, seed=5
    )
    first = train_kuramoto_neural_operator(dataset, epochs=20, seed=7)
    second = train_kuramoto_neural_operator(dataset, epochs=20, seed=7)
    assert first.loss_history == pytest.approx(second.loss_history)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("dataset", {"omega": np.zeros(1)}, "omega must be a one-dimensional"),
        ("dataset", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("dataset", {"omega": np.full(_N, np.inf)}, "must be finite"),
        ("dataset", {"n_trajectories": 0}, "n_trajectories must be positive"),
        ("dataset", {"dt": 0.0}, "dt must be positive"),
        ("dataset", {"n_steps": 0}, "n_steps must be positive"),
        ("train", {"latent_dim": 0}, "latent_dim and hidden_dim must be positive"),
        ("train", {"epochs": 0}, "epochs must be positive"),
        ("train", {"learning_rate": 0.0}, "learning_rate must be positive"),
        ("forecast", {"initial_phases": np.zeros(_N + 1)}, "initial_phases must have length"),
        ("forecast", {"times": np.zeros((2, 2))}, "times must be a non-empty one-dimensional"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    network = _network()
    with pytest.raises(ValueError, match=message):
        if call == "dataset":
            args: dict[str, Any] = {
                "omega": network["omega"],
                "coupling": network["coupling"],
                "n_trajectories": 2,
                "dt": _DT,
                "n_steps": _STEPS,
                "seed": 0,
            }
            args.update(kwargs)
            simulate_operator_dataset(
                args["omega"],
                args["coupling"],
                n_trajectories=args["n_trajectories"],
                dt=args["dt"],
                n_steps=args["n_steps"],
                seed=args["seed"],
            )
        elif call == "train":
            dataset = simulate_operator_dataset(
                network["omega"],
                network["coupling"],
                n_trajectories=2,
                dt=_DT,
                n_steps=_STEPS,
                seed=0,
            )
            train_args: dict[str, Any] = {"latent_dim": 8, "epochs": 5, "learning_rate": 1e-3}
            train_args.update(kwargs)
            train_kuramoto_neural_operator(
                dataset,
                latent_dim=train_args["latent_dim"],
                epochs=train_args["epochs"],
                learning_rate=train_args["learning_rate"],
            )
        else:
            dataset = simulate_operator_dataset(
                network["omega"],
                network["coupling"],
                n_trajectories=2,
                dt=_DT,
                n_steps=_STEPS,
                seed=0,
            )
            operator = train_kuramoto_neural_operator(dataset, epochs=3, seed=0)
            forecast_args: dict[str, Any] = {
                "initial_phases": np.zeros(_N),
                "times": _DT * np.arange(_STEPS + 1),
            }
            forecast_args.update(kwargs)
            operator.forecast(forecast_args["initial_phases"], forecast_args["times"])
