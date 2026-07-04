# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for multiplex (multilayer) Kuramoto networks
"""Module-specific tests for :mod:`multiplex_kuramoto`.

The contracts: with the inter-layer coupling off the field decouples into independent single-layer
Kuramoto systems exactly, and a single layer reproduces the networked force; the block Jacobian
matches finite differences; strong inter-layer coupling locks every node's replicas across layers
(inter-layer synchronisation → 1); the order-parameter diagnostics take their exact values on
constructed states; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel.multiplex_kuramoto import (
    MultiplexTrajectory,
    integrate_multiplex,
    interlayer_synchronisation,
    layer_order_parameters,
    multiplex_field,
    multiplex_jacobian,
)
from oscillatools.accel.networked_kuramoto import networked_kuramoto_force

_L = 3
_N = 6


def _problem(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    intra = np.empty((_L, _N, _N), dtype=np.float64)
    for layer in range(_L):
        raw = rng.uniform(0.0, 1.0, size=(_N, _N))
        intra[layer] = 0.5 * (raw + raw.T)
        np.fill_diagonal(intra[layer], 0.0)
    raw_inter = rng.uniform(0.0, 1.0, size=(_L, _L))
    inter = 0.3 * (raw_inter + raw_inter.T)
    np.fill_diagonal(inter, 0.0)
    return {
        "phases": rng.uniform(0.0, 2.0 * np.pi, size=(_L, _N)),
        "omega": rng.standard_normal((_L, _N)) * 0.2,
        "intra": intra,
        "inter": inter,
    }


def test_decoupled_layers_are_independent_kuramoto() -> None:
    problem = _problem()
    field = multiplex_field(
        problem["phases"], problem["omega"], problem["intra"], np.zeros((_L, _L))
    )
    independent = np.stack(
        [
            problem["omega"][layer]
            + networked_kuramoto_force(problem["phases"][layer], problem["intra"][layer])
            for layer in range(_L)
        ]
    )
    assert field == pytest.approx(independent, abs=1e-12)


def test_single_layer_reduces_to_networked_kuramoto() -> None:
    problem = _problem(1)
    field = multiplex_field(
        problem["phases"][:1], problem["omega"][:1], problem["intra"][:1], np.zeros((1, 1))
    )
    expected = problem["omega"][0] + networked_kuramoto_force(
        problem["phases"][0], problem["intra"][0]
    )
    assert field[0] == pytest.approx(expected, abs=1e-12)


def test_jacobian_matches_finite_differences() -> None:
    problem = _problem(2)
    jacobian = multiplex_jacobian(
        problem["phases"], problem["omega"], problem["intra"], problem["inter"]
    )
    flat = problem["phases"].ravel()
    eps = 1e-6
    finite = np.zeros((_L * _N, _L * _N), dtype=np.float64)

    def flat_field(packed: NDArray[np.float64]) -> NDArray[np.float64]:
        return multiplex_field(
            packed.reshape(_L, _N), problem["omega"], problem["intra"], problem["inter"]
        ).ravel()

    for column in range(_L * _N):
        plus = flat.copy()
        minus = flat.copy()
        plus[column] += eps
        minus[column] -= eps
        finite[:, column] = (flat_field(plus) - flat_field(minus)) / (2.0 * eps)
    assert jacobian == pytest.approx(finite, abs=1e-7)


def test_strong_interlayer_coupling_locks_the_replicas() -> None:
    problem = _problem(3)
    strong_inter = 5.0 * np.ones((_L, _L))
    np.fill_diagonal(strong_inter, 0.0)
    trajectory = integrate_multiplex(
        problem["phases"], problem["omega"], problem["intra"], strong_inter, 0.02, 3000
    )
    assert isinstance(trajectory, MultiplexTrajectory)
    assert interlayer_synchronisation(trajectory.terminal_phases) > 0.99


def test_order_parameter_diagnostics_on_constructed_states() -> None:
    # every layer identical → the replicas are perfectly inter-layer synchronised
    single_layer = np.linspace(0.0, 1.0, _N, dtype=np.float64)
    stacked = np.repeat(single_layer[None, :], _L, axis=0)
    assert interlayer_synchronisation(stacked) == pytest.approx(1.0)
    # a fully phase-synchronised layer has order parameter 1
    synchronous = np.zeros((_L, _N), dtype=np.float64)
    assert layer_order_parameters(synchronous) == pytest.approx(np.ones(_L))


def test_trajectory_container() -> None:
    problem = _problem(4)
    trajectory = integrate_multiplex(
        problem["phases"], problem["omega"], problem["intra"], problem["inter"], 0.05, 8
    )
    assert trajectory.phases.shape == (9, _L, _N)
    assert trajectory.times.shape == (9,)
    assert trajectory.times[-1] == pytest.approx(8 * 0.05)
    assert trajectory.terminal_phases == pytest.approx(trajectory.phases[-1])


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("field", {"phases": np.zeros((3, 1))}, "phases must be an"),
        ("field", {"phases": np.zeros(6)}, "phases must be an"),
        ("field", {"omega": np.zeros((_L, _N + 1))}, "omega must have shape"),
        ("field", {"intra": np.zeros((_L, _N, _N + 1))}, "intra_coupling must have shape"),
        ("field", {"inter": np.zeros((_L, _L + 1))}, "inter_coupling must have shape"),
        ("field", {"phases": np.full((_L, _N), np.nan)}, "must be finite"),
        ("integrate", {"dt": 0.0}, "dt must be positive"),
        ("integrate", {"dt": np.inf}, "dt must be positive"),
        ("integrate", {"n_steps": 0}, "n_steps must be positive"),
        ("order", {"phases": np.zeros(5)}, "phases must be an"),
        ("sync", {"phases": np.zeros((2, 2, 2))}, "phases must be an"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    problem = _problem()
    with pytest.raises(ValueError, match=message):
        if call == "field":
            args: dict[str, Any] = {
                "phases": problem["phases"],
                "omega": problem["omega"],
                "intra": problem["intra"],
                "inter": problem["inter"],
            }
            args.update(kwargs)
            multiplex_field(args["phases"], args["omega"], args["intra"], args["inter"])
        elif call == "integrate":
            args = {"dt": 0.05, "n_steps": 10}
            args.update(kwargs)
            integrate_multiplex(
                problem["phases"],
                problem["omega"],
                problem["intra"],
                problem["inter"],
                args["dt"],
                args["n_steps"],
            )
        elif call == "order":
            layer_order_parameters(kwargs["phases"])
        else:
            interlayer_synchronisation(kwargs["phases"])
