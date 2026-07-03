# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the neural-operator cost model
"""Contract tests for the host-independent surrogate-versus-direct operation-count model.

Every count is fixed by an explicit formula, so the tests assert exact integer values (no tolerance),
exercise each validation branch, and pin the two amortisation outcomes (a crossover exists / never).
The arithmetic here is pure and torch-free, so it runs in the ordinary CPU-only lane.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from scpn_quantum_control.forecasting.neural_operator_cost_model import (
    FORCE_FLOPS_PER_OSCILLATOR_PAIR,
    MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE,
    RK4_COMBINATION_FLOPS_PER_OSCILLATOR,
    RK4_STAGES_PER_STEP,
    TRAINING_FORWARD_BACKWARD_FACTOR,
    SurrogateCostModel,
    amortised_break_even_queries,
    build_cost_model,
    deeponet_forward_flops,
    direct_simulation_flops,
    networked_force_flops,
    rk4_right_hand_side_evaluations,
    rk4_step_flops,
    training_flops,
)


def _expected_forward_flops(n: int, latent: int, hidden: int) -> int:
    channels = 2 * n
    mac = MATMUL_FLOPS_PER_MULTIPLY_ACCUMULATE
    branch = mac * channels * hidden + hidden + mac * hidden * channels * latent
    trunk = mac * 1 * hidden + hidden + mac * hidden * latent
    contraction = mac * channels * latent
    return branch + trunk + contraction + channels


def test_rhs_evaluations_is_four_stages_per_step() -> None:
    assert rk4_right_hand_side_evaluations(20) == RK4_STAGES_PER_STEP * 20
    assert rk4_right_hand_side_evaluations(1) == 4


@pytest.mark.parametrize("n", [1, 2, 3, 8, 32])
def test_networked_force_flops_matches_formula(n: int) -> None:
    assert networked_force_flops(n) == FORCE_FLOPS_PER_OSCILLATOR_PAIR * n * n + n


@pytest.mark.parametrize("n", [1, 3, 16])
def test_rk4_step_flops_is_four_forces_plus_combination(n: int) -> None:
    expected = (
        RK4_STAGES_PER_STEP * networked_force_flops(n) + RK4_COMBINATION_FLOPS_PER_OSCILLATOR * n
    )
    assert rk4_step_flops(n) == expected


def test_direct_simulation_flops_scales_linearly_in_steps() -> None:
    n, steps = 5, 7
    assert direct_simulation_flops(n, steps) == steps * rk4_step_flops(n)
    # linear in the number of steps
    assert direct_simulation_flops(n, 2 * steps) == 2 * direct_simulation_flops(n, steps)


@pytest.mark.parametrize(
    ("n", "latent", "hidden"),
    [(2, 3, 5), (4, 8, 16), (32, 32, 96)],
)
def test_deeponet_forward_flops_matches_explicit_model(n: int, latent: int, hidden: int) -> None:
    assert deeponet_forward_flops(n, latent, hidden) == _expected_forward_flops(n, latent, hidden)


def test_forward_flops_grow_linearly_in_oscillators() -> None:
    # doubling N (with fixed latent/hidden) roughly doubles the dominant branch term; the surrogate is
    # linear in N whereas an RK4 step is quadratic — the source of the asymptotic crossover.
    small = deeponet_forward_flops(8, 16, 32)
    large = deeponet_forward_flops(16, 16, 32)
    step_small = rk4_step_flops(8)
    step_large = rk4_step_flops(16)
    assert large / small < step_large / step_small


def test_training_flops_is_dataset_plus_optimisation() -> None:
    n, steps, traj, epochs, latent, hidden = 4, 6, 10, 5, 8, 16
    dataset = traj * steps * rk4_step_flops(n)
    samples = traj * (steps + 1)
    optimisation = (
        epochs
        * samples
        * TRAINING_FORWARD_BACKWARD_FACTOR
        * deeponet_forward_flops(n, latent, hidden)
    )
    assert (
        training_flops(
            n,
            n_steps=steps,
            n_trajectories=traj,
            epochs=epochs,
            latent_dim=latent,
            hidden_dim=hidden,
        )
        == dataset + optimisation
    )


def test_break_even_exists_when_surrogate_cheaper_per_query() -> None:
    # train=1000, direct=100, surrogate=10 => margin 90 => floor(1000/90)+1 = 12
    assert amortised_break_even_queries(1000, 100, 10) == 12
    # the returned count is the first strictly-cheaper query
    break_even = amortised_break_even_queries(1000, 100, 10)
    assert 1000 + break_even * 10 < break_even * 100
    assert 1000 + (break_even - 1) * 10 >= (break_even - 1) * 100


def test_break_even_is_none_when_surrogate_not_cheaper() -> None:
    assert amortised_break_even_queries(1000, 10, 10) is None
    assert amortised_break_even_queries(1000, 5, 10) is None


def test_break_even_with_zero_training_is_first_query() -> None:
    assert amortised_break_even_queries(0, 100, 10) == 1


@pytest.mark.parametrize(
    "call",
    [
        lambda: rk4_right_hand_side_evaluations(0),
        lambda: networked_force_flops(0),
        lambda: rk4_step_flops(0),
        lambda: direct_simulation_flops(0, 5),
        lambda: direct_simulation_flops(3, 0),
        lambda: deeponet_forward_flops(0, 3, 5),
        lambda: deeponet_forward_flops(3, 0, 5),
        lambda: deeponet_forward_flops(3, 3, 0),
        lambda: training_flops(
            0, n_steps=5, n_trajectories=2, epochs=2, latent_dim=3, hidden_dim=5
        ),
        lambda: training_flops(
            3, n_steps=0, n_trajectories=2, epochs=2, latent_dim=3, hidden_dim=5
        ),
        lambda: training_flops(
            3, n_steps=5, n_trajectories=0, epochs=2, latent_dim=3, hidden_dim=5
        ),
        lambda: training_flops(
            3, n_steps=5, n_trajectories=2, epochs=0, latent_dim=3, hidden_dim=5
        ),
    ],
)
def test_non_positive_arguments_are_rejected(call: Callable[[], int]) -> None:
    with pytest.raises(ValueError):
        call()


def test_break_even_rejects_negative_arguments() -> None:
    with pytest.raises(ValueError, match="training_flops_total"):
        amortised_break_even_queries(-1, 100, 10)
    with pytest.raises(ValueError, match="per-query"):
        amortised_break_even_queries(1000, -1, 10)
    with pytest.raises(ValueError, match="per-query"):
        amortised_break_even_queries(1000, 100, -1)


def test_build_cost_model_assembles_every_field() -> None:
    model = build_cost_model(
        24, n_steps=20, latent_dim=32, hidden_dim=96, n_trajectories=160, epochs=250
    )
    assert isinstance(model, SurrogateCostModel)
    assert model.rk4_right_hand_side_evaluations == 80
    assert model.direct_flops_per_query == direct_simulation_flops(24, 20)
    assert model.surrogate_flops_per_query == deeponet_forward_flops(24, 32, 96)
    assert model.per_query_flop_ratio == pytest.approx(
        model.direct_flops_per_query / model.surrogate_flops_per_query
    )
    assert model.training_flops == training_flops(
        24, n_steps=20, n_trajectories=160, epochs=250, latent_dim=32, hidden_dim=96
    )
    assert model.break_even_queries == amortised_break_even_queries(
        model.training_flops, model.direct_flops_per_query, model.surrogate_flops_per_query
    )


def test_cost_model_to_dict_round_trips_keys() -> None:
    model = build_cost_model(
        8, n_steps=10, latent_dim=16, hidden_dim=32, n_trajectories=20, epochs=30
    )
    payload = model.to_dict()
    assert set(payload) == {
        "n_oscillators",
        "n_steps",
        "latent_dim",
        "hidden_dim",
        "n_trajectories",
        "epochs",
        "rk4_right_hand_side_evaluations",
        "direct_flops_per_query",
        "surrogate_flops_per_query",
        "per_query_flop_ratio",
        "training_flops",
        "break_even_queries",
    }
    assert payload["n_oscillators"] == 8
    assert isinstance(payload["per_query_flop_ratio"], float)
    assert payload["break_even_queries"] is None or isinstance(payload["break_even_queries"], int)


def test_per_query_ratio_grows_with_horizon_and_size() -> None:
    # the per-query FLOP advantage strengthens as either the horizon or the network grows
    base = build_cost_model(
        16, n_steps=20, latent_dim=16, hidden_dim=32, n_trajectories=10, epochs=10
    )
    longer = build_cost_model(
        16, n_steps=80, latent_dim=16, hidden_dim=32, n_trajectories=10, epochs=10
    )
    larger = build_cost_model(
        48, n_steps=20, latent_dim=16, hidden_dim=32, n_trajectories=10, epochs=10
    )
    assert longer.per_query_flop_ratio > base.per_query_flop_ratio
    assert larger.per_query_flop_ratio > base.per_query_flop_ratio


def test_break_even_is_a_positive_integer_when_present() -> None:
    model = build_cost_model(
        32, n_steps=40, latent_dim=32, hidden_dim=96, n_trajectories=200, epochs=300
    )
    assert model.break_even_queries is not None
    assert isinstance(model.break_even_queries, int)
    assert model.break_even_queries >= 1
    assert not math.isnan(model.per_query_flop_ratio)
