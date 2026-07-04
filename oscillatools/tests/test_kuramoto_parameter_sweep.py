# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Parameter-grid sweep and observable-measurement tests
"""Tests for the Kuramoto parameter-grid sweep and observable API.

The correctness argument rests on three anchors. The grid is checked to yield the
exact Cartesian product in row-major order, so a reshaped observable column is the
right phase-diagram array. The physics is checked against the known monotonicity
of the mean-field transition — the time-averaged order parameter rises with the
coupling and the effective-frequency spread falls — so the composed sweep is shown
to measure the real dynamics, not a bookkeeping artefact. Purity is checked by
asserting the input system's state and parameters are byte-for-byte unchanged after
a sweep, since each cell runs on a fresh copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_parameter_sweep import (
    KuramotoParameterGrid,
    Observable,
    ParameterSweepResult,
    frequency_spread,
    mean_order_parameter,
    metastability,
    sweep_parameter_grid,
    terminal_order_parameter,
)
from oscillatools.accel.kuramoto_system import KuramotoSystem


def _mean_field_system() -> KuramotoSystem:
    """A small, deterministic mean-field system for the sweeps."""

    rng = np.random.default_rng(20260702)
    natural_frequencies = rng.normal(0.0, 0.4, size=8)
    initial_phases = rng.uniform(-np.pi, np.pi, size=8)
    return KuramotoSystem.mean_field(
        initial_phases, natural_frequencies, coupling=0.5, dt=0.05, scheme="rk4"
    )


def _networked_system() -> KuramotoSystem:
    """A small ring-coupled networked system for the sweeps."""

    size = 6
    coupling = np.zeros((size, size))
    for node in range(size):
        coupling[node, (node + 1) % size] = 1.0
        coupling[node, (node - 1) % size] = 1.0
    rng = np.random.default_rng(1234)
    natural_frequencies = rng.normal(0.0, 0.2, size=size)
    initial_phases = rng.uniform(-np.pi, np.pi, size=size)
    return KuramotoSystem.networked(
        initial_phases, natural_frequencies, coupling, dt=0.05, scheme="rk4"
    )


# --------------------------------------------------------------------------- grid


def test_grid_rejects_an_empty_specification() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        KuramotoParameterGrid({})


def test_grid_rejects_an_untunable_parameter() -> None:
    with pytest.raises(ValueError, match="unknown Kuramoto parameter"):
        KuramotoParameterGrid({"temperature": [1.0, 2.0]})


def test_grid_rejects_an_empty_axis() -> None:
    with pytest.raises(ValueError, match="axis 'coupling' needs at least one value"):
        KuramotoParameterGrid({"coupling": []})


def test_grid_exposes_axes_shape_and_size() -> None:
    grid = KuramotoParameterGrid({"coupling": [0.5, 1.0, 2.0], "frustration": [0.0, 0.3]})
    assert grid.axis_names == ("coupling", "frustration")
    assert grid.axis_values == ((0.5, 1.0, 2.0), (0.0, 0.3))
    assert grid.shape == (3, 2)
    assert grid.size == 6
    assert len(grid) == 6


def test_grid_points_are_the_row_major_cartesian_product() -> None:
    grid = KuramotoParameterGrid({"coupling": [0.5, 1.0], "frustration": [0.0, 0.3]})
    points = list(grid)
    assert points == [
        {"coupling": 0.5, "frustration": 0.0},
        {"coupling": 0.5, "frustration": 0.3},
        {"coupling": 1.0, "frustration": 0.0},
        {"coupling": 1.0, "frustration": 0.3},
    ]


def test_grid_repr_names_the_axes_and_size() -> None:
    grid = KuramotoParameterGrid({"coupling": [0.5, 1.0, 2.0]})
    assert repr(grid) == "KuramotoParameterGrid(coupling[3], size=3)"


# ---------------------------------------------------------------------- observable


def test_observable_is_callable_on_a_trajectory() -> None:
    trajectory = np.zeros((4, 3))  # a fully synchronised, static trajectory
    observable = mean_order_parameter()
    assert observable.name == "mean_order_parameter"
    assert observable(trajectory, 0.05) == pytest.approx(1.0)


def test_custom_observable_reduces_a_trajectory() -> None:
    # a user-defined measurement: the final-row phase range.
    phase_range = Observable("phase_range", lambda trajectory, dt: float(np.ptp(trajectory[-1])))
    assert phase_range.name == "phase_range"
    assert phase_range(np.array([[0.0, 0.0], [0.0, 1.5]]), 0.05) == pytest.approx(1.5)


def test_terminal_order_parameter_reads_the_final_row() -> None:
    trajectory = np.vstack([np.linspace(0.0, np.pi, 3), np.zeros(3)])
    assert terminal_order_parameter()(trajectory, 0.05) == pytest.approx(1.0)


def test_metastability_is_zero_for_a_static_trajectory() -> None:
    trajectory = np.zeros((5, 4))
    assert metastability()(trajectory, 0.05) == pytest.approx(0.0)


def test_frequency_spread_is_zero_for_a_rigidly_rotating_state() -> None:
    # every oscillator advances by the same increment each step → identical
    # effective frequencies → zero spread.
    steps = np.arange(6).reshape(-1, 1)
    trajectory = steps * 0.1 * np.ones((1, 4))
    assert frequency_spread()(trajectory, 0.05) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- sweep


def test_sweep_requires_at_least_one_observable() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    with pytest.raises(ValueError, match="at least one observable"):
        sweep_parameter_grid(system, grid, [], n_steps=10)


def test_sweep_rejects_duplicate_observable_names() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    with pytest.raises(ValueError, match="distinct"):
        sweep_parameter_grid(
            system, grid, [mean_order_parameter(), mean_order_parameter()], n_steps=10
        )


def test_sweep_rejects_a_non_positive_step_count() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    with pytest.raises(ValueError, match="n_steps must be a positive integer"):
        sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=0)


def test_sweep_rejects_a_negative_transient() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    with pytest.raises(ValueError, match="transient must be non-negative"):
        sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=10, transient=-1)


def test_sweep_rejects_a_transient_that_discards_everything() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    with pytest.raises(ValueError, match="at least one trajectory row"):
        sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=10, transient=11)


def test_sweep_measures_the_full_grid() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.2, 1.0, 4.0]})
    result = sweep_parameter_grid(
        system,
        grid,
        [mean_order_parameter(), frequency_spread()],
        n_steps=400,
        transient=200,
    )
    assert isinstance(result, ParameterSweepResult)
    assert result.observable_names == ("mean_order_parameter", "frequency_spread")
    assert result.measurements.shape == (3, 2)
    assert result.grid_shape == (3,)


def test_mean_order_parameter_rises_with_coupling() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.2, 1.0, 4.0]})
    result = sweep_parameter_grid(
        system, grid, [mean_order_parameter()], n_steps=600, transient=300
    )
    coherence = result.grid_values("mean_order_parameter")
    assert coherence.shape == (3,)
    # the mean-field transition: stronger coupling yields greater coherence.
    assert coherence[0] < coherence[1] < coherence[2]
    assert coherence[2] > 0.9


def test_frequency_spread_falls_with_coupling() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.2, 4.0]})
    result = sweep_parameter_grid(system, grid, [frequency_spread()], n_steps=600, transient=300)
    spread = result.grid_values("frequency_spread")
    # frequency locking: the strongly coupled ensemble spreads less in frequency.
    assert spread[1] < spread[0]


def test_best_selects_the_optimising_cell() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.2, 1.0, 4.0]})
    result = sweep_parameter_grid(
        system, grid, [mean_order_parameter()], n_steps=600, transient=300
    )
    best_point, best_value = result.best("mean_order_parameter")
    assert best_point == {"coupling": 4.0}
    assert best_value == pytest.approx(result.grid_values("mean_order_parameter")[2])
    worst_point, worst_value = result.best("mean_order_parameter", maximise=False)
    assert worst_point == {"coupling": 0.2}
    assert worst_value == pytest.approx(result.grid_values("mean_order_parameter")[0])


def test_records_merge_parameters_and_observables() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.5, 2.0], "frustration": [0.0, 0.4]})
    result = sweep_parameter_grid(
        system, grid, [mean_order_parameter(), metastability()], n_steps=200, transient=100
    )
    records = result.records()
    assert len(records) == 4
    first = records[0]
    assert first["coupling"] == 0.5
    assert first["frustration"] == 0.0
    assert set(first) == {"coupling", "frustration", "mean_order_parameter", "metastability"}
    # the tabulated value matches the grid array at cell (0, 0).
    assert first["mean_order_parameter"] == pytest.approx(
        result.grid_values("mean_order_parameter")[0, 0]
    )


def test_grid_values_rejects_an_unmeasured_observable() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    result = sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=50)
    with pytest.raises(ValueError, match="unknown observable 'metastability'"):
        result.grid_values("metastability")


def test_best_rejects_an_unmeasured_observable() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    result = sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=50)
    with pytest.raises(ValueError, match="unknown observable 'frequency_spread'"):
        result.best("frequency_spread")


def test_sweep_uses_an_explicit_step_when_given() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [1.0]})
    default = sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=100)
    finer = sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=100, dt=0.01)
    # a different step integrates a different amount of time, so the measured
    # coherence differs — evidence the dt override reaches the integrator.
    assert default.measurements[0, 0] != finer.measurements[0, 0]


def test_sweep_does_not_mutate_the_input_system() -> None:
    system = _mean_field_system()
    state_before = system.current_state
    coupling_before = system.current_parameters.coupling
    time_before = system.current_time
    grid = KuramotoParameterGrid({"coupling": [0.5, 2.0], "frustration": [0.0, 0.4]})
    sweep_parameter_grid(system, grid, [mean_order_parameter()], n_steps=100, transient=20)
    assert np.array_equal(system.current_state, state_before)
    assert system.current_parameters.coupling == coupling_before
    assert system.current_time == time_before


def test_sweep_runs_on_a_networked_system() -> None:
    system = _networked_system()
    grid = KuramotoParameterGrid({"frustration": [0.0, 0.3]})
    result = sweep_parameter_grid(
        system, grid, [mean_order_parameter(), metastability()], n_steps=300, transient=150
    )
    assert result.measurements.shape == (2, 2)
    assert np.all(np.isfinite(result.measurements))


def test_two_parameter_sweep_reshapes_to_the_grid() -> None:
    system = _mean_field_system()
    grid = KuramotoParameterGrid({"coupling": [0.5, 2.0, 4.0], "frustration": [0.0, 0.5]})
    result = sweep_parameter_grid(
        system, grid, [mean_order_parameter()], n_steps=200, transient=100
    )
    values = result.grid_values("mean_order_parameter")
    assert values.shape == (3, 2)
    # the reshaped array agrees cell-by-cell with the flat records.
    for cell, record in enumerate(result.records()):
        row, column = divmod(cell, 2)
        assert record["mean_order_parameter"] == pytest.approx(values[row, column])
